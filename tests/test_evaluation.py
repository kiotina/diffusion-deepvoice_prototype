import csv
import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from deepvoice_diffusion.audio import load_waveform, make_training_segments, segment_to_logmel
from deepvoice_diffusion.calibration import check_threshold, create_threshold
from deepvoice_diffusion.checkpoint import load_checkpoint, save_checkpoint
from deepvoice_diffusion.config import load_config
from deepvoice_diffusion.data_contract import load_verified_contract, verify_processed_data
from deepvoice_diffusion.diffusion import masked_mse_per_sample
from deepvoice_diffusion.evaluation import classify, compute_metrics
from deepvoice_diffusion.evaluation_config import EvaluationConfig
from deepvoice_diffusion.evaluation_data import (
    build_inventory,
    validate_inventory,
    validate_inventory_against_manifest,
)
from deepvoice_diffusion.model import NoisePredictorUNet
from deepvoice_diffusion.scoring import ScoreEngine
from deepvoice_diffusion.training import fit
from deepvoice_diffusion.training_config import TrainingConfig


PROJECT = Path(__file__).resolve().parents[1]
PREPROCESS = PROJECT / "configs/preprocess.yaml"


def wave(path, seconds=2.0, sample_rate=16000, frequency=220):
    t = np.arange(round(seconds * sample_rate), dtype=np.float32) / sample_rate
    samples = (0.3 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
    sf.write(path, samples, sample_rate)
    return path


def tiny_engine(tmp_path, *, seconds=2.0, batch_size=8):
    source = wave(tmp_path / "source.wav", seconds=seconds)
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("split,source_path,segment_index,mel_path,mask_path\nvalidation,"
                        f"{source},0,missing.npy,missing_mask.npy\n", encoding="utf-8")
    model = NoisePredictorUNet(8, 16)
    optimizer = torch.optim.AdamW(model.parameters())
    checkpoint = tmp_path / "legacy.pt"
    save_checkpoint(checkpoint, model, optimizer, {},
                    {"base_channels": 8, "time_dim": 16, "timesteps": 1000,
                     "beta_start": 0.0001, "beta_end": 0.02}, {})
    config = EvaluationConfig(manifest_path=manifest, preprocess_config=PREPROCESS,
                              external_dir=tmp_path, checkpoint=checkpoint,
                              output_dir=tmp_path / "output", timesteps=(100, 300),
                              noise_seeds=(1729, 2718), batch_size=batch_size, cpu_threads=1)
    return ScoreEngine(config, legacy_smoke=True), source


def test_masked_mse_is_per_segment_and_excludes_padding():
    target = torch.zeros((2, 1, 2, 3))
    predicted = torch.tensor([[[[1., 2., 99.], [1., 2., 99.]]],
                              [[[3., 4., 99.], [3., 4., 99.]]]])
    mask = torch.tensor([[[[1., 1., 0.]]], [[[1., 1., 0.]]]])
    result = masked_mse_per_sample(predicted, target, mask)
    torch.testing.assert_close(result, torch.tensor([2.5, 12.5]))
    with pytest.raises(ValueError, match="Each sample"):
        masked_mse_per_sample(predicted, target, mask * 0)


def test_scoring_reuses_noise_across_names_calls_and_batch_sizes(tmp_path):
    torch.manual_seed(7)
    engine, source = tiny_engine(tmp_path, seconds=3.2)
    renamed = tmp_path / "renamed.wav"
    renamed.write_bytes(source.read_bytes())
    first, segments = engine.score_file(source)
    again, _ = engine.score_file(source)
    other, _ = engine.score_file(renamed)
    small_engine = ScoreEngine(replace(engine.config, batch_size=1), legacy_smoke=True)
    small, _ = small_engine.score_file(source)
    assert first["segment_count"] == 3
    assert segments[-1]["end_seconds"] == pytest.approx(3.2, abs=1 / 16000)
    assert first["score"] == pytest.approx(np.mean([row["score"] for row in segments]))
    assert first["max_score"] == max(row["score"] for row in segments)
    assert first == again == other
    assert small["score"] == pytest.approx(first["score"], rel=1e-5, abs=1e-6)
    for row in segments:
        assert set(row) >= {"t100", "t300"}


def test_v1_requires_explicit_smoke_mode(tmp_path):
    engine, _ = tiny_engine(tmp_path)
    with pytest.raises(ValueError, match="legacy functional smoke"):
        ScoreEngine(engine.config)


def test_wav_length_resampling_zero_and_nonfinite_model(tmp_path):
    engine, _ = tiny_engine(tmp_path)
    one_second = wave(tmp_path / "one.wav", seconds=1)
    short = wave(tmp_path / "short.wav", seconds=0.999)
    resampled = wave(tmp_path / "resampled.wav", seconds=2, sample_rate=48000)
    silent = tmp_path / "silent.wav"
    sf.write(silent, np.zeros(16000, dtype=np.float32), 16000)
    assert engine.score_file(one_second)[0]["status"] == "scored"
    assert engine.score_file(short)[0]["status"] == "excluded_short"
    assert len(load_waveform(resampled, engine.preprocess.audio)) == 32000
    assert engine.score_file(resampled)[0]["status"] == "scored"
    assert engine.score_file(silent)[0]["status"] == "excluded_zero_signal"
    broken = tmp_path / "broken.wav"
    broken.write_bytes(b"not audio")
    with pytest.raises(Exception):
        engine.score_file(broken)
    engine.model.forward = lambda noisy, steps, mask: torch.full_like(noisy, float("nan"))
    with pytest.raises(FloatingPointError, match="Non-finite"):
        engine.score_file(one_second)


class StubEngine:
    def __init__(self, scores, manifest_path):
        self.scores = scores
        self.binding = {"model": "one"}
        self.binding_hash = "one"
        self.config = type("Config", (), {
            "quantile": 0.95,
            "minimum_calibration_files": 20,
            "manifest_path": manifest_path,
        })()

    def score_file(self, path):
        return {"status": "scored", "score": self.scores[str(path)]}, []


def inventory_entry(index, *, split="validation", source="real_manifest", label="0", sha=None):
    return {"file_id": str(index), "path": f"file_{index}.wav", "source": source, "split": split,
            "declared_label": "0" if source == "real_manifest" else "1", "label": label,
            "label_status": "verified" if label else "unverified",
            "evidence": "manifest" if label else "", "sha256": sha or str(index)}


def test_calibration_quantile_tie_binding_and_smoke_rejection(tmp_path):
    inventory = [inventory_entry(i) for i in range(20)]
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "split,source_path,segment_index,mel_path,mask_path\n" + "".join(
            f"validation,{row['path']},0,mel_{index}.npy,mask_{index}.npy\n"
            for index, row in enumerate(inventory)
        ),
        encoding="utf-8",
    )
    scores = {row["path"]: float(i) for i, row in enumerate(inventory)}
    engine = StubEngine(scores, manifest)
    artifact, measured = create_threshold(engine, inventory)
    assert len(measured) == 20
    assert artifact["threshold"] == 18.0
    assert artifact["calibration_exceedance_rate"] == pytest.approx(0.05)
    assert classify(18.0, 18.0) == (0, "기준 범위 내")
    assert classify(19.0, 18.0)[0] == 1
    check_threshold(artifact, engine, inventory)
    engine.binding_hash = "other"
    with pytest.raises(ValueError, match="binding"):
        check_threshold(artifact, engine, inventory)
    engine.binding_hash = "one"
    smoke, _ = create_threshold(engine, inventory, purpose="functional_smoke", selected=inventory[:4])
    with pytest.raises(ValueError, match="Functional smoke"):
        check_threshold(smoke, engine, inventory)
    with pytest.raises(ValueError, match="every real validation"):
        create_threshold(engine, inventory, selected=inventory[:4])
    engine.score_file = lambda path: (_ for _ in ()).throw(OSError("unreadable WAV"))
    with pytest.raises(OSError, match="unreadable WAV"):
        create_threshold(engine, inventory)


def test_inventory_duplicates_and_unverified_labels():
    rows = [inventory_entry(1), inventory_entry(2, split="external", source="external", label="")]
    validate_inventory(rows)
    assert rows[1]["label"] == "" and rows[1]["label_status"] == "unverified"
    with pytest.raises(ValueError, match="Duplicate content"):
        validate_inventory([rows[0], {**rows[1], "sha256": rows[0]["sha256"]}])
    with pytest.raises(ValueError, match="Duplicate inventory"):
        validate_inventory([rows[0], {**rows[1], "path": rows[0]["path"]}])
    with pytest.raises(ValueError, match="evidence"):
        validate_inventory([rows[0], {**rows[1], "label": "1", "label_status": "verified"}])


def test_inventory_real_split_must_match_manifest(tmp_path):
    source = tmp_path / "real.wav"
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "split,source_path,segment_index,mel_path,mask_path\n"
        f"test,{source},0,mel.npy,mask.npy\n",
        encoding="utf-8",
    )
    row = inventory_entry(1, split="validation")
    row["path"] = str(source)
    with pytest.raises(ValueError, match="split differs"):
        validate_inventory_against_manifest([row], manifest)


def test_external_folder_name_does_not_create_fake_truth(tmp_path):
    real = wave(tmp_path / "real.wav", frequency=220)
    external_dir = tmp_path / "fake"
    external_dir.mkdir()
    external = wave(external_dir / "candidate.wav", frequency=550)
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("split,source_path,segment_index,mel_path,mask_path\n"
                        f"test,{real},0,unused.npy,unused_mask.npy\n", encoding="utf-8")
    unknown = build_inventory(manifest, external_dir)
    assert unknown[1]["declared_label"] == "1"
    assert unknown[1]["label"] == "" and unknown[1]["label_status"] == "unverified"
    evidence = tmp_path / "verified.csv"
    evidence.write_text(f"path,evidence\n{external},synthetic_generation_log\n", encoding="utf-8")
    confirmed = build_inventory(manifest, external_dir, evidence)
    assert confirmed[1]["label"] == "1" and confirmed[1]["label_status"] == "verified"


def test_metrics_known_matrix_and_missing_fake():
    rows = []
    for label, prediction, score in [(0, 0, 0.1), (0, 1, 0.8), (1, 0, 0.2), (1, 1, 0.9)]:
        rows.append({"label": str(label), "label_status": "verified", "status": "scored",
                     "predicted_label": prediction, "score": score})
    rows.append({"label": "", "label_status": "unverified", "status": "scored",
                 "predicted_label": 1, "score": 100})
    metric = compute_metrics(rows)
    assert metric["confusion_matrix"] == [[1, 1], [1, 1]]
    assert metric["accuracy"] == metric["precision"] == metric["recall"] == metric["f1"] == 0.5
    assert metric["real_fpr"] == metric["specificity"] == metric["balanced_accuracy"] == 0.5
    if importlib.util.find_spec("sklearn") is None:
        assert metric["roc_auc"] is None and "scikit-learn" in metric["roc_auc_reason"]
    else:
        assert metric["roc_auc"] == pytest.approx(0.75)
    assert metric["counts"]["unverified_scored"] == 1
    real_only = compute_metrics(rows[:2])
    assert real_only["accuracy"] is None and real_only["roc_auc"] is None
    assert real_only["real_fpr"] == 0.5


def test_full_reproduction_contract_and_v2_checkpoint(tmp_path):
    config = load_config(PREPROCESS)
    rows = []
    for split, frequency in (("train", 220), ("validation", 330)):
        source = wave(tmp_path / f"{split}.wav", seconds=1.1, frequency=frequency)
        segment = make_training_segments(load_waveform(source, config.audio), config.audio)[0]
        mel, mask = segment_to_logmel(segment, config.audio, config.mel)
        mel_path, mask_path = tmp_path / f"{split}_mel.npy", tmp_path / f"{split}_mask.npy"
        np.save(mel_path, mel)
        np.save(mask_path, mask)
        rows.append({"split": split, "source_path": str(source), "segment_index": "0",
                     "mel_path": str(mel_path), "mask_path": str(mask_path)})
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    contract = verify_processed_data(manifest, config)
    assert contract["verified_segments"] == 2
    (tmp_path / "preprocess_contract.json").write_text(json.dumps(contract), encoding="utf-8")
    assert load_verified_contract(manifest) == contract
    model = NoisePredictorUNet(8, 16)
    checkpoint = tmp_path / "v2.pt"
    save_checkpoint(checkpoint, model, torch.optim.AdamW(model.parameters()), {},
                    {"base_channels": 8, "time_dim": 16, "timesteps": 1000,
                     "beta_start": 0.0001, "beta_end": 0.02}, {}, contract)
    assert load_checkpoint(checkpoint)["format_version"] == 2
    engine = ScoreEngine(EvaluationConfig(manifest_path=manifest, preprocess_config=PREPROCESS,
                                          external_dir=tmp_path, checkpoint=checkpoint,
                                          output_dir=tmp_path / "scores", timesteps=(100,),
                                          noise_seeds=(1729,), cpu_threads=1))
    assert engine.score_file(rows[1]["source_path"])[0]["status"] == "scored"
    training = TrainingConfig(manifest_path=str(manifest), output_dir=str(tmp_path / "training"),
                              base_channels=8, time_dim=16, batch_size=1, epochs=1,
                              checkpoint_every=1, cpu_threads=1)
    fit(training)
    assert load_checkpoint(tmp_path / "training/best.pt")["format_version"] == 2
    np.save(rows[0]["mel_path"], np.zeros((1, 80, 126), dtype=np.float32))
    with pytest.raises(ValueError, match="arrays changed"):
        load_verified_contract(manifest)
