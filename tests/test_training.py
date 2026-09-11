from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from deepvoice_diffusion.checkpoint import load_checkpoint
from deepvoice_diffusion.diffusion import DiffusionSchedule
from deepvoice_diffusion.model import NoisePredictorUNet
from deepvoice_diffusion.training import evaluate, fit
from deepvoice_diffusion.training_config import TrainingConfig
from deepvoice_diffusion.training_data import MelDataset, read_manifest
from test_training_data import make_manifest


def test_mid_epoch_resume_matches_uninterrupted_training(tmp_path):
    manifest = make_manifest(tmp_path)
    config = TrainingConfig(manifest_path=str(manifest), output_dir=str(tmp_path / "continuous"),
                            base_channels=8, time_dim=16, batch_size=1, epochs=2, cpu_threads=1,
                            checkpoint_every=1)
    full = fit(config)
    resumed_config = replace(config, output_dir=str(tmp_path / "resumed"))
    partial = fit(resumed_config, max_steps=1)
    assert partial["next_batch"] == 1 and partial["completed_epochs"] == 0
    checkpoint = tmp_path / "resumed/last.pt"
    finished = fit(resumed_config, resume=checkpoint)
    assert finished["total_steps"] == full["total_steps"] == 4
    expected = load_checkpoint(tmp_path / "continuous/last.pt")
    actual = load_checkpoint(checkpoint)
    for key in expected["model"]:
        torch.testing.assert_close(expected["model"][key], actual["model"][key], rtol=0, atol=0)
    assert expected["state"] == actual["state"]
    best = load_checkpoint(tmp_path / "resumed/best.pt")
    assert best["state"]["best_loss"] == min(r["validation_loss"] for r in actual["state"]["history"])
    with pytest.raises(FileExistsError):
        fit(resumed_config)
    with pytest.raises(ValueError, match="configuration mismatch"):
        fit(replace(resumed_config, learning_rate=0.001), resume=checkpoint)
    rows = read_manifest(manifest)
    array = np.load(rows[0]["mel_path"])
    array[..., 0] = 0.5
    np.save(rows[0]["mel_path"], array)
    with pytest.raises(ValueError, match="data changed"):
        fit(resumed_config, resume=checkpoint)


def test_early_stopping_and_validation_repeatability(tmp_path):
    manifest = make_manifest(tmp_path)
    rows = read_manifest(manifest)
    # Test 배열이 없어도 학습/검증은 가능해야 한다.
    for row in rows:
        if row["split"] == "test":
            Path(row["mel_path"]).unlink()
            Path(row["mask_path"]).unlink()
    # 유효 길이가 다른 배치들도 전체 유효 원소 수로 loss를 집계한다.
    mask = np.load(rows[0]["mask_path"])
    mask[..., 65:] = 0
    np.save(rows[0]["mask_path"], mask)
    # 반올림으로 실질적인 업데이트가 없는 lr: 검증값이 개선되지 않으면 중단해야 한다.
    config = TrainingConfig(manifest_path=str(manifest), output_dir=str(tmp_path / "stopping"),
                            base_channels=8, time_dim=16, batch_size=1, learning_rate=1e-30,
                            epochs=10, patience=1, cpu_threads=1)
    report = fit(config)
    assert report["stop_reason"] == "early_stopping"
    assert report["completed_epochs"] == 2
    rows = read_manifest(manifest)
    dataset = MelDataset(rows, "train")
    model = NoisePredictorUNet(8, 16)
    schedule = DiffusionSchedule()
    first = evaluate(model, dataset, schedule, config, torch.device("cpu"))
    second = evaluate(model, dataset, schedule, config, torch.device("cpu"))
    assert first == second
    batched = evaluate(model, dataset, schedule, replace(config, batch_size=2), torch.device("cpu"))
    assert batched == pytest.approx(first, rel=1e-6)


@pytest.mark.parametrize("change", [{"batch_size": 0}, {"learning_rate": float("nan")},
                                    {"epochs": -1}, {"time_dim": 3}, {"beta_end": 1}])
def test_invalid_training_configuration(change):
    with pytest.raises(ValueError):
        TrainingConfig(**change)
