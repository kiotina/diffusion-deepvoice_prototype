"""WAV 이상 점수, real validation calibration, 파일 단위 평가 진입점."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from deepvoice_diffusion.calibration import (  # noqa: E402
    calibration_sources, check_threshold, create_threshold, load_threshold, save_threshold,
)
from deepvoice_diffusion.config import load_config  # noqa: E402
from deepvoice_diffusion.data_contract import (  # noqa: E402
    contract_path, sha256_file, verify_processed_data,
)
from deepvoice_diffusion.evaluation import (  # noqa: E402
    compute_metrics, score_entries, write_csv_new, write_json_new,
)
from deepvoice_diffusion.evaluation_config import load_evaluation_config, override_paths  # noqa: E402
from deepvoice_diffusion.evaluation_data import (  # noqa: E402
    build_inventory, read_inventory, validate_inventory, write_inventory,
)
from deepvoice_diffusion.scoring import ScoreEngine  # noqa: E402
from deepvoice_diffusion.training_data import read_manifest  # noqa: E402


PREDICTION_FIELDS = ["purpose", "file_id", "path", "source", "split", "label", "label_status", "status",
                     "score", "max_score", "segment_count", "predicted_label", "decision", "error"]


def _file_id(path: Path) -> str:
    return hashlib.sha256(str(path).casefold().encode("utf-8")).hexdigest()[:20]


def _run(engine, purpose, files, threshold):
    return {"purpose": purpose, "binding": engine.binding, "binding_hash": engine.binding_hash,
            "threshold": threshold, "file_count": files,
            "execution": {"device": "cpu", "batch_size": engine.config.batch_size,
                          "cpu_threads": engine.config.cpu_threads,
                          "quantile": engine.config.quantile,
                          "minimum_calibration_files": engine.config.minimum_calibration_files},
            "checkpoint_state": engine.payload.get("state", {}),
            "interpretation": "functional only" if purpose == "functional_smoke" else "anomaly score, not probability"}


def _write_predictions(output_dir, predictions, segments):
    write_csv_new(output_dir / "predictions.csv", predictions, PREDICTION_FIELDS)
    segment_fields = ["purpose", "file_id", "path", "segment_index", "start_seconds", "end_seconds",
                      "valid_seconds", "score"]
    if segments:
        segment_fields += [key for key in segments[0] if key.startswith("t") and key[1:].isdigit()]
    write_csv_new(output_dir / "segments.csv", segments, segment_fields)


def _smoke_inventory(config):
    rows = read_manifest(config.manifest_path)
    validation = sorted({str(Path(row["source_path"]).resolve()) for row in rows
                         if row["split"] == "validation"})[:6]
    external = sorted(path.resolve() for path in config.external_dir.rglob("*")
                      if path.is_file() and path.suffix.lower() == ".wav")[:2]
    if len(validation) != 6 or len(external) != 2:
        raise ValueError("Smoke needs six real validation and two external WAV files")
    inventory = []
    for name in validation:
        path = Path(name)
        inventory.append(dict(purpose="functional_smoke", file_id=_file_id(path), path=str(path), source="real_manifest",
                              split="validation", declared_label="0", label="0",
                              label_status="verified", evidence="existing_real_manifest",
                              sha256=sha256_file(path)))
    for path in external:
        inventory.append(dict(purpose="functional_smoke", file_id=_file_id(path), path=str(path), source="external",
                              split="external", declared_label="1", label="", label_status="unverified",
                              evidence="", sha256=sha256_file(path)))
    validate_inventory(inventory)
    return inventory


def main():
    parser = argparse.ArgumentParser(description="WAV diffusion anomaly scoring and evaluation")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/evaluate.yaml")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--inventory", type=Path, help="Prepared inventory.csv; defaults to output directory")
    parser.add_argument("--threshold", type=Path, help="Threshold JSON; defaults to output directory")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Prepare source inventory; optionally verify all saved train/validation arrays")
    prepare.add_argument("--verify-contract", action="store_true")
    commands.add_parser("calibrate", help="Calibrate from all real validation source WAVs")
    predict = commands.add_parser("predict", help="Score explicitly supplied WAVs")
    predict.add_argument("--wav", type=Path, action="append", required=True)
    commands.add_parser("evaluate", help="Score held-out real test and external WAVs")
    commands.add_parser("smoke", help="Four calibration + two distinct validation + two external functional check")
    args = parser.parse_args()
    config = override_paths(load_evaluation_config(args.config), checkpoint=args.checkpoint,
                            output_dir=args.output_dir)
    if args.command == "smoke" and args.output_dir is None:
        config = replace(config, output_dir=config.output_dir / "functional_smoke")
    output = config.output_dir
    inventory_path = args.inventory.resolve() if args.inventory else output / "inventory.csv"
    threshold_path = args.threshold.resolve() if args.threshold else output / "threshold.json"

    if args.command == "prepare":
        if inventory_path.exists():
            raise FileExistsError(f"Inventory already exists: {inventory_path}")
        inventory = build_inventory(config.manifest_path, config.external_dir, config.verified_fake_manifest)
        if args.verify_contract:
            destination = contract_path(config.manifest_path)
            if destination.exists():
                raise FileExistsError(f"Contract already exists: {destination}")
            contract = verify_processed_data(config.manifest_path, load_config(config.preprocess_config))
            write_json_new(destination, contract)
        output.mkdir(parents=True, exist_ok=True)
        write_inventory(inventory_path, inventory)
        print(json.dumps({"inventory": str(inventory_path), "files": len(inventory),
                          "verified_fake": sum(row["label"] == "1" for row in inventory)}, ensure_ascii=False))
        return

    if args.command == "smoke":
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Smoke output is not empty: {output}")
        engine = ScoreEngine(config, legacy_smoke=True)
        inventory = _smoke_inventory(config)
        selected = calibration_sources(inventory)[:4]
        artifact, _ = create_threshold(engine, inventory, purpose="functional_smoke", selected=selected)
        check_threshold(artifact, engine, inventory, allow_smoke=True)
        selected_ids = {row["file_id"] for row in selected}
        targets = [row for row in inventory if row["file_id"] not in selected_ids]
        predictions, segments = score_entries(engine, targets, artifact["threshold"])
        output.mkdir(parents=True, exist_ok=True)
        write_inventory(inventory_path, inventory)
        save_threshold(threshold_path, artifact)
        _write_predictions(output, predictions, segments)
        write_json_new(output / "metrics.json", {"purpose": "functional_smoke", "fake_detection_metrics": None,
                                                     "reason": "Functional smoke does not estimate detection performance"})
        write_json_new(output / "run.json", _run(engine, "functional_smoke", len(targets), artifact["threshold"]))
        print(json.dumps({"purpose": "functional_smoke", "output_dir": str(output),
                          "calibration_valid": artifact["calibration_valid"],
                          "scored": sum(row["status"] == "scored" for row in predictions)}, ensure_ascii=False))
        return

    engine = ScoreEngine(config)
    inventory = read_inventory(inventory_path, check_files=True)
    if args.command == "calibrate":
        if threshold_path.exists():
            raise FileExistsError(f"Threshold already exists: {threshold_path}")
        artifact, _ = create_threshold(engine, inventory)
        threshold_path.parent.mkdir(parents=True, exist_ok=True)
        save_threshold(threshold_path, artifact)
        print(json.dumps({"threshold": artifact["threshold"],
                          "valid_files": artifact["calibration_valid"],
                          "observed_exceedance_rate": artifact["calibration_exceedance_rate"]}, ensure_ascii=False))
        return

    artifact = load_threshold(threshold_path)
    check_threshold(artifact, engine, inventory)
    for name in ("predictions.csv", "segments.csv", "run.json"):
        if (output / name).exists():
            raise FileExistsError(f"Output already exists: {output / name}")
    if args.command == "evaluate":
        if (output / "metrics.json").exists():
            raise FileExistsError(f"Output already exists: {output / 'metrics.json'}")
        targets = [row for row in inventory if row["split"] in {"test", "external"}]
    else:
        targets = []
        for candidate in args.wav:
            path = candidate.resolve()
            targets.append(dict(file_id=_file_id(path), path=str(path), source="user_input",
                                split="user_input", label="", label_status="unverified"))
        if len({row["path"].casefold() for row in targets}) != len(targets):
            raise ValueError("Repeated input WAV")
    predictions, segments = score_entries(engine, targets, artifact["threshold"])
    output.mkdir(parents=True, exist_ok=True)
    _write_predictions(output, predictions, segments)
    if args.command == "evaluate":
        write_json_new(output / "metrics.json", compute_metrics(predictions))
    write_json_new(output / "run.json", _run(engine, args.command, len(targets), artifact["threshold"]))
    print(json.dumps({"purpose": args.command, "output_dir": str(output),
                      "scored": sum(row["status"] == "scored" for row in predictions),
                      "excluded": sum(row["status"].startswith("excluded_") for row in predictions),
                      "errors": sum(row["status"] == "error" for row in predictions)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
