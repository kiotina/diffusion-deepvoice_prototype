"""저장된 모델 점수에서 임시 외부 라벨로 개발/시험 분리 임계값 실험."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from deepvoice_diffusion.evaluation import classify, compute_metrics, write_csv_new, write_json_new  # noqa: E402
from deepvoice_diffusion.scoring import canonical_hash  # noqa: E402


def split_external_files(rows: list[dict], *, seed: int = 42) -> tuple[list[dict], list[dict]]:
    """파일 ID의 고정 해시로 외부 파일을 1:1 개발/시험 분할한다."""
    if len(rows) < 2:
        raise ValueError("Need at least two external files")
    ids = [row["file_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("External file IDs must be unique")
    ordered = sorted(rows, key=lambda row: hashlib.sha256(
        f"{seed}:{row['file_id']}".encode("utf-8")).digest())
    middle = len(ordered) // 2
    return ordered[:middle], ordered[middle:]


def choose_balanced_threshold(real_scores: list[float], fake_scores: list[float]) -> tuple[float, float]:
    """개발용 real/fake의 balanced accuracy 최대값. 동점이면 높은 임계값."""
    real = np.asarray(real_scores, dtype=np.float64)
    fake = np.asarray(fake_scores, dtype=np.float64)
    if real.ndim != 1 or fake.ndim != 1 or real.size == 0 or fake.size == 0:
        raise ValueError("Need nonempty 1D real and external score arrays")
    if not np.isfinite(real).all() or not np.isfinite(fake).all():
        raise ValueError("Scores must be finite")
    candidates = np.unique(np.concatenate((real, fake)))
    candidates = np.concatenate(([np.nextafter(candidates[0], -np.inf)], candidates))
    best_threshold = None
    best_numerator = -1
    for threshold in candidates:
        correct_real = int(np.count_nonzero(real <= threshold))
        correct_fake = int(np.count_nonzero(fake > threshold))
        numerator = correct_real * fake.size + correct_fake * real.size
        if numerator > best_numerator or (numerator == best_numerator and threshold > best_threshold):
            best_numerator = numerator
            best_threshold = float(threshold)
    balanced_accuracy = best_numerator / (2 * real.size * fake.size)
    return best_threshold, float(balanced_accuracy)


def _metrics(rows, threshold):
    # 외부 폴더 라벨을 이 함수 안에서만 임시 정답으로 사용한다.
    predictions = []
    for row in rows:
        label = 0 if row["group"] in {"validation", "real_test"} else 1
        score = float(row["score"])
        predictions.append({"label": str(label), "label_status": "verified", "status": "scored",
                            "score": score, "predicted_label": classify(score, threshold)[0]})
    metrics = compute_metrics(predictions)
    counts = metrics.pop("counts")
    metrics["counts_under_assumption"] = {
        "real_scored": counts["verified_real_scored"],
        "external_assumed_fake_scored": counts["verified_fake_scored"],
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Exploratory labeled-threshold experiment on saved model scores")
    parser.add_argument("--scores-dir", type=Path,
                        default=PROJECT_ROOT / "artifacts/evaluation/full_smoke_inference_001")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "artifacts/evaluation/labeled_threshold_experiment_001")
    parser.add_argument("--split-seed", type=int, default=42)
    args = parser.parse_args()
    source = args.scores_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    previous = json.loads((source / "run.json").read_text(encoding="utf-8"))
    if previous.get("purpose") != "provisional_full_smoke_inference" or previous.get("label_verified") is not False:
        raise ValueError("Expected full v1 smoke inference with unverified external labels")
    baseline = json.loads((source / "provisional_threshold.json").read_text(encoding="utf-8"))
    if baseline["binding_hash"] != previous["binding_hash"] or baseline["inventory_fingerprint"] != previous["inventory_fingerprint"]:
        raise ValueError("Source score run and baseline threshold do not match")
    with (source / "scores.csv").open(encoding="utf-8", newline="") as handle:
        scores = list(csv.DictReader(handle))
    if len(scores) != sum(previous["groups"].values()) or len({row["file_id"] for row in scores}) != len(scores):
        raise ValueError("Source score records are incomplete or repeated")
    if len({row["sha256"] for row in scores}) != len(scores):
        raise ValueError("Duplicate audio content in source score records")
    validation = [row for row in scores if row["group"] == "validation"]
    real_test = [row for row in scores if row["group"] == "real_test"]
    external_all = [row for row in scores if row["group"] == "external_unverified"]
    if (len(validation) != previous["groups"]["validation"] or
            len(real_test) != previous["groups"]["real_test"] or
            len(external_all) != previous["groups"]["external_unverified"]):
        raise ValueError("Score group counts differ from source run")
    if any(row["status"] != "scored" for row in validation + real_test):
        raise ValueError("Real development/test scores must all be valid")
    external = [row for row in external_all if row["status"] == "scored"]
    excluded = [row for row in external_all if row["status"] != "scored"]
    if any(row["status"] not in {"excluded_short", "excluded_zero_signal"} for row in excluded):
        raise ValueError("Unexpected external scoring error")
    fake_development, fake_test = split_external_files(external, seed=args.split_seed)
    development = validation + fake_development
    held_out = real_test + fake_test
    if {row["file_id"] for row in development} & {row["file_id"] for row in held_out}:
        raise ValueError("Development and held-out files overlap")
    threshold, development_balanced_accuracy = choose_balanced_threshold(
        [float(row["score"]) for row in validation],
        [float(row["score"]) for row in fake_development],
    )
    baseline_threshold = float(baseline["threshold"])
    report = {
        "purpose": "provisional_labeled_threshold_experiment", "label_verified": False,
        "label_assumption": "External folder is temporarily treated as fake=1",
        "model": "Existing real-only v1 smoke checkpoint (16 train segments, 20 updates)",
        "score_source": str(source / "scores.csv"), "score_binding_hash": previous["binding_hash"],
        "fake_file_split": "stable file-level SHA-256 hash, 50/50; filename prefix is shared, so not speaker-independent",
        "split_seed": args.split_seed, "threshold_objective": "maximize development balanced accuracy; ties choose higher threshold",
        "threshold": threshold, "development_balanced_accuracy_at_choice": development_balanced_accuracy,
        "baseline_real_95th_threshold": baseline_threshold,
        "counts": {"real_validation": len(validation), "external_development": len(fake_development),
                   "real_test": len(real_test), "external_test": len(fake_test),
                   "external_excluded_before_split": len(excluded)},
        "development_metrics_under_assumption": _metrics(development, threshold),
        "heldout_metrics_under_assumption": _metrics(held_out, threshold),
        "heldout_at_original_threshold": _metrics(held_out, baseline_threshold),
        "limitation": "Exploratory only: external labels are unverified, all external filenames share one prefix, and overall scores were previously inspected",
    }
    output.mkdir(parents=True, exist_ok=True)
    selection = []
    for split, rows in (("real_validation", validation), ("external_development", fake_development),
                        ("real_test", real_test), ("external_test", fake_test)):
        selection.extend({"split": split, "file_id": row["file_id"], "path": row["path"],
                          "sha256": row["sha256"], "score": row["score"],
                          "predicted_label": classify(float(row["score"]), threshold)[0]} for row in rows)
    write_csv_new(output / "predictions.csv", selection,
                  ["split", "file_id", "path", "sha256", "score", "predicted_label"])
    threshold_artifact = {key: report[key] for key in ("purpose", "label_verified", "split_seed",
                                                  "threshold_objective", "threshold", "counts")}
    threshold_artifact["score_binding_hash"] = previous["binding_hash"]
    threshold_artifact["development_fingerprint"] = canonical_hash({"files": [
        {"file_id": row["file_id"], "sha256": row["sha256"]} for row in development]})
    write_json_new(output / "threshold.json", threshold_artifact)
    write_json_new(output / "metrics.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
