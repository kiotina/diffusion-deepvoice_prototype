"""파일별 판정, 검증된 라벨의 지표, JSON/CSV 보고서."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from .scoring import ScoreEngine


def classify(score: float, threshold: float) -> tuple[int, str]:
    if not math.isfinite(score) or not math.isfinite(threshold):
        raise ValueError("Classification requires finite score and threshold")
    above = int(score > threshold)
    return above, "이상 점수 기준 초과" if above else "기준 범위 내"


def score_entries(engine: ScoreEngine, rows: list[dict], threshold: float) -> tuple[list[dict], list[dict]]:
    predictions, segments = [], []
    for entry in rows:
        record = {key: entry[key] for key in ("file_id", "path", "source", "split", "label", "label_status")}
        record["purpose"] = entry.get("purpose", "evaluation")
        try:
            result, details = engine.score_file(entry["path"])
            record.update(result)
            if result["status"] == "scored":
                record["predicted_label"], record["decision"] = classify(result["score"], threshold)
            else:
                record["predicted_label"], record["decision"] = None, None
            record["error"] = None
            for detail in details:
                segments.append({"purpose": record["purpose"], "file_id": entry["file_id"],
                                 "path": entry["path"], **detail})
        except Exception as exc:
            record.update(status="error", score=None, max_score=None, segment_count=0,
                          predicted_label=None, decision=None, error=f"{type(exc).__name__}: {exc}")
        predictions.append(record)
    return predictions, segments


def compute_metrics(predictions: list[dict]) -> dict:
    known = [row for row in predictions if row["label_status"] == "verified" and row["label"] in {"0", "0.0", 0, "1", "1.0", 1}]
    valid = [row for row in known if row["status"] == "scored"]
    real = [row for row in valid if int(float(row["label"])) == 0]
    fake = [row for row in valid if int(float(row["label"])) == 1]
    tn = sum(row["predicted_label"] == 0 for row in real)
    fp = len(real) - tn
    tp = sum(row["predicted_label"] == 1 for row in fake)
    fn = len(fake) - tp
    reason = None if real and fake else "Both verified real and verified fake are required for fake-detection metrics"
    metrics = {
        "class_order": ["real", "fake"],
        "confusion_matrix": [[tn, fp], [fn, tp]] if real and fake else None,
        "accuracy": None, "precision": None, "recall": None, "f1": None,
        "real_fpr": fp / len(real) if real else None,
        "specificity": tn / len(real) if real else None,
        "balanced_accuracy": None, "roc_auc": None,
        "undefined_reason": reason, "roc_auc_reason": reason,
        "metric_reasons": {"precision": reason, "recall": reason, "f1": reason,
                           "specificity": None if real else "No verified real files",
                           "real_fpr": None if real else "No verified real files"},
        "counts": {
            "verified_real_scored": len(real), "verified_fake_scored": len(fake),
            "verified_real_excluded": sum(row["status"].startswith("excluded_") for row in known if int(float(row["label"])) == 0),
            "verified_fake_excluded": sum(row["status"].startswith("excluded_") for row in known if int(float(row["label"])) == 1),
            "verified_real_errors": sum(row["status"] == "error" for row in known if int(float(row["label"])) == 0),
            "verified_fake_errors": sum(row["status"] == "error" for row in known if int(float(row["label"])) == 1),
            "unverified_scored": sum(row["label_status"] == "unverified" and row["status"] == "scored" for row in predictions),
            "unverified_excluded": sum(row["label_status"] == "unverified" and row["status"].startswith("excluded_") for row in predictions),
            "unverified_errors": sum(row["label_status"] == "unverified" and row["status"] == "error" for row in predictions),
        },
    }
    if real and fake:
        metrics.update(accuracy=(tp + tn) / len(valid),
                       precision=tp / (tp + fp) if tp + fp else None,
                       recall=tp / (tp + fn),
                       f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
                       balanced_accuracy=((tp / len(fake)) + (tn / len(real))) / 2)
        metrics["metric_reasons"] = {"precision": None if tp + fp else "No files predicted as fake",
                                     "recall": None, "f1": None, "specificity": None, "real_fpr": None}
        try:
            from sklearn.metrics import roc_auc_score
        except ImportError:
            metrics["roc_auc_reason"] = "Install optional evaluation dependency scikit-learn"
        else:
            metrics["roc_auc"] = float(roc_auc_score(
                [int(float(row["label"])) for row in valid], [row["score"] for row in valid]))
            metrics["roc_auc_reason"] = None
    return metrics


def write_json_new(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def write_csv_new(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
