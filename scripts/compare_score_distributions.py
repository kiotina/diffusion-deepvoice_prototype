"""Compare two WAV score files; external-folder statistics remain provisional."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def summarize(scores_path: Path, threshold_path: Path) -> dict:
    with scores_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    threshold = float(json.loads(threshold_path.read_text(encoding="utf-8"))["threshold"])

    def values(group: str) -> np.ndarray:
        selected = []
        for row in rows:
            category = row.get("group") or {"test": "real_test", "external": "external_unverified"}.get(row.get("split"))
            if category == group and row["status"] == "scored":
                selected.append(float(row["score"]))
        return np.asarray(selected, dtype=float)

    real, external = values("real_test"), values("external_unverified")
    if not len(real) or not len(external) or not np.isfinite(real).all() or not np.isfinite(external).all():
        raise ValueError("Both groups require finite scores")
    return {
        "threshold": threshold,
        "real_test": {"count": len(real), "median": float(np.median(real)),
                      "above_threshold": int(np.count_nonzero(real > threshold))},
        "fake_folder_unverified": {"count": len(external), "median": float(np.median(external)),
                                   "above_threshold": int(np.count_nonzero(external > threshold))},
        "provisional_auc_if_external_is_fake": float(roc_auc_score(
            np.r_[np.zeros(len(real)), np.ones(len(external))], np.r_[real, external])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-scores", type=Path, required=True)
    parser.add_argument("--old-threshold", type=Path, required=True)
    parser.add_argument("--new-scores", type=Path, required=True)
    parser.add_argument("--new-threshold", type=Path, required=True)
    args = parser.parse_args()
    with args.old_scores.open(encoding="utf-8", newline="") as handle:
        old_paths = {row["path"] for row in csv.DictReader(handle)
                     if row["status"] == "scored" and row["group"] in {"real_test", "external_unverified"}}
    with args.new_scores.open(encoding="utf-8", newline="") as handle:
        new_paths = {row["path"] for row in csv.DictReader(handle) if row["status"] == "scored"}
    if old_paths != new_paths:
        raise ValueError(f"Scored WAV sets differ: old={len(old_paths)} new={len(new_paths)} "
                         f"old_only={len(old_paths - new_paths)} new_only={len(new_paths - old_paths)}; "
                         f"examples={list(old_paths - new_paths)[:2]}, {list(new_paths - old_paths)[:2]}")
    print(json.dumps({"smoke_20_updates": summarize(args.old_scores, args.old_threshold),
                      "full_3_epochs": summarize(args.new_scores, args.new_threshold)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
