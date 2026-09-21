"""파일별 이상 점수를 색깔 점으로 비교한다."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Scatter plot of real and external WAV anomaly scores")
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=Path)
    parser.add_argument("--title", default="WAV anomaly score distribution")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with args.scores.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    def group(row):
        if "group" in row:
            return row["group"]
        return {"test": "real_test", "external": "external_unverified"}.get(row.get("split"))

    groups = {
        key: np.array([float(row["score"]) for row in rows
                       if group(row) == key and row["status"] == "scored"])
        for key in ("real_test", "external_unverified")
    }
    if any(values.size == 0 or not np.isfinite(values).all() for values in groups.values()):
        raise ValueError("Both groups need finite file scores")
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(13, 6), dpi=170)
    fig.patch.set_facecolor("#fbfcfd")
    ax.set_facecolor("#fbfcfd")
    rng = np.random.default_rng(42)
    for y, key, color, label, alpha, size in (
        (1, "real_test", "#15803d", "Real test", 0.65, 24),
        (0, "external_unverified", "#dc2626", "Fake folder (unverified)", 0.26, 17),
    ):
        values = groups[key]
        jitter = rng.uniform(-0.28, 0.28, len(values))
        ax.scatter(values, y + jitter, s=size, c=color, alpha=alpha, edgecolors="none",
                   label=f"{label}: {len(values):,} files", rasterized=True)
        median = float(np.median(values))
        ax.plot([median, median], [y - 0.35, y + 0.35], color=color, lw=3)
    if args.threshold is not None:
        artifact = json.loads(args.threshold.read_text(encoding="utf-8"))
        value = float(artifact["threshold"])
        ax.axvline(value, color="#4b5563", lw=1.6, ls="--",
                   label=f"Threshold: {value:.6f}")
    ax.set_yticks([0, 1], ["Fake folder\n(unverified)", "Real test"])
    ax.set_ylim(-0.55, 1.55)
    ax.set_xlabel("File anomaly score (higher = more anomalous)")
    ax.set_title(args.title, loc="left", fontsize=16, weight="bold", pad=20)
    ax.grid(axis="x", color="#d1d5db", alpha=0.6)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(loc="upper left", frameon=False, ncol=3, bbox_to_anchor=(0, -0.14))
    fig.text(0.06, 0.015, "Each dot = one WAV file. Fake-folder labels are provisional; source conditions differ.",
             fontsize=9, color="#6b7280")
    fig.subplots_adjust(bottom=0.24, left=0.12, right=0.98, top=0.86)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
