"""검증되지 않은 외부 라벨을 사용하는 탐색적 임계값 실험.

일반 calibration 경로와 분리한다. 임계값은 개발용 점수만으로 고른다.
"""
from __future__ import annotations

import hashlib

import numpy as np


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
