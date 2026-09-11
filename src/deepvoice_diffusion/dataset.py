# 전처리할 파일 목록 관리

from __future__ import annotations

import random
from pathlib import Path


def find_wav_files(input_dir: str | Path) -> list[Path]:
    """입력 폴더 아래에 있는 모든 WAV 파일을 재귀적으로 찾는다."""
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {input_dir}")

    # 정렬해 두면 운영체제가 파일을 반환하는 순서와 관계없이
    # 실행할 때마다 같은 index와 같은 출력 파일명을 사용할 수 있다.
    return sorted(
        path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() == ".wav"
    )


def select_files(files: list[Path], minimum: int, maximum: int | None) -> list[Path]:
    """최소 데이터 개수를 보장하고 설정된 최대 개수만큼 선택한다."""
    # 잘못된 경로나 불완전한 데이터로 학습이 시작되는 것을 미리 막는다.
    if len(files) < minimum:
        raise RuntimeError(
            f"Need at least {minimum:,} WAV files, but found only {len(files):,}."
        )

    # maximum이 None이면 현재 발견한 1,819개를 모두 사용한다.
    return files if maximum is None else files[:maximum]


def split_files(
    files: list[Path],
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[Path]]:
    """원본 파일을 재현 가능한 train/validation/test 그룹으로 나눈다."""
    if len(files) < 3:
        raise ValueError("At least three files are required to create all splits")
    ratios = (train_ratio, validation_ratio, test_ratio)
    if any(ratio <= 0 for ratio in ratios) or not abs(sum(ratios) - 1.0) < 1e-9:
        raise ValueError("split ratios must be positive and sum to 1")

    shuffled = list(files)
    random.Random(seed).shuffle(shuffled)
    validation_count = max(1, round(len(shuffled) * validation_ratio))
    test_count = max(1, round(len(shuffled) * test_ratio))
    train_count = len(shuffled) - validation_count - test_count
    if train_count < 1:
        raise ValueError("split ratios leave no training files")
    return {
        "train": shuffled[:train_count],
        "validation": shuffled[train_count : train_count + validation_count],
        "test": shuffled[train_count + validation_count :],
    }
