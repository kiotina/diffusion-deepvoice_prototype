"""전처리 manifest의 분할을 그대로 사용하는 학습 데이터."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

MEL_SHAPE = (1, 80, 126)
MASK_SHAPE = (1, 1, 126)


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    path = Path(path).resolve()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"split", "source_path", "segment_index", "mel_path", "mask_path"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Manifest requires columns: {sorted(required)}")
        rows = list(reader)
    if not rows:
        raise ValueError("Manifest is empty")
    source_splits: dict[str, str] = {}
    seen_segments: set[tuple[str, str]] = set()
    seen_paths: set[str] = set()
    for row in rows:
        split = row["split"]
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Invalid split: {split}")
        source = str(Path(row["source_path"]).resolve()).casefold()
        if source_splits.setdefault(source, split) != split:
            raise ValueError(f"Source crosses splits: {row['source_path']}")
        key = (source, row["segment_index"])
        if key in seen_segments:
            raise ValueError(f"Duplicate source segment: {key}")
        seen_segments.add(key)
        for field in ("mel_path", "mask_path"):
            file_path = Path(row[field])
            # 새 상대 경로 manifest도 지원: manifest 위치 기준으로 해석한다.
            if not file_path.is_absolute():
                file_path = path.parent / file_path
            file_path = file_path.resolve()
            normalized = str(file_path).casefold()
            if normalized in seen_paths:
                raise ValueError(f"Reused array path: {file_path}")
            seen_paths.add(normalized)
            row[field] = str(file_path)
    return rows


class MelDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], split: str, limit: int | None = None):
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Invalid split: {split}")
        if limit is not None and limit < 1:
            raise ValueError("Dataset limit must be positive")
        self.rows = [row for row in rows if row["split"] == split]
        if limit is not None:
            self.rows = self.rows[:limit]
        if not self.rows:
            raise ValueError(f"No samples in {split}")
        self.split = split

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        row = self.rows[index]
        mel = np.load(row["mel_path"], allow_pickle=False)
        mask = np.load(row["mask_path"], allow_pickle=False)
        if mel.shape != MEL_SHAPE or mask.shape != MASK_SHAPE:
            raise ValueError(f"Invalid Mel/mask shapes: {mel.shape}, {mask.shape}")
        if mel.dtype != np.float32 or mask.dtype != np.float32:
            raise ValueError("Mel and mask must be float32")
        if not np.isfinite(mel).all() or not np.isfinite(mask).all():
            raise ValueError("Mel/mask contains non-finite values")
        if mel.min() < -1.00001 or mel.max() > 1.00001:
            raise ValueError("Mel must be normalized to [-1, 1]")
        if not np.isin(mask, [0, 1]).all() or mask.sum() == 0:
            raise ValueError("Mask must be binary with at least one valid frame")
        if np.any(np.diff(mask.reshape(-1)) > 0):
            raise ValueError("Expected right-only padding mask")
        return {"mel": torch.from_numpy(mel), "mask": torch.from_numpy(mask), "index": index}

    def fingerprint(self) -> str:
        """재개 시 목록뿐 아니라 실제 입력 파일 변경도 확인한다."""
        digest = hashlib.sha256()
        for row in self.rows:
            for field in ("source_path", "segment_index", "mel_path", "mask_path"):
                digest.update(row[field].encode("utf-8") + b"\0")
            for field in ("mel_path", "mask_path"):
                digest.update(Path(row[field]).read_bytes())
        return digest.hexdigest()
