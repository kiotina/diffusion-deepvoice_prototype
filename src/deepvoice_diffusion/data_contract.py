# 전처리 결과 검증
"""전처리 재현성과 checkpoint에 연결할 입력 계약."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import librosa
import numpy as np
import soundfile

from .audio import load_waveform, make_training_segments, segment_to_logmel
from .config import PrototypeConfig
from .training_data import read_manifest


def sha256_file(path: str | Path) -> str:
    """파일 내용을 SHA-256 지문으로 요약해 변경 여부를 확인할 수 있게 한다."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frontend(config: PrototypeConfig) -> dict:
    """Mel 입력을 만든 전처리 규칙과 라이브러리 버전을 기록한다."""
    return {
        "audio": asdict(config.audio),
        "mel": asdict(config.mel),
        "rules": {
            "mono": True, "power": 2.0, "center": True,
            "db_reference": "segment_max", "normalization": "clip(2*(db+top_db)/top_db-1,-1,1)",
            "padding": "right_zero", "mask": "frame_center_sample", "training_tail": "at_least_minimum_remainder",
        },
        "libraries": {"numpy": np.__version__, "librosa": librosa.__version__,
                      "soundfile": soundfile.__version__},
    }


def contract_path(manifest_path: str | Path) -> Path:
    """manifest 옆에 둘 전처리 검증 결과 파일 경로를 반환한다."""
    return Path(manifest_path).resolve().parent / "preprocess_contract.json"


def verify_processed_data(manifest_path: str | Path, config: PrototypeConfig) -> dict:
    """train/validation의 모든 저장 배열을 원본에서 다시 만들어 대조한다."""
    manifest_path = Path(manifest_path).resolve()
    rows = read_manifest(manifest_path)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["split"] in {"train", "validation"}:
            grouped[row["source_path"]].append(row)
    if not grouped:
        raise ValueError("No train/validation sources to verify")
    sources = {}
    arrays = hashlib.sha256()
    count = 0
    for source_path, source_rows in sorted(grouped.items()):
        waveform = load_waveform(source_path, config.audio)
        segments = make_training_segments(waveform, config.audio)
        if len(segments) != len(source_rows):
            raise ValueError(f"Segment count mismatch: {source_path}")
        by_index = {int(row["segment_index"]): row for row in source_rows}
        if set(by_index) != set(range(len(segments))):
            raise ValueError(f"Segment indices mismatch: {source_path}")
        sources[source_path] = sha256_file(source_path)
        for index, segment in enumerate(segments):
            row = by_index[index]
            actual_mel = np.load(row["mel_path"], allow_pickle=False)
            actual_mask = np.load(row["mask_path"], allow_pickle=False)
            expected_mel, expected_mask = segment_to_logmel(segment, config.audio, config.mel)
            if not np.array_equal(actual_mel, expected_mel) or not np.array_equal(actual_mask, expected_mask):
                raise ValueError(f"Saved Mel/mask differs from source: {source_path} segment {index}")
            for field in ("mel_path", "mask_path"):
                arrays.update(row[field].encode("utf-8") + b"\0")
                arrays.update(sha256_file(row[field]).encode("ascii") + b"\0")
            count += 1
    return {
        "schema_version": 1, "verification": "full", "frontend": frontend(config),
        "manifest_sha256": sha256_file(manifest_path),
        "source_sha256": sources, "arrays_sha256": arrays.hexdigest(),
        "verified_splits": ["train", "validation"], "verified_sources": len(grouped),
        "verified_segments": count,
    }


def load_verified_contract(manifest_path: str | Path) -> dict | None:
    """저장된 계약과 현재 원본·Mel·mask의 지문이 같은지 확인한다."""
    path = contract_path(manifest_path)
    if not path.is_file():
        return None
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("verification") != "full" or contract.get("schema_version") != 1:
        raise ValueError("Preprocessing contract is not fully verified")
    if contract.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("Preprocessing contract manifest changed")
    for source, digest in contract.get("source_sha256", {}).items():
        if sha256_file(source) != digest:
            raise ValueError(f"Preprocessing contract source changed: {source}")
    rows = read_manifest(manifest_path)
    arrays = hashlib.sha256()
    for row in sorted((row for row in rows if row["split"] in {"train", "validation"}),
                      key=lambda row: (row["source_path"], int(row["segment_index"]))):
        for field in ("mel_path", "mask_path"):
            arrays.update(row[field].encode("utf-8") + b"\0")
            arrays.update(sha256_file(row[field]).encode("ascii") + b"\0")
    if arrays.hexdigest() != contract.get("arrays_sha256"):
        raise ValueError("Preprocessing contract arrays changed")
    return contract
