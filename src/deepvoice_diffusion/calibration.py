# 판정 임계값 관리
"""real validation WAV 점수의 분위수로 임계값을 만들고 사용 조건을 확인한다."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .evaluation_data import validate_inventory_against_manifest
from .scoring import ScoreEngine, canonical_hash


def calibration_sources(rows: list[dict]) -> list[dict]:
    """평가 목록에서 real validation 파일만 안정적인 순서로 고른다."""
    return sorted((row for row in rows if row["source"] == "real_manifest" and row["split"] == "validation"),
                  key=lambda row: row["file_id"])


def create_threshold(engine: ScoreEngine, inventory: list[dict], *, purpose: str = "evaluation",
                     selected: list[dict] | None = None) -> tuple[dict, list[dict]]:
    """real validation 파일 점수의 지정 분위수로 임계값을 만든다."""
    if purpose not in {"evaluation", "functional_smoke"}:
        raise ValueError("Unknown threshold purpose")
    validate_inventory_against_manifest(
        inventory,
        Path(engine.config.manifest_path),
        allow_real_subset=purpose == "functional_smoke",
    )
    if purpose == "evaluation" and selected is not None:
        raise ValueError("Ordinary calibration must use every real validation source")
    sources = calibration_sources(inventory) if selected is None else selected
    if not sources or any(row not in calibration_sources(inventory) for row in sources):
        raise ValueError("Calibration requires existing real validation entries")
    if len({row["file_id"] for row in sources}) != len(sources):
        raise ValueError("Repeated calibration source")
    required = 4 if purpose == "functional_smoke" else engine.config.minimum_calibration_files
    scored = []
    excluded = []
    for row in sources:
        result, _ = engine.score_file(row["path"])
        if result["status"] == "scored":
            scored.append({"file_id": row["file_id"], "score": result["score"]})
        elif result["status"] in {"excluded_short", "excluded_zero_signal"}:
            excluded.append({"file_id": row["file_id"], "status": result["status"]})
        else:
            raise ValueError("Unexpected calibration status")
    if len(scored) < required:
        raise ValueError(f"Calibration needs at least {required} valid files, got {len(scored)}")
    values = np.array([entry["score"] for entry in scored], dtype=np.float64)
    if not np.isfinite(values).all():
        raise FloatingPointError("Non-finite calibration score")
    # test 점수는 사용하지 않고 real validation 파일 점수만으로 임계값을 정한다.
    threshold = float(np.quantile(values, engine.config.quantile, method="inverted_cdf"))
    artifact = {
        "schema_version": 1, "purpose": purpose, "binding": engine.binding,
        "binding_hash": engine.binding_hash,
        "calibration_files": [{key: row[key] for key in ("file_id", "path", "sha256")} for row in sources],
        "calibration_fingerprint": canonical_hash({"files": [
            {key: row[key] for key in ("file_id", "path", "sha256")} for row in sources]}),
        "quantile": engine.config.quantile, "quantile_method": "inverted_cdf",
        "comparison": "score > threshold", "threshold": threshold,
        "calibration_total": len(sources), "calibration_valid": len(scored),
        "calibration_excluded": excluded,
        "calibration_exceedance_rate": float(np.mean(values > threshold)),
    }
    return artifact, scored


def check_threshold(artifact: dict, engine: ScoreEngine, inventory: list[dict],
                    *, allow_smoke: bool = False) -> None:
    """저장된 임계값이 현재 모델·점수 설정·validation 파일과 맞는지 검사한다."""
    if artifact.get("schema_version") != 1 or artifact.get("comparison") != "score > threshold":
        raise ValueError("Unsupported threshold format or comparison rule")
    if artifact.get("purpose") != "evaluation" and not (allow_smoke and artifact.get("purpose") == "functional_smoke"):
        raise ValueError("Functional smoke threshold cannot be used for ordinary evaluation")
    validate_inventory_against_manifest(
        inventory,
        Path(engine.config.manifest_path),
        allow_real_subset=artifact.get("purpose") == "functional_smoke",
    )
    if artifact.get("binding_hash") != engine.binding_hash or artifact.get("binding") != engine.binding:
        raise ValueError("Threshold model/frontend/scoring binding differs")
    if artifact.get("quantile") != engine.config.quantile or artifact.get("quantile_method") != "inverted_cdf":
        raise ValueError("Threshold quantile settings differ")
    current = {row["file_id"]: row for row in calibration_sources(inventory)}
    selected = artifact.get("calibration_files", [])
    if not selected or len({row["file_id"] for row in selected}) != len(selected):
        raise ValueError("Invalid calibration inventory")
    for row in selected:
        original = current.get(row["file_id"])
        if original is None or any(original[key] != row[key] for key in ("path", "sha256")):
            raise ValueError("Calibration source changed")
    if artifact["purpose"] == "evaluation" and len(selected) != len(current):
        raise ValueError("Validation inventory changed since calibration")
    if artifact.get("calibration_fingerprint") != canonical_hash({"files": selected}):
        raise ValueError("Calibration fingerprint differs")
    if not np.isfinite(artifact.get("threshold", float("nan"))):
        raise ValueError("Threshold is not finite")


def save_threshold(path: Path, artifact: dict) -> None:
    """임계값과 계산 근거를 새 JSON 파일로 저장한다."""
    with path.open("x", encoding="utf-8") as handle:
        json.dump(artifact, handle, ensure_ascii=False, indent=2, allow_nan=False)


def load_threshold(path: Path) -> dict:
    """저장된 임계값 JSON을 읽는다."""
    return json.loads(path.read_text(encoding="utf-8"))
