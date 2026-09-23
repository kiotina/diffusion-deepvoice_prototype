# 학습 체크포인트 관리
"""체크포인트 파일을 완전히 기록한 뒤 교체해 중간 저장 파일을 피한다."""
from __future__ import annotations

import os
from pathlib import Path

import torch


def save_checkpoint(path: Path, model, optimizer, state, config, fingerprints, preprocess_contract=None):
    """가중치와 재개에 필요한 상태를 임시 파일에 저장한 뒤 교체한다."""
    payload = {
        "format_version": 2 if preprocess_contract is not None else 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "state": state,
        "config": config,
        "torch_version": str(torch.__version__),
        "fingerprints": fingerprints,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    if preprocess_contract is not None:
        if preprocess_contract.get("verification") != "full":
            raise ValueError("Checkpoint v2 requires a fully verified preprocessing contract")
        payload["preprocess_contract"] = preprocess_contract
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path: Path):
    """체크포인트를 CPU로 읽고 지원하는 형식인지 검사한다."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") not in {1, 2}:
        raise ValueError("Unsupported checkpoint format")
    if payload["format_version"] == 2 and payload.get("preprocess_contract", {}).get("verification") != "full":
        raise ValueError("Checkpoint v2 has no verified preprocessing contract")
    return payload


def restore_checkpoint(payload, model, optimizer):
    """모델·optimizer·난수 상태를 되살리고 학습 위치를 반환한다."""
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    torch.set_rng_state(payload["torch_rng"])
    if torch.cuda.is_available() and payload["cuda_rng"]:
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    return payload["state"]
