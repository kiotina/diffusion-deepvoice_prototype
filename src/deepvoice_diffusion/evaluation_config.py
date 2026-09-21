# 평가 설정 관리
"""WAV 평가 설정과 명령행 override의 공통 검증."""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import yaml


@dataclass(frozen=True)
class EvaluationConfig:
    manifest_path: Path
    preprocess_config: Path
    external_dir: Path
    checkpoint: Path
    output_dir: Path
    verified_fake_manifest: Path | None = None
    timesteps: tuple[int, ...] = (100, 300, 500)
    noise_seeds: tuple[int, ...] = (1729, 2718)
    batch_size: int = 8
    cpu_threads: int = 4
    quantile: float = 0.95
    minimum_calibration_files: int = 20

    def __post_init__(self):
        for name in ("manifest_path", "preprocess_config", "external_dir", "checkpoint", "output_dir"):
            if not isinstance(getattr(self, name), Path):
                raise ValueError(f"{name} must be a path")
        if self.verified_fake_manifest is not None and not isinstance(self.verified_fake_manifest, Path):
            raise ValueError("verified_fake_manifest must be a path or null")
        if not self.timesteps or any(type(t) is not int or t < 0 for t in self.timesteps):
            raise ValueError("timesteps must contain nonnegative integers")
        if len(set(self.timesteps)) != len(self.timesteps):
            raise ValueError("timesteps must be unique")
        if not self.noise_seeds or any(type(s) is not int or not 0 <= s < 2**63 for s in self.noise_seeds):
            raise ValueError("noise_seeds must contain valid integer seeds")
        if len(set(self.noise_seeds)) != len(self.noise_seeds):
            raise ValueError("noise_seeds must be unique")
        if self.batch_size < 1 or self.cpu_threads < 1 or self.minimum_calibration_files < 20:
            raise ValueError("batch_size/cpu_threads must be positive; normal calibration needs at least 20 files")
        if not 0 < self.quantile < 1:
            raise ValueError("quantile must be between 0 and 1")

    def scoring_settings(self) -> dict:
        return {"timesteps": list(self.timesteps), "noise_seeds": list(self.noise_seeds),
                "segment_aggregation": "arithmetic_mean", "noise_metric": "masked_mse_per_sample",
                "noise_reuse": "same_two_cpu_tensors_all_segments_and_timesteps",
                "dtype": "float32"}


def load_evaluation_config(path: str | Path) -> EvaluationConfig:
    path = Path(path).resolve()
    project_root = path.parent.parent
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Evaluation config must be a mapping")
    allowed = {"manifest_path", "preprocess_config", "external_dir", "checkpoint", "output_dir",
               "verified_fake_manifest", "timesteps", "noise_seeds", "batch_size", "cpu_threads",
               "quantile", "minimum_calibration_files"}
    if set(raw) - allowed:
        raise ValueError(f"Unknown evaluation settings: {sorted(set(raw)-allowed)}")
    def path_value(key):
        value = raw.get(key)
        if value is None:
            return None
        candidate = Path(value).expanduser()
        return (candidate if candidate.is_absolute() else project_root / candidate).resolve()
    return EvaluationConfig(
        manifest_path=path_value("manifest_path"), preprocess_config=path_value("preprocess_config"),
        external_dir=path_value("external_dir"), checkpoint=path_value("checkpoint"),
        output_dir=path_value("output_dir"), verified_fake_manifest=path_value("verified_fake_manifest"),
        timesteps=tuple(raw.get("timesteps", [100, 300, 500])),
        noise_seeds=tuple(raw.get("noise_seeds", [1729, 2718])),
        batch_size=int(raw.get("batch_size", 8)), cpu_threads=int(raw.get("cpu_threads", 4)),
        quantile=float(raw.get("quantile", 0.95)),
        minimum_calibration_files=int(raw.get("minimum_calibration_files", 20)),
    )


def override_paths(config: EvaluationConfig, *, checkpoint=None, output_dir=None) -> EvaluationConfig:
    return replace(config,
                   checkpoint=Path(checkpoint).resolve() if checkpoint else config.checkpoint,
                   output_dir=Path(output_dir).resolve() if output_dir else config.output_dir)
