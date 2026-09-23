# WAV 이상 점수 계산
"""원본 WAV에서 재현 가능한 diffusion noise 예측 오차를 계산한다."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .audio import load_waveform, make_inference_segments, segment_to_logmel
from .checkpoint import load_checkpoint
from .config import load_config
from .data_contract import frontend, sha256_file
from .diffusion import DiffusionSchedule, masked_mse_per_sample
from .evaluation_config import EvaluationConfig
from .model import NoisePredictorUNet


def canonical_hash(value: dict) -> str:
    """설정 dict를 일정한 순서의 JSON으로 바꿔 SHA-256 지문을 만든다."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


class ScoreEngine:
    """체크포인트와 고정 noise를 사용해 WAV의 이상 점수를 계산한다."""

    def __init__(self, config: EvaluationConfig, *, legacy_smoke: bool = False):
        """모델·전처리 설정을 검증하고 평가용 noise와 결합 지문을 준비한다."""
        self.config = config
        self.preprocess = load_config(config.preprocess_config)
        self.payload = load_checkpoint(config.checkpoint)
        if self.payload["format_version"] == 1 and not legacy_smoke:
            raise ValueError("Checkpoint v1 can only score in explicit legacy functional smoke mode")
        if self.payload["format_version"] == 2:
            contract = self.payload["preprocess_contract"]
            if contract["frontend"] != frontend(self.preprocess):
                raise ValueError("Evaluation frontend differs from checkpoint contract")
            if contract["manifest_sha256"] != sha256_file(config.manifest_path):
                raise ValueError("Evaluation split manifest differs from checkpoint contract")
        if self.preprocess.audio.trim_silence:
            raise ValueError("Evaluation requires trim_silence=false for the agreed WAV contract")
        if self.preprocess.audio.target_samples != 32000 or self.preprocess.mel.n_mels != 80:
            raise ValueError("Evaluation expects 2-second 16 kHz / 80-Mel model inputs")
        if self.preprocess.mel.hop_length != 256:
            raise ValueError("Evaluation expects 126 Mel frames per 2-second segment")
        if self.preprocess.audio.sample_rate != 16000:
            raise ValueError("Evaluation expects 16 kHz audio")
        if self.preprocess.audio.inference_hop_samples != 16000:
            raise ValueError("Evaluation expects a 1-second inference hop")
        if self.preprocess.audio.minimum_remainder_samples != 16000:
            raise ValueError("Evaluation expects a 1-second minimum usable duration")
        for step in config.timesteps:
            if step >= self.payload["config"]["timesteps"]:
                raise ValueError(f"Timestep {step} exceeds checkpoint schedule")
        torch.set_num_threads(config.cpu_threads)
        model_config = self.payload["config"]
        self.model = NoisePredictorUNet(model_config["base_channels"], model_config["time_dim"]).cpu()
        self.model.load_state_dict(self.payload["model"])
        self.model.eval()
        self.schedule = DiffusionSchedule(model_config["timesteps"], model_config["beta_start"],
                                          model_config["beta_end"]).cpu()
        self.noises = []
        for seed in config.noise_seeds:
            generator = torch.Generator(device="cpu").manual_seed(seed)
            self.noises.append(torch.randn((1, 80, 126), generator=generator, dtype=torch.float32))
        self.binding = {
            "checkpoint_sha256": sha256_file(config.checkpoint),
            "checkpoint_format": self.payload["format_version"],
            "frontend": frontend(self.preprocess),
            "scoring": config.scoring_settings(),
            "schedule": {key: model_config[key] for key in ("timesteps", "beta_start", "beta_end")},
            "manifest_sha256": sha256_file(config.manifest_path),
            "runtime": {"torch": str(torch.__version__), "numpy": np.__version__},
        }
        self.binding_hash = canonical_hash(self.binding)

    @torch.inference_mode()
    def score_file(self, path: str | Path) -> tuple[dict, list[dict]]:
        """WAV의 구간별 masked MSE를 평균해 파일 점수와 상세 기록을 반환한다."""
        path = Path(path).resolve()
        waveform = load_waveform(path, self.preprocess.audio)
        if waveform.shape[0] < self.preprocess.audio.minimum_remainder_samples:
            return {"status": "excluded_short", "score": None, "max_score": None,
                    "segment_count": 0}, []
        if np.all(waveform == 0):
            return {"status": "excluded_zero_signal", "score": None, "max_score": None,
                    "segment_count": 0}, []
        segments = make_inference_segments(waveform, self.preprocess.audio)
        records: list[dict] = []
        # 모든 구간을 같은 고정 timestep·noise로 측정해 우연한 차이를 줄인다.
        for start in range(0, len(segments), self.config.batch_size):
            chunk = [segment_to_logmel(segment, self.preprocess.audio, self.preprocess.mel)
                     for segment in segments[start:start + self.config.batch_size]]
            clean = torch.from_numpy(np.stack([pair[0] for pair in chunk])).float()
            mask = torch.from_numpy(np.stack([pair[1] for pair in chunk])).float()
            by_step = {}
            for step in self.config.timesteps:
                steps = torch.full((len(chunk),), step, dtype=torch.long)
                scores = []
                for noise in self.noises:
                    batch_noise = noise.unsqueeze(0).expand_as(clean)
                    noisy = self.schedule.add_noise(clean, steps, batch_noise, mask)
                    prediction = self.model(noisy, steps, mask)
                    if not torch.isfinite(prediction).all():
                        raise FloatingPointError("Non-finite model prediction")
                    measured = masked_mse_per_sample(prediction, batch_noise, mask)
                    if not torch.isfinite(measured).all():
                        raise FloatingPointError("Non-finite segment score")
                    scores.append(measured.numpy())
                by_step[step] = np.mean(np.stack(scores, axis=0), axis=0)
            for offset, segment in enumerate(segments[start:start + self.config.batch_size]):
                index = start + offset
                values = {f"t{step}": float(by_step[step][offset]) for step in self.config.timesteps}
                records.append({"segment_index": index,
                                "start_seconds": segment.source_start_sample / self.preprocess.audio.sample_rate,
                                "end_seconds": (segment.source_start_sample + segment.valid_samples) / self.preprocess.audio.sample_rate,
                                "valid_seconds": segment.valid_samples / self.preprocess.audio.sample_rate,
                                "score": float(np.mean(list(values.values()))), **values})
        scores = [row["score"] for row in records]
        # 구간 평균이 파일 대표 점수이며 최고 구간 점수는 보조 정보다.
        return {"status": "scored", "score": float(np.mean(scores)),
                "max_score": float(np.max(scores)), "segment_count": len(records)}, records
