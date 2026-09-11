from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 프로젝트를 editable install하지 않은 상태에서도 src 모듈을 불러오기 위한 경로다.
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from deepvoice_diffusion.audio import audio_file_to_logmel  # noqa: E402
from deepvoice_diffusion.config import load_config  # noqa: E402
from deepvoice_diffusion.dataset import find_wav_files, select_files  # noqa: E402


def parse_args() -> argparse.Namespace:
    """기본 설정 파일 대신 다른 YAML을 검사에 사용할 수 있게 한다."""
    parser = argparse.ArgumentParser(description="Inspect the real-audio dataset.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "preprocess.yaml",
    )
    return parser.parse_args()


def main() -> None:
    # 설정과 최소 데이터 개수 조건은 실제 전처리와 동일하게 적용한다.
    config = load_config(parse_args().config)
    files = select_files(
        find_wav_files(config.dataset.input_dir),
        config.dataset.min_samples,
        config.dataset.max_samples,
    )

    sample_rates: Counter[int] = Counter()
    channels: Counter[int] = Counter()
    subtypes: Counter[str] = Counter()
    durations: list[float] = []
    failures: list[dict[str, str]] = []

    # waveform 전체를 변환하지 않고 WAV header에서 빠르게 메타데이터를 읽는다.
    for path in files:
        try:
            info = sf.info(path)
            sample_rates[info.samplerate] += 1
            channels[info.channels] += 1
            subtypes[info.subtype] += 1
            durations.append(float(info.duration))
        except Exception as exc:
            failures.append({"path": str(path), "error": str(exc)})

    if not durations:
        raise RuntimeError("No readable WAV files found")

    # 길이 분포를 보고 모델 입력 길이와 padding/crop 정책을 판단할 수 있다.
    ordered = sorted(durations)
    summary = {
        "input_dir": str(config.dataset.input_dir),
        "selected_files": len(files),
        "readable_files": len(durations),
        "failed_files": failures,
        "sample_rates": dict(sample_rates),
        "channels": dict(channels),
        "subtypes": dict(subtypes),
        "duration_seconds": {
            "min": min(ordered),
            "median": statistics.median(ordered),
            "mean": statistics.mean(ordered),
            "p95": ordered[max(0, int(len(ordered) * 0.95) - 1)],
            "max": max(ordered),
        },
    }

    # 사람이 다시 확인할 수 있도록 통계는 JSON으로 저장한다.
    artifacts_dir = config.project_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    report_path = artifacts_dir / "dataset_inspection.json"
    report_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 첫 WAV의 중앙 2초를 실제 전처리와 같은 방식으로 변환해 그림으로 확인한다.
    sample_mel = audio_file_to_logmel(files[0], config.audio, config.mel)[0]
    image_path = artifacts_dir / "sample_logmel.png"
    plt.figure(figsize=(11, 4))
    plt.imshow(sample_mel, origin="lower", aspect="auto", cmap="magma", vmin=-1, vmax=1)
    plt.title("Sample normalized log-Mel")
    plt.xlabel("Frame")
    plt.ylabel("Mel bin")
    plt.colorbar(label="Normalized amplitude")
    plt.tight_layout()
    plt.savefig(image_path, dpi=150)
    plt.close()

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nReport: {report_path}")
    print(f"Preview: {image_path}")


if __name__ == "__main__":
    main()
