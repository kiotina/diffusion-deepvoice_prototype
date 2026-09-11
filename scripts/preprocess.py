from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 패키지를 설치하지 않고 이 파일을 직접 실행해도 src의 모듈을 찾게 한다.
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from deepvoice_diffusion.audio import (  # noqa: E402
    load_waveform,
    make_training_segments,
    segment_to_logmel,
)
from deepvoice_diffusion.config import load_config  # noqa: E402
from deepvoice_diffusion.dataset import find_wav_files, select_files, split_files  # noqa: E402


def parse_args() -> argparse.Namespace:
    """명령행에서 설정 파일, 임시 처리 개수, 출력 위치를 받을 수 있게 한다."""
    parser = argparse.ArgumentParser(description="Convert WAV files to normalized log-Mels.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "preprocess.yaml",
    )
    parser.add_argument("--limit", type=int, default=None, help="Temporary smoke-test limit.")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def output_name(path: Path, input_dir: Path, segment_index: int) -> str:
    """원본 경로 hash와 구간 번호로 안정적인 출력 파일명을 만든다."""
    relative = path.relative_to(input_dir).as_posix()
    digest = hashlib.sha1(relative.encode("utf-8")).hexdigest()[:10]
    return f"{digest}_segment{segment_index:03d}.npy"


def main() -> None:
    # 1) YAML 설정을 읽고 입력 경로와 파라미터가 유효한지 검사한다.
    args = parse_args()
    config = load_config(args.config)

    # 2) WAV를 재귀적으로 찾고 최소 1,000개 조건을 먼저 확인한다.
    files = select_files(
        find_wav_files(config.dataset.input_dir),
        config.dataset.min_samples,
        config.dataset.max_samples,
    )
    if args.limit is not None:
        # --limit은 전체 실행 전 소수 파일로 빠르게 확인할 때만 사용한다.
        if args.limit < 3:
            raise ValueError("--limit must be at least 3 so every split has a file")
        files = files[: args.limit]

    partitions = split_files(
        files,
        config.dataset.train_ratio,
        config.dataset.validation_ratio,
        config.dataset.test_ratio,
        config.dataset.split_seed,
    )

    # 3) Mel과 mask를 분리해 저장할 출력 폴더를 준비한다.
    output_dir = args.output_dir or config.dataset.output_dir
    if not output_dir.is_absolute():
        output_dir = (config.project_root / output_dir).resolve()
    split_dirs: dict[str, dict[str, Path]] = {}
    for split_name in partitions:
        mel_dir = output_dir / split_name / "mels"
        mask_dir = output_dir / split_name / "masks"
        mel_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        split_dirs[split_name] = {"mels": mel_dir, "masks": mask_dir}

    # manifest의 각 행, 실패 내역, 출력 shape 통계를 실행 중에 모은다.
    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    shapes: Counter[str] = Counter()
    mask_shapes: Counter[str] = Counter()
    split_statistics = {
        name: {"source_files": len(paths), "processed_sources": 0, "segments": 0, "padded_segments": 0, "discarded_tails": 0}
        for name, paths in partitions.items()
    }

    for split_name, split_sources in partitions.items():
        for source_path in tqdm(split_sources, desc=f"Creating {split_name} log-Mels"):
            try:
                source_info = sf.info(source_path)
                waveform = load_waveform(source_path, config.audio)
                segments = make_training_segments(waveform, config.audio)
                remainder = waveform.shape[0] % config.audio.target_samples
                if 0 < remainder < config.audio.minimum_remainder_samples:
                    split_statistics[split_name]["discarded_tails"] += 1

                for segment_index, segment in enumerate(segments):
                    mel, frame_mask = segment_to_logmel(segment, config.audio, config.mel)
                    filename = output_name(
                        source_path,
                        config.dataset.input_dir,
                        segment_index,
                    )
                    destination = split_dirs[split_name]["mels"] / filename
                    mask_destination = split_dirs[split_name]["masks"] / filename
                    np.save(destination, mel, allow_pickle=False)
                    np.save(mask_destination, frame_mask, allow_pickle=False)

                    shapes[str(tuple(mel.shape))] += 1
                    mask_shapes[str(tuple(frame_mask.shape))] += 1
                    is_padded = segment.valid_samples < config.audio.target_samples
                    split_statistics[split_name]["segments"] += 1
                    split_statistics[split_name]["padded_segments"] += int(is_padded)
                    start_seconds = segment.source_start_sample / config.audio.sample_rate
                    valid_duration = segment.valid_samples / config.audio.sample_rate
                    rows.append(
                        {
                            "index": len(rows),
                            "split": split_name,
                            "source_path": str(source_path),
                            "segment_index": segment_index,
                            "source_start_seconds": round(start_seconds, 6),
                            "source_end_seconds": round(start_seconds + valid_duration, 6),
                            "valid_duration_seconds": round(valid_duration, 6),
                            "is_padded": is_padded,
                            "mel_path": str(destination),
                            "mask_path": str(mask_destination),
                            "source_duration_seconds": round(float(source_info.duration), 6),
                            "source_sample_rate": source_info.samplerate,
                            "valid_samples": segment.valid_samples,
                            "valid_frames": int(frame_mask.sum()),
                            "valid_frame_ratio": round(float(frame_mask.mean()), 6),
                            "shape": "x".join(map(str, mel.shape)),
                            "mask_shape": "x".join(map(str, frame_mask.shape)),
                            "dtype": str(mel.dtype),
                            "min": float(mel.min()),
                            "max": float(mel.max()),
                        }
                    )
                split_statistics[split_name]["processed_sources"] += 1
            except Exception as exc:
                failures.append({"path": str(source_path), "error": str(exc)})

    # 8) 원본 WAV, Mel, mask의 대응 관계를 CSV로 저장한다.
    manifest_path = output_dir / "manifest.csv"
    fieldnames = list(rows[0].keys()) if rows else ["index", "source_path", "mel_path"]
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # 9) 전체 성공/실패 개수와 실제 적용 설정을 JSON으로 남긴다.
    summary = {
        "input_dir": str(config.dataset.input_dir),
        "output_dir": str(output_dir),
        "requested_files": len(files),
        "processed_files": sum(
            int(stats["processed_sources"]) for stats in split_statistics.values()
        ),
        "generated_segments": len(rows),
        "failed_files": failures,
        "splits": split_statistics,
        "shapes": dict(shapes),
        "mask_shapes": dict(mask_shapes),
        "split_seed": config.dataset.split_seed,
        "minimum_remainder_seconds": config.audio.minimum_remainder_seconds,
        "inference_overlap_seconds": config.audio.inference_overlap_seconds,
        "normalization_range": [-1.0, 1.0],
        "config": {
            "sample_rate": config.audio.sample_rate,
            "duration_seconds": config.audio.duration_seconds,
            "target_samples": config.audio.target_samples,
            "n_mels": config.mel.n_mels,
            "n_fft": config.mel.n_fft,
            "hop_length": config.mel.hop_length,
            "top_db": config.mel.top_db,
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nManifest: {manifest_path}")
    print(f"Summary:  {summary_path}")

    # 출력 파일을 먼저 남긴 뒤 실패 사실을 종료 코드로도 알린다.
    if failures:
        raise RuntimeError(f"Preprocessing failed for {len(failures)} file(s).")
    processed_files = sum(
        int(stats["processed_sources"]) for stats in split_statistics.values()
    )
    if args.limit is None and processed_files < config.dataset.min_samples:
        raise RuntimeError(
            f"Only {processed_files:,} files were processed; minimum is {config.dataset.min_samples:,}."
        )


if __name__ == "__main__":
    main()
