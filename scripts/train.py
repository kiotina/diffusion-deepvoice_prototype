"""학습 진입점. --smoke는 작은 subset과 제한된 update만 실행한다."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from deepvoice_diffusion.checkpoint import load_checkpoint  # noqa: E402
from deepvoice_diffusion.training import fit  # noqa: E402
from deepvoice_diffusion.training_config import TrainingConfig, load_training_config, project_path  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Real-only DDPM noise-prediction training")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true", help="16 train / 8 validation samples, at most 20 updates")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path, help="Resume the saved run from its last.pt")
    parser.add_argument("--epochs", type=int, help="Maximum epochs; may be increased on resume")
    parser.add_argument("--max-steps", type=int, help="Maximum optimizer updates in this invocation")
    args = parser.parse_args()
    if args.resume:
        if args.config or args.smoke:
            parser.error("--resume uses saved configuration; do not combine with --config or --smoke")
        args.resume = project_path(args.resume)
        config = TrainingConfig(**load_checkpoint(args.resume)["config"])
    else:
        config = load_training_config(args.config or PROJECT_ROOT / "configs/train.yaml")
        if args.smoke:
            config = replace(config, train_limit=16, validation_limit=8, epochs=10,
                             checkpoint_every=2, output_dir="artifacts/training_smoke")
            args.max_steps = min(args.max_steps, 20) if args.max_steps is not None else 20
    if args.epochs is not None:
        config = replace(config, epochs=args.epochs)
    if args.output_dir:
        config = replace(config, output_dir=str(args.output_dir))
    try:
        report = fit(config, resume=args.resume, max_steps=args.max_steps)
    except KeyboardInterrupt:
        print("Interrupted. Resume from the last completed checkpoint using --resume.", file=sys.stderr)
        raise SystemExit(130)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
