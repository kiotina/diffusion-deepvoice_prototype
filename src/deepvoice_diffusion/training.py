"""학습/검증 실행. epoch와 배치 커서를 저장해 안전한 지점에서 재개한다."""
from __future__ import annotations

import csv
import json
import math
import os
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .checkpoint import load_checkpoint, restore_checkpoint, save_checkpoint
from .diffusion import DiffusionSchedule, masked_error, seeded_noise
from .model import NoisePredictorUNet
from .training_config import TrainingConfig, project_path
from .training_data import MelDataset, read_manifest


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def choose_device(requested):
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; install the correct build or use CPU")
    return torch.device("cuda" if requested == "auto" and torch.cuda.is_available() else
                        "cpu" if requested == "auto" else requested)


@torch.inference_mode()
def evaluate(model, dataset, schedule, config, device):
    model.eval()
    total, count = 0.0, 0.0
    # 별도 generator 사용: 검증 loader가 학습의 전역 RNG를 소모하지 않는다.
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False,
                        generator=torch.Generator().manual_seed(config.validation_seed))
    for batch in loader:
        mel, mask = batch["mel"].to(device), batch["mask"].to(device)
        steps, noise = seeded_noise(mel, batch["index"], schedule.timesteps, config.validation_seed, 0)
        prediction = model(schedule.add_noise(mel, steps, noise, mask), steps, mask)
        error, valid = masked_error(prediction, noise, mask)
        if not torch.isfinite(error):
            raise FloatingPointError("Non-finite validation loss")
        total += error.item()
        count += valid.item()
    return total / count


def save_history(output_dir, history):
    write_json(output_dir / "history.json", history)
    with (output_dir / "history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "step", "train_loss", "validation_loss"])
        writer.writeheader()
        writer.writerows(history)
    if history:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot([r["epoch"] for r in history], [r["train_loss"] for r in history], label="Train")
        ax.plot([r["epoch"] for r in history], [r["validation_loss"] for r in history], label="Validation")
        ax.set(xlabel="Epoch", ylabel="Masked noise MSE")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "loss_curve.png", dpi=150)
        plt.close(fig)


def fit(config: TrainingConfig, *, resume: Path | None = None, max_steps: int | None = None):
    """max_steps는 이번 호출에서 허용하는 optimizer update 수이다."""
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be positive")
    config = replace(config, manifest_path=str(project_path(config.manifest_path)),
                     output_dir=str(project_path(config.output_dir)))
    output_dir = Path(config.output_dir)
    if resume is None and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Output directory is not empty; use --resume or a new --output-dir")
    if resume is not None and resume.resolve() != output_dir / "last.pt":
        raise ValueError("Resume from last.pt in the same output directory")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = choose_device(config.device)
    torch.set_num_threads(config.cpu_threads)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    torch.use_deterministic_algorithms(True)
    rows = read_manifest(config.manifest_path)
    train = MelDataset(rows, "train", config.train_limit)
    validation = MelDataset(rows, "validation", config.validation_limit)
    # Test의 split 메타데이터만 검사한다. Test 배열은 열거나 평가하지 않는다.
    fingerprints = {"train": train.fingerprint(), "validation": validation.fingerprint()}
    model = NoisePredictorUNet(config.base_channels, config.time_dim).to(device)
    schedule = DiffusionSchedule(config.timesteps, config.beta_start, config.beta_end).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    state = dict(epoch=1, batch_cursor=0, step=0, error_sum=0.0, valid_count=0.0,
                 best_loss=None, bad_epochs=0, history=[])
    if resume is not None:
        payload = load_checkpoint(resume)
        previous = payload["config"]
        for name, value in asdict(config).items():
            if name not in {"epochs", "device", "cpu_threads"} and previous[name] != value:
                raise ValueError(f"Resume configuration mismatch: {name}")
        if fingerprints != payload["fingerprints"]:
            raise ValueError("Training/validation data changed since checkpoint")
        if config.epochs < previous["epochs"]:
            raise ValueError("Cannot reduce epochs on resume")
        state = restore_checkpoint(payload, model, optimizer)
        if state["best_loss"] is not None and not (output_dir / "best.pt").is_file():
            raise FileNotFoundError("The matching best.pt must remain in the run directory")
    else:
        state["initial_validation_loss"] = evaluate(model, validation, schedule, config, device)
    output_dir.mkdir(parents=True, exist_ok=True)
    run = dict(config=asdict(config), device=str(device), torch_version=str(torch.__version__),
               parameters=sum(p.numel() for p in model.parameters()), fingerprints=fingerprints,
               train_samples=len(train), validation_samples=len(validation),
               test_metadata_rows=sum(r["split"] == "test" for r in rows), test_evaluated=False)
    write_json(output_dir / "run_config.json", run)
    print(f"Device={device}, parameters={run['parameters']:,}, train={len(train)}, validation={len(validation)}", flush=True)
    starting_step = state["step"]
    starting_history_length = len(state["history"])
    stop_reason = "epochs_complete"
    train_seconds, validation_seconds, measured_steps, trained_samples = 0.0, 0.0, 0, 0
    invocation_start = time.perf_counter()

    def save_last():
        save_checkpoint(output_dir / "last.pt", model, optimizer, state, asdict(config), fingerprints)

    save_last()
    while state["epoch"] <= config.epochs:
        if state["bad_epochs"] >= config.patience:
            stop_reason = "early_stopping"
            break
        epoch = state["epoch"]
        generator = torch.Generator().manual_seed(config.seed + epoch)
        order = torch.randperm(len(train), generator=generator).tolist()
        batches = [order[i:i+config.batch_size] for i in range(0, len(order), config.batch_size)]
        loader = DataLoader(train, batch_sampler=batches[state["batch_cursor"]:],
                            generator=generator, num_workers=0)
        model.train()
        for batch in loader:
            tick = time.perf_counter()
            mel, mask = batch["mel"].to(device), batch["mask"].to(device)
            steps, noise = seeded_noise(mel, batch["index"], config.timesteps, config.seed, epoch)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(schedule.add_noise(mel, steps, noise, mask), steps, mask)
            error, count = masked_error(prediction, noise, mask)
            loss = error / count
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss; resume from last.pt after diagnosis")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip, error_if_nonfinite=True)
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            train_seconds += time.perf_counter() - tick
            measured_steps += 1
            trained_samples += mel.shape[0]
            state["step"] += 1
            state["batch_cursor"] += 1
            state["error_sum"] += error.item()
            state["valid_count"] += count.item()
            if state["step"] % config.checkpoint_every == 0:
                save_last()
                print(f"epoch={epoch} batch={state['batch_cursor']}/{len(batches)} loss={loss.item():.5f}", flush=True)
            if max_steps is not None and state["step"] - starting_step >= max_steps:
                stop_reason = "step_limit"
                break

        if state["batch_cursor"] == len(batches):
            tick = time.perf_counter()
            validation_loss = evaluate(model, validation, schedule, config, device)
            validation_seconds += time.perf_counter() - tick
            row = dict(epoch=epoch, step=state["step"],
                       train_loss=state["error_sum"] / state["valid_count"], validation_loss=validation_loss)
            state["history"].append(row)
            improved = state["best_loss"] is None or validation_loss < state["best_loss"]
            state["best_loss"] = validation_loss if improved else state["best_loss"]
            state["bad_epochs"] = 0 if improved else state["bad_epochs"] + 1
            state.update(epoch=epoch+1, batch_cursor=0, error_sum=0.0, valid_count=0.0)
            if improved:
                save_checkpoint(output_dir / "best.pt", model, optimizer, state, asdict(config), fingerprints)
            save_history(output_dir, state["history"])
            print(f"epoch={epoch} train={row['train_loss']:.6f} validation={validation_loss:.6f}", flush=True)
        save_last()
        if stop_reason == "step_limit":
            break

    save_history(output_dir, state["history"])
    seconds_per_step = train_seconds / measured_steps if measured_steps else None
    full_train_count = sum(r["split"] == "train" for r in rows)
    validations = max(1, len(state["history"]) - starting_history_length)
    estimated_epoch = None
    if seconds_per_step is not None:
        estimated_epoch = seconds_per_step * math.ceil(full_train_count / config.batch_size)
        estimated_epoch += validation_seconds / validations * (sum(r["split"] == "validation" for r in rows) / len(validation))
    report = dict(**run, stop_reason=stop_reason, total_steps=state["step"],
                  completed_epochs=state["epoch"]-1, next_batch=state["batch_cursor"],
                  best_validation_loss=state["best_loss"], measured_steps=measured_steps,
                  initial_validation_loss=state["initial_validation_loss"],
                  train_seconds=train_seconds, validation_seconds=validation_seconds,
                  elapsed_seconds=time.perf_counter()-invocation_start,
                  seconds_per_step=seconds_per_step, trained_samples=trained_samples,
                  estimated_full_epoch_seconds=estimated_epoch,
                  estimate_note="Small-subset timing only; excludes full I/O and may vary by load.")
    write_json(output_dir / "summary.json", report)
    return report
