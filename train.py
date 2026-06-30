"""MiniMind pre-training entry point."""
import argparse
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from model.model import MiniMind, ModelConfig
from data.dataset import create_dataloader
from utils.config import load_config, save_config
from utils.logging import setup_logging
from utils.lr_scheduler import get_cosine_schedule_with_warmup


def parse_args():
    parser = argparse.ArgumentParser(description="MiniMind Pre-training")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def get_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_grad_scaler(cfg: dict, device: torch.device) -> torch.cuda.amp.GradScaler | None:
    if cfg["training"]["mixed_precision"] and device.type == "cuda":
        return torch.cuda.amp.GradScaler()
    return None


def compute_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy loss, ignoring padding (-100)."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def train_step(
    model: MiniMind,
    batch: dict,
    scaler: torch.cuda.amp.GradScaler | None,
    device: torch.device,
) -> float:
    """Single training step. Returns loss value."""
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)

    amp_ctx = torch.autocast(device_type=device.type, dtype=torch.float16) if scaler else nullcontext()
    with amp_ctx:
        logits = model(input_ids)
        loss = compute_loss(logits, labels)

    if scaler:
        scaler.scale(loss).backward()
    else:
        loss.backward()

    return loss.item()


@torch.no_grad()
def evaluate(model: MiniMind, dataloader, device: torch.device, max_batches: int = 20) -> float:
    """Compute average eval loss."""
    model.eval()
    total_loss = 0.0
    count = 0
    for batch in dataloader:
        if count >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids)
        loss = compute_loss(logits, labels)
        total_loss += loss.item()
        count += 1
    model.train()
    return total_loss / max(1, count)


def save_checkpoint(
    model: MiniMind,
    optimizer: torch.optim.Optimizer,
    scheduler,
    cfg: dict,
    step: int,
    loss: float,
    save_dir: str,
    save_optimizer: bool = True,
):
    """Save model and optimizer state."""
    os.makedirs(save_dir, exist_ok=True)
    ckpt = {
        "step": step,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "config": cfg,
    }
    if save_optimizer:
        ckpt["optimizer_state_dict"] = optimizer.state_dict()
        ckpt["scheduler_state_dict"] = scheduler.state_dict()

    path = os.path.join(save_dir, f"step_{step:07d}.pt")
    torch.save(ckpt, path)
    return path


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = get_device(args.device)

    # Logger
    log_cfg = cfg.get("logging", {})
    log_dir = log_cfg.get("log_dir", "logs")
    logger = setup_logging(log_dir)
    logger.info(f"Device: {device}")

    # Model
    model_cfg = ModelConfig(**cfg["model"])
    model = MiniMind(model_cfg).to(device)
    logger.info(f"Model params: {model.get_num_params():,}")

    # Data
    data_cfg = cfg["data"]
    train_loader = create_dataloader(
        file_path=data_cfg["train_path"],
        tokenizer_path=data_cfg["tokenizer_path"],
        max_seq_len=model_cfg.max_seq_len,
        batch_size=cfg["training"]["micro_batch_size"],
        num_workers=data_cfg.get("num_workers", 0),
        pin_memory=data_cfg.get("pin_memory", False),
        shuffle=True,
    )

    eval_loader = create_dataloader(
        file_path=data_cfg["eval_path"],
        tokenizer_path=data_cfg["tokenizer_path"],
        max_seq_len=model_cfg.max_seq_len,
        batch_size=cfg["training"]["micro_batch_size"],
        num_workers=data_cfg.get("num_workers", 0),
        pin_memory=data_cfg.get("pin_memory", False),
        shuffle=False,
    )

    # Optimizer & scheduler
    train_cfg = cfg["training"]
    optimizer = model.configure_optimizers(
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        betas=(train_cfg["beta1"], train_cfg["beta2"]),
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_steps=train_cfg["warmup_steps"],
        total_steps=train_cfg["max_steps"],
        min_lr_ratio=train_cfg["min_lr"] / train_cfg["learning_rate"],
    )

    # Mixed precision
    scaler = get_grad_scaler(cfg, device)

    # Resume
    ckpt_cfg = cfg.get("checkpoint", {})
    save_dir = ckpt_cfg.get("save_dir", "checkpoints")
    start_step = 0
    if ckpt_cfg.get("resume"):
        ckpt_path = ckpt_cfg["resume"]
        logger.info(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_step = ckpt["step"]

    # TensorBoard
    writer = None
    if log_cfg.get("tensorboard", False):
        writer = SummaryWriter(log_dir=log_dir)

    # WandB
    if log_cfg.get("wandb", False):
        import wandb
        wandb.init(project="minimind", config=cfg)

    # Training loop
    max_steps = train_cfg["max_steps"]
    grad_accum = train_cfg.get("gradient_accumulation_steps", 1)
    log_interval = train_cfg["log_interval"]
    save_interval = train_cfg["save_interval"]
    eval_interval = train_cfg["eval_interval"]
    max_grad_norm = train_cfg["max_grad_norm"]

    model.train()
    global_step = start_step
    accum_loss = 0.0
    train_iter = iter(train_loader)
    t_start = time.time()

    logger.info(f"Training from step {start_step} to {max_steps}")

    for step in range(start_step, max_steps):
        # Fetch batch (recreate iterator when exhausted)
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        loss = train_step(model, batch, scaler, device)
        accum_loss += loss

        # Gradient accumulation
        if (step + 1) % grad_accum == 0:
            if scaler:
                scaler.unscale_(optimizer)

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()
            scheduler.step()
            global_step += 1

        # Logging
        if (step + 1) % log_interval == 0:
            avg_loss = accum_loss / log_interval
            lr = scheduler.get_last_lr()[0]
            tokens_per_sec = (log_interval * model_cfg.max_seq_len * cfg["training"]["micro_batch_size"]) / (time.time() - t_start + 1e-8)

            logger.info(
                f"step {step + 1:>7d}/{max_steps} | loss: {avg_loss:.4f} | lr: {lr:.2e} | tok/s: {tokens_per_sec:.0f}"
            )

            if writer:
                writer.add_scalar("train/loss", avg_loss, step + 1)
                writer.add_scalar("train/lr", lr, step + 1)
                writer.add_scalar("train/tokens_per_sec", tokens_per_sec, step + 1)
            if log_cfg.get("wandb", False):
                import wandb
                wandb.log({"train/loss": avg_loss, "train/lr": lr, "step": step + 1})

            accum_loss = 0.0
            t_start = time.time()

        # Evaluation
        if (step + 1) % eval_interval == 0:
            eval_loss = evaluate(model, eval_loader, device)
            logger.info(f"step {step + 1:>7d} | eval loss: {eval_loss:.4f}")
            if writer:
                writer.add_scalar("eval/loss", eval_loss, step + 1)

        # Checkpoint
        if (step + 1) % save_interval == 0 or step + 1 == max_steps:
            sp = save_checkpoint(
                model, optimizer, scheduler, cfg,
                step=step + 1, loss=accum_loss / max(1, log_interval),
                save_dir=save_dir,
                save_optimizer=ckpt_cfg.get("save_optimizer", True),
            )
            logger.info(f"Checkpoint saved: {sp}")

    # Final save
    final_path = save_checkpoint(
        model, optimizer, scheduler, cfg,
        step=max_steps, loss=0.0,
        save_dir=save_dir,
        save_optimizer=ckpt_cfg.get("save_optimizer", True),
    )
    logger.info(f"Training complete. Final checkpoint: {final_path}")

    if writer:
        writer.close()
    if log_cfg.get("wandb", False):
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
