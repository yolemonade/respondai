"""
Training loop
=============

Trains a :class:`model.transformer.RespondAITransformer` on the binary cache
produced by :mod:`data.preprocess`. Designed to run on a single GPU (Colab
T4 / A100, or local MPS for tiny sanity checks).

Key choices
-----------
* **AdamW**, lr 3e-4, weight_decay 0.01. Standard.
* **Warmup + cosine schedule**: 500 warmup steps, then cosine decay to 10% of
  peak. Numbers from the spec.
* **Gradient clipping** at 1.0. Cheap insurance against exploding gradients
  early in training.
* **Mixed precision** when CUDA is available (``torch.amp``). Falls back to
  fp32 on CPU/MPS where it's either unnecessary or unstable.

Checkpoints
-----------
Saved every ``save_every`` steps and at the end. We also keep a separate
``best.pt`` that tracks the lowest validation loss. ``last.pt`` is overwritten
each save.
"""
from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from data.dataset import CallResponseDataset, make_collate_fn
from data.tokenizer import Tokenizer
from model.transformer import RespondAITransformer, TransformerConfig

log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Training config (separate from model config)
# -----------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # Data
    cache_path: str = "data_cache/train.npz"
    val_cache_path: Optional[str] = None
    max_len: int = 512

    # Optimization
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0

    # Schedule
    max_steps: int = 20_000
    warmup_steps: int = 500
    min_lr_ratio: float = 0.1  # final LR = lr * min_lr_ratio

    # Logging / checkpointing
    log_every: int = 50
    eval_every: int = 500
    save_every: int = 1_000
    output_dir: str = "checkpoints/run"

    # Misc
    num_workers: int = 2
    seed: int = 42
    device: str = "auto"  # auto | cuda | mps | cpu
    use_amp: bool = True


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def pick_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def lr_at_step(step: int, cfg: TrainConfig) -> float:
    """Linear warmup followed by cosine decay to ``min_lr_ratio * lr``."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    # Cosine decay from peak to floor.
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    progress = min(1.0, progress)
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    floor = cfg.lr * cfg.min_lr_ratio
    return floor + (cfg.lr - floor) * cos


def make_dataloaders(
    tcfg: TrainConfig,
    tokenizer: Tokenizer,
) -> tuple[DataLoader, Optional[DataLoader]]:
    collate = make_collate_fn(pad_id=tokenizer.pad_id)
    train_ds = CallResponseDataset(tcfg.cache_path, tokenizer, max_len=tcfg.max_len)
    train_loader = DataLoader(
        train_ds,
        batch_size=tcfg.batch_size,
        shuffle=True,
        num_workers=tcfg.num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    val_loader: Optional[DataLoader] = None
    if tcfg.val_cache_path is not None:
        val_ds = CallResponseDataset(tcfg.val_cache_path, tokenizer, max_len=tcfg.max_len)
        val_loader = DataLoader(
            val_ds,
            batch_size=tcfg.batch_size,
            shuffle=False,
            num_workers=tcfg.num_workers,
            collate_fn=collate,
            pin_memory=torch.cuda.is_available(),
        )
    return train_loader, val_loader


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: RespondAITransformer,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 50,
) -> dict:
    model.eval()
    total_loss = 0.0
    total_tokens = 0.0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        out = model.loss(
            batch["input_ids"].to(device),
            batch["target_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            loss_mask=batch["loss_mask"].to(device),
        )
        n = batch["loss_mask"].sum().item()
        total_loss += out["loss"].item() * n
        total_tokens += n
    model.train()
    avg_loss = total_loss / max(total_tokens, 1.0)
    return {"val_loss": avg_loss, "val_ppl": math.exp(min(avg_loss, 20.0))}


# -----------------------------------------------------------------------------
# Main training entrypoint
# -----------------------------------------------------------------------------

def train(
    tcfg: TrainConfig,
    mcfg: TransformerConfig,
    tokenizer: Tokenizer,
) -> RespondAITransformer:
    """Train a model. Returns the trained model (also written to disk)."""
    torch.manual_seed(tcfg.seed)
    device = pick_device(tcfg.device)
    log.info("Using device: %s", device)

    output_dir = Path(tcfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Persist configs alongside checkpoints for reproducibility.
    with (output_dir / "train_config.json").open("w") as f:
        json.dump(asdict(tcfg), f, indent=2)
    with (output_dir / "model_config.json").open("w") as f:
        json.dump(asdict(mcfg), f, indent=2)

    train_loader, val_loader = make_dataloaders(tcfg, tokenizer)
    model = RespondAITransformer(mcfg).to(device)
    log.info("Model parameters: %.2fM", model.count_parameters() / 1e6)

    optimizer = AdamW(
        model.parameters(),
        lr=tcfg.lr,
        betas=tcfg.betas,
        weight_decay=tcfg.weight_decay,
    )
    use_amp = tcfg.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val = float("inf")
    step = 0
    t0 = time.time()
    model.train()

    # Epoch loop: we count *steps*, not epochs. This way the schedule lines
    # up with TrainConfig.max_steps regardless of dataset size.
    while step < tcfg.max_steps:
        for batch in train_loader:
            if step >= tcfg.max_steps:
                break

            # Schedule LR.
            lr = lr_at_step(step, tcfg)
            for g in optimizer.param_groups:
                g["lr"] = lr

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            target_ids = batch["target_ids"].to(device, non_blocking=True)
            attn_mask = batch["attention_mask"].to(device, non_blocking=True)
            loss_mask = batch["loss_mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model.loss(
                    input_ids,
                    target_ids,
                    attention_mask=attn_mask,
                    loss_mask=loss_mask,
                )
                loss = out["loss"]

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
                optimizer.step()

            if step % tcfg.log_every == 0:
                elapsed = time.time() - t0
                log.info(
                    "step=%d  loss=%.4f  lr=%.2e  elapsed=%.1fs",
                    step, loss.item(), lr, elapsed,
                )

            if val_loader is not None and step > 0 and step % tcfg.eval_every == 0:
                metrics = evaluate(model, val_loader, device)
                log.info("step=%d  val_loss=%.4f  val_ppl=%.2f",
                         step, metrics["val_loss"], metrics["val_ppl"])
                if metrics["val_loss"] < best_val:
                    best_val = metrics["val_loss"]
                    _save_checkpoint(model, optimizer, mcfg, tcfg, step,
                                     output_dir / "best.pt")

            if step > 0 and step % tcfg.save_every == 0:
                _save_checkpoint(model, optimizer, mcfg, tcfg, step,
                                 output_dir / "last.pt")

            step += 1

    # Final save.
    _save_checkpoint(model, optimizer, mcfg, tcfg, step,
                     output_dir / "last.pt")
    log.info("Training complete. step=%d, elapsed=%.1fs", step, time.time() - t0)
    return model


def _save_checkpoint(
    model: RespondAITransformer,
    optimizer: AdamW,
    mcfg: TransformerConfig,
    tcfg: TrainConfig,
    step: int,
    path: Path,
) -> None:
    """Atomic-ish checkpoint save."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "step": step,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "model_config": asdict(mcfg),
            "train_config": asdict(tcfg),
        },
        tmp,
    )
    tmp.replace(path)
    log.info("Saved checkpoint: %s", path)


def load_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> dict:
    """Load a checkpoint. Returns the raw dict; build the model separately."""
    return torch.load(path, map_location=map_location, weights_only=False)
