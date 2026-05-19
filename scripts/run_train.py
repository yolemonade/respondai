"""
Full training run.

Trains the ~10M model on the main cache (typically Lakh MIDI). Designed to
fit a single A100 (RunPod Community) or T4 (Colab Pro) within a few hours.

Usage:
    python -m scripts.run_train \
        --cache data_cache/lakh.npz \
        --val-cache data_cache/lakh_val.npz \
        --steps 20000

Resume from an existing checkpoint by passing ``--resume PATH``.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import fields

import torch

from data.tokenizer import Tokenizer
from model.config import full_config
from model.transformer import RespondAITransformer, TransformerConfig
from training.train import TrainConfig, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--val-cache", default=None)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output-dir", default="checkpoints/full")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", default=None,
                        help="Checkpoint .pt to resume from.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    tokenizer = Tokenizer()
    mcfg = full_config(tokenizer.vocab_size, pad_id=tokenizer.pad_id)
    tcfg = TrainConfig(
        cache_path=args.cache,
        val_cache_path=args.val_cache,
        batch_size=args.batch_size,
        lr=args.lr,
        max_steps=args.steps,
        output_dir=args.output_dir,
        device=args.device,
    )

    if args.resume:
        # Hot-start path: we don't yet plumb resume through train(), so this is
        # a stub for future-you. Build the model, load state, then call train()
        # with a custom starting step.
        logging.warning("--resume is not yet implemented end-to-end; this is a stub.")

    train(tcfg, mcfg, tokenizer)


if __name__ == "__main__":
    main()
