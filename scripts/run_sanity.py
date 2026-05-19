"""
Sanity-check training run.

Trains the small (~3M) model on the Nottingham cache for a fixed number of
steps and reports loss. The goal is to confirm that the entire pipeline
(tokenizer → dataset → model → optimizer → checkpointing) is wired correctly
*before* spending money on Lakh MIDI.

Expected behaviour
------------------
On Nottingham, with default settings:

  * step    0: loss ~ ln(180) ≈ 5.2 (random)
  * step  500: loss ~ 2.5
  * step 2000: loss < 1.5

If you don't see that curve, something is wrong (typically the loss mask).

Usage:
    python -m scripts.run_sanity --cache data_cache/nottingham.npz
"""
from __future__ import annotations

import argparse
import logging

from data.tokenizer import Tokenizer
from model.config import sanity_config
from training.train import TrainConfig, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, help="Path to the tokenized cache .npz")
    parser.add_argument("--val-cache", default=None,
                        help="Optional held-out cache for validation.")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output-dir", default="checkpoints/sanity")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    tokenizer = Tokenizer()
    mcfg = sanity_config(tokenizer.vocab_size, pad_id=tokenizer.pad_id)
    tcfg = TrainConfig(
        cache_path=args.cache,
        val_cache_path=args.val_cache,
        batch_size=args.batch_size,
        lr=args.lr,
        max_steps=args.steps,
        # Tighter logging cadence for a short run.
        log_every=20,
        eval_every=200,
        save_every=500,
        output_dir=args.output_dir,
        device=args.device,
    )

    train(tcfg, mcfg, tokenizer)


if __name__ == "__main__":
    main()
