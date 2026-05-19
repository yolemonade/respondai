"""
Named model presets matching the project spec.

Two profiles ship:

* ``sanity_config(vocab_size)`` — ~3M params. Used for fast iteration on
  Nottingham; should train to a non-trivial loss within a few minutes on any
  GPU. The point is to catch bugs, not to make music.

* ``full_config(vocab_size)`` — ~10M params. The real model. Train this on
  Lakh MIDI after sanity has passed.

Override individual fields as needed; both are dataclasses, not frozen.
"""
from __future__ import annotations

from .transformer import TransformerConfig


def sanity_config(vocab_size: int, *, pad_id: int = 0) -> TransformerConfig:
    return TransformerConfig(
        vocab_size=vocab_size,
        n_layers=4,
        n_heads=4,
        d_model=256,
        d_ff=512,
        max_seq_len=512,
        dropout=0.1,
        attn_dropout=0.1,
        tie_embeddings=False,
        pad_id=pad_id,
    )


def full_config(vocab_size: int, *, pad_id: int = 0) -> TransformerConfig:
    return TransformerConfig(
        vocab_size=vocab_size,
        n_layers=6,
        n_heads=8,
        d_model=512,
        d_ff=2048,
        max_seq_len=512,
        dropout=0.1,
        attn_dropout=0.1,
        tie_embeddings=False,
        pad_id=pad_id,
    )
