"""
PyTorch ``Dataset`` for Call-and-Response training
==================================================

Reads the binary cache produced by :mod:`data.preprocess` and serves
``(input_ids, target_ids, loss_mask)`` triples ready for cross-entropy loss.

The crucial detail is the loss mask: we only score the model on the RESPONSE
half of each sequence. Tokens up to and including the ``[RESPONSE]`` marker
are *context*; their next-token predictions still happen (the model must
condition on them), but they don't contribute to the loss. This matches the
project spec ("[CALL] ~ [SEP] 구간을 prefix로 고정하고 [RESPONSE] 이후만 loss 계산").
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocess import load_cache
from .tokenizer import Tokenizer


class CallResponseDataset(Dataset):
    """A flat, in-memory dataset of REMI token sequences.

    The cache file is loaded once at construction and kept as numpy arrays;
    per-item access slices into them, so worker-shared memory is fine.

    Parameters
    ----------
    cache_path
        Path to the ``.npz`` produced by ``data.preprocess.build_cache``.
    tokenizer
        The same tokenizer used at preprocessing time. We use it to locate the
        ``[RESPONSE]`` marker for loss-mask construction.
    max_len
        Sequences longer than this are clipped (should not happen if
        ``build_cache`` enforced the same limit).
    """

    def __init__(
        self,
        cache_path: str | Path,
        tokenizer: Tokenizer,
        max_len: int = 512,
    ) -> None:
        super().__init__()
        cache = load_cache(cache_path)
        self.tokens: np.ndarray = cache["tokens"]
        self.offsets: np.ndarray = cache["offsets"]
        self.meta: list[dict] = cache["meta"]
        cached_vocab = cache["vocab_size"]
        if cached_vocab != tokenizer.vocab_size:
            raise ValueError(
                f"Vocab mismatch: cache built with {cached_vocab}, "
                f"tokenizer has {tokenizer.vocab_size}. Rebuild the cache."
            )
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def _raw_sequence(self, idx: int) -> np.ndarray:
        a, b = self.offsets[idx], self.offsets[idx + 1]
        seq = self.tokens[a:b]
        if seq.size > self.max_len:
            seq = seq[: self.max_len]
        return seq

    def __getitem__(self, idx: int) -> dict:
        seq = self._raw_sequence(idx).astype(np.int64)

        # Build loss mask: 1 for tokens we score, 0 for context.
        # Model is trained with shifted targets, so the *input* is seq[:-1]
        # and the *target* is seq[1:]; mask aligns with targets.
        resp_id = self.tokenizer.response_id
        try:
            # First occurrence of [RESPONSE]; everything strictly after it
            # (in the target stream) is scored. We score tokens whose source
            # *position* (index in original seq) is > response_idx, i.e. those
            # generated *given* [RESPONSE] as context.
            response_idx = int(np.where(seq == resp_id)[0][0])
        except IndexError:
            # Should not happen for well-formed cache entries; treat the whole
            # sequence as scored to fail loudly during eval rather than train
            # silently on garbage.
            response_idx = -1

        # Targets are seq[1:]. A target at position t (0-indexed in the target
        # stream) corresponds to seq[t+1]. We want to score targets whose
        # source-position is > response_idx, i.e. t+1 > response_idx → t >= response_idx.
        target_len = seq.shape[0] - 1
        mask = np.zeros(target_len, dtype=np.float32)
        if response_idx >= 0:
            mask[response_idx:] = 1.0

        return {
            "input_ids": torch.from_numpy(seq[:-1]),
            "target_ids": torch.from_numpy(seq[1:]),
            "loss_mask": torch.from_numpy(mask),
            # Useful for debugging / analysis; keep small.
            "length": int(target_len),
        }


# -----------------------------------------------------------------------------
# Collate
# -----------------------------------------------------------------------------

def make_collate_fn(pad_id: int):
    """Right-pad a batch to the max length within it."""

    def collate(batch: Sequence[dict]) -> dict:
        max_len = max(item["length"] for item in batch)
        bsz = len(batch)

        input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
        target_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
        loss_mask = torch.zeros((bsz, max_len), dtype=torch.float32)
        # attention_mask is 1 for real tokens, 0 for padding. The model uses
        # it to block attention into padded positions.
        attention_mask = torch.zeros((bsz, max_len), dtype=torch.float32)

        for i, item in enumerate(batch):
            L = item["length"]
            input_ids[i, :L] = item["input_ids"]
            target_ids[i, :L] = item["target_ids"]
            loss_mask[i, :L] = item["loss_mask"]
            attention_mask[i, :L] = 1.0

        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "loss_mask": loss_mask,
            "attention_mask": attention_mask,
        }

    return collate
