"""
Model evaluation
================

Two flavours of metric:

* **Token-level**: perplexity on a held-out cache. Standard LM measure.
* **Sample-level**: generate responses for a batch of CALLs and average the
  game scores (:func:`analysis.scoring.score_response`). This is closer to
  the real downstream metric and catches degenerate-but-low-PPL failures
  (e.g. always-rest outputs).
"""
from __future__ import annotations

import logging
import math
from typing import List, Optional, Sequence

import torch
from torch.utils.data import DataLoader

from analysis.scoring import score_response
from data.dataset import CallResponseDataset, make_collate_fn
from data.tokenizer import Tokenizer
from inference.generate import generate
from model.transformer import RespondAITransformer

log = logging.getLogger(__name__)


@torch.no_grad()
def perplexity(
    model: RespondAITransformer,
    cache_path: str,
    tokenizer: Tokenizer,
    *,
    batch_size: int = 16,
    max_batches: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> dict:
    """Compute response-region perplexity on a cached dataset.

    Only RESPONSE tokens contribute to the score, matching how the model is
    trained.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    ds = CallResponseDataset(cache_path, tokenizer)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=make_collate_fn(tokenizer.pad_id),
    )

    total_loss = 0.0
    total_tokens = 0.0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
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

    avg_loss = total_loss / max(total_tokens, 1.0)
    ppl = math.exp(min(avg_loss, 20.0))
    return {"loss": avg_loss, "ppl": ppl, "tokens": total_tokens}


@torch.no_grad()
def evaluate_samples(
    model: RespondAITransformer,
    cache_path: str,
    tokenizer: Tokenizer,
    *,
    n_samples: int = 50,
    temperature: float = 1.0,
    top_p: float = 0.9,
    device: Optional[torch.device] = None,
) -> dict:
    """Generate responses for ``n_samples`` cached CALLs and average game scores.

    Slower than :func:`perplexity` (one inference per sample) so kept on a
    small budget by default.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    ds = CallResponseDataset(cache_path, tokenizer)
    # Take the first n_samples for reproducibility; if you want random,
    # set a seed and shuffle yourself.
    n_samples = min(n_samples, len(ds))

    key_total = 0
    rhythm_total = 0
    motif_total = 0
    creativity_total = 0
    grand_total = 0

    for i in range(n_samples):
        seq = ds._raw_sequence(i).tolist()
        call_body, _ = tokenizer.split_call_response(seq)
        call_notes = tokenizer.decode_notes(call_body)
        if not call_notes:
            continue

        meta = ds.meta[i]
        result = generate(
            model, tokenizer, call_notes,
            key=meta["key"], tempo=meta["tempo"],
            temperature=temperature, top_p=top_p,
            return_attention=False,
        )
        score = score_response(call_notes, result.response_notes, key=meta["key"])
        key_total += score["key_consistency"]
        rhythm_total += score["rhythm_similarity"]
        motif_total += score["motif_usage"]
        creativity_total += score["creativity_bonus"]
        grand_total += score["total"]

    return {
        "n_samples": n_samples,
        "avg_key": key_total / n_samples,
        "avg_rhythm": rhythm_total / n_samples,
        "avg_motif": motif_total / n_samples,
        "avg_creativity": creativity_total / n_samples,
        "avg_total": grand_total / n_samples,
    }
