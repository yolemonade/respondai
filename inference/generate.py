"""
Auto-regressive generation
==========================

The primary public function here is :func:`generate`, which is what team B
calls from the Streamlit UI:

    response_tokens, attn_scores = generate(
        call_notes=...,
        key="Dm",
        tempo=92,
        ...,
    )

Internally we:

  1. Tokenize ``(key, tempo, call_notes)`` into the inference prompt.
  2. Run the prompt through the model *once* to populate the KV cache.
  3. Sample one token at a time, feeding only the new token back in. With the
     KV cache active this is O(T) instead of O(T^2).
  4. Stop on ``[EOS]``, ``max_new_tokens``, or when ``num_bars`` bars of
     output have been generated.
  5. Optionally collect per-step average attention to the prompt — used by the
     UI's VIZ-02 visualisation.

Sampling is **top-p (nucleus) + temperature**, which is the most musical
choice in our experience: greedy collapses to repetition, pure top-k can
over-prune at low confidence, and unrestricted temperature spits noise.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from data.tokenizer import Note, STEPS_PER_BAR, Tokenizer
from model.transformer import RespondAITransformer

log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Sampling primitives
# -----------------------------------------------------------------------------

def _top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Zero out (set to -inf) the smallest logits whose softmax mass exceeds
    ``1 - top_p``. Operates on the last dimension."""
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = F.softmax(sorted_logits, dim=-1)
    cum = torch.cumsum(probs, dim=-1)
    # Keep tokens up to (and including) the one whose cumulative mass first
    # exceeds top_p.
    remove = cum > top_p
    # Always keep the top-1.
    remove[..., 0] = False
    # Shift so that the *first* token above the threshold is still kept;
    # we only remove tokens *after* it.
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    # Scatter back to original positions.
    out = torch.full_like(logits, float("-inf"))
    out.scatter_(-1, sorted_idx, sorted_logits)
    return out


def sample_token(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_p: float = 0.9,
    forbidden_ids: Optional[Sequence[int]] = None,
) -> int:
    """Sample one token id from ``logits`` (shape ``(vocab,)``)."""
    if forbidden_ids:
        logits = logits.clone()
        for fid in forbidden_ids:
            logits[fid] = float("-inf")

    if temperature <= 0:
        return int(torch.argmax(logits).item())

    logits = logits / max(temperature, 1e-6)
    logits = _top_p_filter(logits.unsqueeze(0), top_p).squeeze(0)
    probs = F.softmax(logits, dim=-1)
    # Multinomial wants ``probs``, never logits.
    return int(torch.multinomial(probs, num_samples=1).item())


# -----------------------------------------------------------------------------
# High-level generate()
# -----------------------------------------------------------------------------

@dataclass
class GenerationResult:
    """Structured return value of :func:`generate`."""

    response_tokens: List[int]
    response_notes: List[Note]
    attn_scores: List[float]  # per generated token, mean attention back to prompt
    stop_reason: str          # "eos" | "max_tokens" | "max_bars"


@torch.no_grad()
def generate(
    model: RespondAITransformer,
    tokenizer: Tokenizer,
    call_notes: Sequence[Note],
    *,
    key: str,
    tempo: int,
    max_new_tokens: int = 256,
    max_bars: int = 4,
    temperature: float = 1.0,
    top_p: float = 0.9,
    device: Optional[torch.device] = None,
    return_attention: bool = True,
) -> GenerationResult:
    """Generate a RESPONSE for the given CALL.

    This is **the** function team B imports. The interface is intentionally
    decoupled from the model's KV-cache plumbing: callers pass musical inputs
    and receive musical outputs.

    Parameters
    ----------
    model, tokenizer
        Already loaded and (for the model) moved to the right device.
    call_notes
        The user's melody, on the sixteenth-note grid.
    key, tempo
        Musical context. ``key`` must be one of
        :data:`data.tokenizer.KEY_NAMES` (e.g. ``"Dm"``).
    max_new_tokens
        Hard cap on tokens to sample. 256 is generous for a 4-bar response.
    max_bars
        Stop after this many [BAR] tokens have been emitted. This is the
        primary length controller; ``max_new_tokens`` is a safety net.
    temperature, top_p
        Sampling temperature and nucleus-sampling threshold.
    return_attention
        If True, record per-step average attention back to the prompt for
        VIZ-02.
    """
    if device is None:
        device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    prompt = tokenizer.build_prompt(call_notes, key=key, tempo=tempo)
    input_ids = torch.tensor([prompt], dtype=torch.long, device=device)
    prompt_len = input_ids.size(1)
    attn_mask = torch.ones_like(input_ids, dtype=torch.float)

    # Tokens we never want to sample at generation time:
    #   PAD     — would create invalid sequences
    #   BOS / CALL / SEP / RESPONSE  — structural markers, only meaningful in the prompt
    # EOS, BAR, KEY_*, TEMPO_* are valid (EOS terminates; BAR delimits bars).
    forbidden = [
        tokenizer.pad_id,
        tokenizer.bos_id,
        tokenizer.call_id,
        tokenizer.sep_id,
        tokenizer.response_id,
    ]

    # --- prefill: run the whole prompt once to seed the KV cache --------------
    out = model(
        input_ids,
        attention_mask=attn_mask,
        return_attn=return_attention,
    )
    past_kv = out["past_kv"]
    last_logits = out["logits"][0, -1]

    # First sampling step uses the prompt's final-position logits.
    generated: List[int] = []
    attn_scores: List[float] = []
    bars_seen = 0
    stop_reason = "max_tokens"

    next_id = sample_token(
        last_logits,
        temperature=temperature,
        top_p=top_p,
        forbidden_ids=forbidden,
    )
    if return_attention:
        # Mean attention from the *next* position back to the prompt would
        # require an extra forward; we approximate with the final-position
        # row of the prefill's attention. Cheap and informative for VIZ-02.
        a = out["attn_avg"][0, -1, :prompt_len].mean().item()
        attn_scores.append(float(a))

    # --- decoding loop --------------------------------------------------------
    for _ in range(max_new_tokens):
        generated.append(next_id)
        if next_id == tokenizer.eos_id:
            stop_reason = "eos"
            break
        if next_id == tokenizer.bar_id:
            bars_seen += 1
            if bars_seen >= max_bars + 1:
                # We've passed max_bars completed bars (the +1 accounts for the
                # initial leading [BAR] token before any note).
                stop_reason = "max_bars"
                break

        new_input = torch.tensor([[next_id]], dtype=torch.long, device=device)
        # Attention mask must cover both cached and new positions.
        cur_len = prompt_len + len(generated)
        attn_mask = torch.ones((1, cur_len), dtype=torch.float, device=device)

        out = model(
            new_input,
            attention_mask=attn_mask,
            past_kv=past_kv,
            return_attn=return_attention,
        )
        past_kv = out["past_kv"]
        last_logits = out["logits"][0, -1]

        if return_attention:
            # Attention from the new token to the prompt-only positions.
            a = out["attn_avg"][0, -1, :prompt_len].mean().item()
            attn_scores.append(float(a))

        next_id = sample_token(
            last_logits,
            temperature=temperature,
            top_p=top_p,
            forbidden_ids=forbidden,
        )
    else:
        # for-loop exhausted without break
        generated.append(next_id)

    response_notes = tokenizer.decode_notes(generated)

    if was_training:
        model.train()
    return GenerationResult(
        response_tokens=generated,
        response_notes=response_notes,
        attn_scores=attn_scores,
        stop_reason=stop_reason,
    )


# -----------------------------------------------------------------------------
# Lightweight loader, so the UI doesn't need to know model internals
# -----------------------------------------------------------------------------

def load_model_for_inference(
    checkpoint_path: str,
    *,
    device: str | torch.device = "auto",
) -> tuple[RespondAITransformer, Tokenizer, torch.device]:
    """Reconstruct a model from a checkpoint and put it in eval mode.

    Convenience wrapper for downstream code (and team B's Streamlit app).
    """
    from dataclasses import fields

    if device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    mcfg_dict = ckpt["model_config"]
    from model.transformer import TransformerConfig
    # Filter only known fields, in case the checkpoint was made with an
    # older / newer version of the dataclass.
    known = {f.name for f in fields(TransformerConfig)}
    mcfg = TransformerConfig(**{k: v for k, v in mcfg_dict.items() if k in known})

    tokenizer = Tokenizer()
    if tokenizer.vocab_size != mcfg.vocab_size:
        log.warning(
            "Tokenizer vocab (%d) differs from checkpoint vocab (%d); "
            "using checkpoint value.",
            tokenizer.vocab_size, mcfg.vocab_size,
        )

    model = RespondAITransformer(mcfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    model = torch.compile(model, mode="reduce-overhead")
    return model, tokenizer, device
