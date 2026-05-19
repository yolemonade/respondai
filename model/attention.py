"""
Causal multi-head self-attention
================================

Pure-PyTorch implementation. We deliberately avoid ``nn.MultiheadAttention``
and ``F.scaled_dot_product_attention`` shortcuts because the assignment is
"implement attention from scratch". However, the math here is exactly what
those library functions compute.

Shape conventions
-----------------
``B``  batch size,
``T``  sequence length,
``H``  number of attention heads,
``D``  per-head dimension (so total embedding dim is ``H * D``).

Inputs are always ``(B, T, H*D)``; outputs match.

KV cache
--------
For inference we expose an optional ``past_kv`` argument that lets the
generation loop reuse previously computed keys and values. The cache is a
``(K, V)`` tuple where both tensors have shape ``(B, H, T_past, D)``.
Returning the new cache is the caller's responsibility (see :meth:`forward`).
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Single block of multi-head causal self-attention."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})."
            )
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Single fused projection for Q, K, V. Slightly faster and identical
        # in expressiveness to three separate Linear layers.
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.resid_drop = nn.Dropout(dropout)
        self.attn_drop = nn.Dropout(attn_dropout)

    def forward(
        self,
        x: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_attn: bool = False,
    ) -> dict:
        """Apply causal self-attention to ``x``.

        Parameters
        ----------
        x
            Float tensor of shape ``(B, T, d_model)``.
        attention_mask
            Optional float tensor ``(B, T_total)`` with 1 for real tokens and 0
            for padding (where ``T_total = T_past + T``). Padded positions are
            blocked from being attended to.
        past_kv
            Optional ``(K, V)`` cache from previous steps; each ``(B, H, T_past, D)``.
        return_attn
            If True, also return the post-softmax attention weights averaged
            across heads, shape ``(B, T, T_total)``. Used for visualisation.

        Returns
        -------
        dict with keys:
          - ``out``: ``(B, T, d_model)``
          - ``kv``:  ``(K, V)`` for caching, each ``(B, H, T_total, D)``
          - ``attn``: optional, shape ``(B, T, T_total)``
        """
        B, T, _ = x.shape

        qkv = self.qkv(x)  # (B, T, 3*d_model)
        q, k, v = qkv.chunk(3, dim=-1)

        # (B, T, H*D) → (B, H, T, D)
        def _reshape(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q = _reshape(q)
        k = _reshape(k)
        v = _reshape(v)

        # Concatenate cached keys/values, if any. This is how an
        # auto-regressive generator avoids recomputing the whole prefix.
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        T_total = k.size(2)
        T_past = T_total - T

        # Scaled dot-product. (B, H, T, T_total)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Causal mask: query at position i (in the *new* segment) can attend
        # to keys at positions [0, T_past + i]. We build the mask in one shot.
        # `causal[i, j] = True` when query i must be blocked from key j.
        device = scores.device
        q_pos = torch.arange(T, device=device).unsqueeze(1) + T_past  # (T, 1)
        k_pos = torch.arange(T_total, device=device).unsqueeze(0)     # (1, T_total)
        causal = k_pos > q_pos  # (T, T_total), True = blocked
        scores = scores.masked_fill(causal[None, None, :, :], float("-inf"))

        # Padding mask: block attention into padded *key* positions.
        if attention_mask is not None:
            # attention_mask: (B, T_total), 1 = real, 0 = pad
            pad_block = attention_mask == 0  # (B, T_total)
            scores = scores.masked_fill(
                pad_block[:, None, None, :], float("-inf")
            )

        # Numerical safety: a row that is entirely -inf (can happen when a
        # padded query attends only to padded keys, e.g. on padding-only
        # batches) would yield NaN after softmax. Replace such rows with 0.
        attn = F.softmax(scores, dim=-1)
        # Detect all-NaN rows and zero them. Cheaper than masking before
        # softmax for the common case where no row is fully blocked.
        attn = torch.nan_to_num(attn, nan=0.0)
        attn_dropped = self.attn_drop(attn)

        # (B, H, T, T_total) @ (B, H, T_total, D) → (B, H, T, D)
        out = torch.matmul(attn_dropped, v)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.resid_drop(self.out_proj(out))

        result = {"out": out, "kv": (k, v)}
        if return_attn:
            # Average over heads → (B, T, T_total). Cheap, useful for VIZ-02.
            result["attn"] = attn.mean(dim=1).detach()
        return result
