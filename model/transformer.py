"""
Decoder-only Transformer for symbolic music generation.

Architecture
------------
We use the Pre-LayerNorm variant of the standard Transformer block — i.e.
LayerNorm sits *before* each sub-layer, with a residual connection added
after. Pre-LN trains much more stably than the original Post-LN for the
model sizes we care about (~3M–10M), and matches GPT-2 / LLaMA in spirit.

Per block::

    x ← x + Attention(LN(x))
    x ← x + MLP(LN(x))

A final LayerNorm is applied before the output projection. The output
projection is *not* tied to the embedding (we have plenty of vocab budget
and untied gives a tiny perplexity win).

KV cache contract
-----------------
``forward(input_ids, past_kv=None, return_attn=False)`` returns a dict with::

    logits   : (B, T, vocab)
    past_kv  : list of (K, V) tuples, one per layer
    attn_avg : optional, (B, T, T_total) averaged across heads & layers

Pass ``past_kv`` back in on the next call (along with only the new tokens)
to skip recomputing the prefix. This is what :func:`inference.generate.generate`
does step-by-step.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import CausalSelfAttention
from .positional import SinusoidalPositionalEncoding


@dataclass
class TransformerConfig:
    """Hyperparameters for :class:`RespondAITransformer`.

    Two named presets live in :mod:`model.config`. Anything else is a custom
    sweep point.
    """

    vocab_size: int
    n_layers: int = 6
    n_heads: int = 8
    d_model: int = 512
    d_ff: int = 2048
    max_seq_len: int = 512
    dropout: float = 0.1
    attn_dropout: float = 0.1
    tie_embeddings: bool = False
    pad_id: int = 0  # used to ignore_index in cross-entropy

    def n_params_estimate(self) -> int:
        """Rough parameter count, ignoring bias terms and LayerNorm."""
        emb = self.vocab_size * self.d_model
        attn = 4 * self.d_model * self.d_model  # qkv (3) + out_proj (1)
        ffn = 2 * self.d_model * self.d_ff
        per_block = attn + ffn
        head = 0 if self.tie_embeddings else self.vocab_size * self.d_model
        return emb + self.n_layers * per_block + head


# -----------------------------------------------------------------------------
# Building blocks
# -----------------------------------------------------------------------------

class FeedForward(nn.Module):
    """Two-layer MLP with GELU. Standard Transformer FFN."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    """Pre-LN attention + FFN with residual connections."""

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            dropout=cfg.dropout,
            attn_dropout=cfg.attn_dropout,
        )
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_attn: bool = False,
    ) -> dict:
        attn_out = self.attn(
            self.ln1(x),
            attention_mask=attention_mask,
            past_kv=past_kv,
            return_attn=return_attn,
        )
        x = x + attn_out["out"]
        x = x + self.ffn(self.ln2(x))
        result = {"out": x, "kv": attn_out["kv"]}
        if return_attn:
            result["attn"] = attn_out["attn"]
        return result


# -----------------------------------------------------------------------------
# Full model
# -----------------------------------------------------------------------------

class RespondAITransformer(nn.Module):
    """Decoder-only Transformer LM over our REMI vocabulary.

    See module docstring for the KV-cache contract.
    """

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_enc = SinusoidalPositionalEncoding(cfg.d_model, max_len=cfg.max_seq_len)
        self.emb_drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.ln_final = nn.LayerNorm(cfg.d_model)

        if cfg.tie_embeddings:
            self.lm_head = None  # use tok_emb.weight.T at call time
        else:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self._init_weights()

    # -- initialisation --------------------------------------------------------

    def _init_weights(self) -> None:
        """GPT-style initialisation: linear layers N(0, 0.02), embeddings same,
        LayerNorm at (1, 0). Output projection of each block is scaled down to
        improve early-training stability with deep stacks (residual scaling).
        """
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)
            elif "bias" in name:
                nn.init.zeros_(p)

        # Residual scaling, à la GPT-2.
        for block in self.blocks:
            nn.init.normal_(
                block.attn.out_proj.weight, mean=0.0,
                std=0.02 / (2 * self.cfg.n_layers) ** 0.5,
            )
            nn.init.normal_(
                block.ffn.fc2.weight, mean=0.0,
                std=0.02 / (2 * self.cfg.n_layers) ** 0.5,
            )

    # -- forward ---------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        past_kv: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        return_attn: bool = False,
    ) -> dict:
        """Compute logits over the vocabulary.

        Parameters
        ----------
        input_ids
            Long tensor ``(B, T)`` of token ids.
        attention_mask
            Optional float tensor ``(B, T_total)`` — 1 for real tokens, 0 for
            pad. Must cover *both* cached and new positions when ``past_kv``
            is supplied.
        past_kv
            List of ``(K, V)`` caches from a previous call, one per layer.
        return_attn
            If True, return per-layer attention weights averaged across heads,
            plus a global average across layers, for visualisation.
        """
        B, T = input_ids.shape
        T_past = past_kv[0][0].size(2) if past_kv is not None else 0

        x = self.tok_emb(input_ids)
        x = self.pos_enc(x, offset=T_past)
        x = self.emb_drop(x)

        new_kv: List[Tuple[torch.Tensor, torch.Tensor]] = []
        attns: List[torch.Tensor] = []

        for i, block in enumerate(self.blocks):
            layer_past = past_kv[i] if past_kv is not None else None
            block_out = block(
                x,
                attention_mask=attention_mask,
                past_kv=layer_past,
                return_attn=return_attn,
            )
            x = block_out["out"]
            new_kv.append(block_out["kv"])
            if return_attn:
                attns.append(block_out["attn"])

        x = self.ln_final(x)

        if self.lm_head is not None:
            logits = self.lm_head(x)
        else:
            logits = x @ self.tok_emb.weight.T

        out = {"logits": logits, "past_kv": new_kv}
        if return_attn:
            # Stack & average across layers → (B, T, T_total)
            out["attn_avg"] = torch.stack(attns, dim=0).mean(dim=0)
            out["attn_per_layer"] = attns
        return out

    # -- convenience -----------------------------------------------------------

    @torch.no_grad()
    def count_parameters(self, trainable_only: bool = True) -> int:
        return sum(
            p.numel() for p in self.parameters() if (p.requires_grad or not trainable_only)
        )

    def loss(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """Cross-entropy loss, optionally masked to the RESPONSE region.

        ``loss_mask`` (float, ``(B, T)``) is multiplied with the per-token loss
        before averaging. Tokens with mask=0 do not contribute. We always
        ignore the pad id as a safety net.
        """
        out = self(input_ids, attention_mask=attention_mask)
        logits = out["logits"]  # (B, T, V)
        B, T, V = logits.shape

        ce = F.cross_entropy(
            logits.reshape(B * T, V),
            target_ids.reshape(B * T),
            ignore_index=self.cfg.pad_id,
            reduction="none",
        ).reshape(B, T)

        if loss_mask is not None:
            ce = ce * loss_mask
            denom = loss_mask.sum().clamp(min=1.0)
        else:
            denom = (target_ids != self.cfg.pad_id).float().sum().clamp(min=1.0)

        loss = ce.sum() / denom
        return {"loss": loss, "logits": logits}
