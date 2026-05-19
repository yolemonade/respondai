"""
Sinusoidal positional encoding (Vaswani et al., 2017).

Kept as a separate module so that swapping in RoPE or learned positions later
is a one-line change in the Transformer.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional embeddings, registered as a buffer.

    Supports an ``offset`` argument so the KV-cached inference path can request
    "give me positions starting at ``offset``", avoiding an extra branch.
    """

    def __init__(self, d_model: int, max_len: int = 4096) -> None:
        super().__init__()
        self.d_model = d_model

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Shape: (max_len, d_model). Stored as buffer so it follows .to(device).
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Add positional encoding to ``x``.

        Parameters
        ----------
        x
            Tensor of shape ``(B, T, d_model)``.
        offset
            Position index of the first token in ``x``. Defaults to 0
            (training); set to ``T_past`` during cached generation.
        """
        T = x.size(1)
        if offset + T > self.pe.size(0):
            raise ValueError(
                f"Requested positions [{offset}, {offset+T}) exceed max_len "
                f"{self.pe.size(0)}. Increase max_len when constructing PE."
            )
        return x + self.pe[offset : offset + T].unsqueeze(0)
