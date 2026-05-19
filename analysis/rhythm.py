"""
Rhythm similarity
=================

We compare CALL and RESPONSE rhythms by sampling them onto a 16-step-per-bar
binary onset grid and computing the Pearson correlation between the two
vectors. Returns a value in ``[-1, 1]``; the scoring layer rescales it to a
non-negative score.

Why this representation?
  * It captures *when* notes start, not *which* notes they are. That's the
    right granularity for "did the response keep the groove?".
  * Pearson is symmetric and degenerates gracefully when one side is silent
    (we return 0.0 instead of NaN).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from data.tokenizer import Note, STEPS_PER_BAR


def onset_vector(notes: Sequence[Note], num_bars: int) -> np.ndarray:
    """Build a binary onset vector of length ``num_bars * STEPS_PER_BAR``."""
    length = num_bars * STEPS_PER_BAR
    v = np.zeros(length, dtype=np.float32)
    for n in notes:
        if 0 <= n.start < length:
            v[n.start] = 1.0
    return v


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation, returning 0.0 if either side has zero variance."""
    if a.std() < 1e-8 or b.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def rhythm_similarity(
    call: Sequence[Note],
    response: Sequence[Note],
    *,
    num_bars: int = 4,
) -> float:
    """Pearson correlation between CALL and RESPONSE onset vectors.

    ``num_bars`` should match the segment length used at training time
    (4 bars in our setup).
    """
    a = onset_vector(call, num_bars)
    b = onset_vector(response, num_bars)
    return _safe_pearson(a, b)
