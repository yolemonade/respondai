"""
Motif analysis
==============

We measure how much of the CALL's motivic material survives into the RESPONSE.
The metric is **n-gram pitch-interval overlap**: we extract every contiguous
sliding window of ``n`` notes from each side, convert it to its interval
sequence (relative pitches; transposition-invariant), and compute the fraction
of CALL n-grams that also appear in RESPONSE.

Why intervals, not absolute pitches?
  * The model often answers a phrase by transposing it (a classic call-and-
    response move). Absolute matching would miss this.
  * Two melodies in the same key with the same shape have identical interval
    sequences.

Default ``n=3`` (three notes → two intervals). With shorter windows we'd
match by accident, with longer ones we'd rarely match at all.
"""
from __future__ import annotations

from collections import Counter
from typing import List, Sequence, Tuple

from data.tokenizer import Note


def pitch_intervals(notes: Sequence[Note]) -> List[int]:
    """Return the sequence of consecutive pitch intervals (semitones)."""
    return [b.pitch - a.pitch for a, b in zip(notes, notes[1:])]


def ngrams(seq: Sequence[int], n: int) -> List[Tuple[int, ...]]:
    if len(seq) < n:
        return []
    return [tuple(seq[i : i + n]) for i in range(len(seq) - n + 1)]


def motif_overlap(
    call: Sequence[Note],
    response: Sequence[Note],
    *,
    n: int = 3,
) -> float:
    """Fraction of CALL n-grams that also appear in RESPONSE.

    Each n-gram is an interval window of length ``n - 1``. Returns ``0.0`` if
    the CALL is too short to form any n-gram, on the assumption that the
    score should not be rewarded for trivial CALLs.
    """
    call_grams = ngrams(pitch_intervals(call), n - 1)
    resp_grams = ngrams(pitch_intervals(response), n - 1)
    if not call_grams:
        return 0.0
    resp_set = set(resp_grams)
    hits = sum(1 for g in call_grams if g in resp_set)
    return hits / len(call_grams)


def motif_overlap_weighted(
    call: Sequence[Note],
    response: Sequence[Note],
    *,
    n: int = 3,
) -> float:
    """Like :func:`motif_overlap` but weights by frequency in the CALL.

    A motif that occurs twice in the CALL counts twice towards the
    denominator and twice if it's matched. Slightly more musical for
    repetitive themes (think Beethoven's 5th).
    """
    call_grams = ngrams(pitch_intervals(call), n - 1)
    resp_grams = ngrams(pitch_intervals(response), n - 1)
    if not call_grams:
        return 0.0
    resp_counter = Counter(resp_grams)
    call_counter = Counter(call_grams)
    hits = 0
    for g, c in call_counter.items():
        hits += min(c, resp_counter.get(g, 0))
    return hits / len(call_grams)
