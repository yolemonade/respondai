"""
Call-and-Response pair construction
===================================

Slide a window over a monophonic melody to produce training pairs of the form
``(call_notes, response_notes, key, tempo)``. The pair builder enforces:

  * both halves are non-empty,
  * pitch range fits the tokenizer,
  * the response is in the same key as the call (we estimate the key per
    window, not per song, to allow modulations to live in their own segments
    rather than be dropped wholesale).

Key estimation uses ``music21``'s Krumhansl-Schmuckler analyzer, which is
robust for short fragments and runs in pure Python.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence

import music21

from .midi_utils import num_bars, slice_bars
from .tokenizer import KEY_NAMES, Note


@dataclass
class CRPair:
    """A single Call-and-Response training example."""

    call: List[Note]
    response: List[Note]
    key: str
    tempo: int


# -----------------------------------------------------------------------------
# Key estimation
# -----------------------------------------------------------------------------

_KEY_NAME_FIX = {
    # music21 uses flats; our vocabulary uses sharps. Map enharmonic spellings.
    "D-": "C#",
    "E-": "D#",
    "G-": "F#",
    "A-": "G#",
    "B-": "A#",
}


def estimate_key(notes: Sequence[Note]) -> Optional[str]:
    """Return the most likely key for a note sequence, or ``None`` on failure.

    Result is in our canonical 24-key vocabulary (e.g. ``"C"``, ``"F#m"``).
    """
    if not notes:
        return None

    stream = music21.stream.Stream()
    for n in notes:
        m21n = music21.note.Note(n.pitch)
        m21n.quarterLength = max(0.25, (n.end - n.start) / 4.0)
        stream.append(m21n)

    try:
        k = stream.analyze("key")
    except Exception:
        return None

    tonic = k.tonic.name  # e.g. "C", "D-"
    # Strip the octave-style sharp/flat to single character + accidental.
    tonic = _KEY_NAME_FIX.get(tonic, tonic.replace("-", "b"))
    # If we ended up with a flat that has no enharmonic mapping above, drop it.
    if "b" in tonic:
        return None

    mode = k.mode  # "major" or "minor"
    name = tonic if mode == "major" else f"{tonic}m"
    return name if name in KEY_NAMES else None


# -----------------------------------------------------------------------------
# Pair generation
# -----------------------------------------------------------------------------

def generate_pairs(
    notes: Sequence[Note],
    tempo: int,
    *,
    window_bars: int = 4,
    stride_bars: int = 2,
    min_notes_per_half: int = 4,
    require_same_key: bool = True,
) -> Iterator[CRPair]:
    """Yield :class:`CRPair` instances from a single melody.

    Default settings produce overlapping 4-bar call / 4-bar response pairs with
    a 2-bar hop, matching the project spec.
    """
    total = num_bars(notes)
    if total < 2 * window_bars:
        return

    for start in range(0, total - 2 * window_bars + 1, stride_bars):
        call = slice_bars(notes, start, window_bars)
        resp = slice_bars(notes, start + window_bars, window_bars)
        if len(call) < min_notes_per_half or len(resp) < min_notes_per_half:
            continue

        key_call = estimate_key(call)
        if key_call is None:
            continue
        if require_same_key:
            key_resp = estimate_key(resp)
            if key_resp != key_call:
                continue

        yield CRPair(
            call=call,
            response=resp,
            key=key_call,
            tempo=tempo,
        )
