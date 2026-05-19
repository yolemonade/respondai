"""
MIDI utilities
==============

Helpers for turning arbitrary MIDI files into clean monophonic ``Note``
sequences on a sixteenth-note grid.

The pipeline is intentionally tolerant: malformed files, polyphonic tracks,
out-of-range pitches, and tempo changes are all handled by either filtering
the offending content or falling back to a sensible default. Anything we
cannot recover from is reported by raising ``MidiParseError``; callers should
treat that as "skip this file".
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import pretty_midi

from .tokenizer import Note, PITCH_MAX, PITCH_MIN, STEPS_PER_BAR


class MidiParseError(Exception):
    """Raised when a MIDI file cannot be processed into a usable melody."""


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------

def load_midi(path: str | Path) -> pretty_midi.PrettyMIDI:
    """Load a MIDI file, raising ``MidiParseError`` on any failure."""
    try:
        return pretty_midi.PrettyMIDI(str(path))
    except Exception as exc:  # pretty_midi raises a broad set of errors
        raise MidiParseError(f"Failed to load {path}: {exc}") from exc


# -----------------------------------------------------------------------------
# Monophonic extraction
# -----------------------------------------------------------------------------

def is_drum_instrument(inst: pretty_midi.Instrument) -> bool:
    return inst.is_drum or inst.program > 119  # >119 is mostly SFX/percussion


def polyphony_score(inst: pretty_midi.Instrument) -> float:
    """Fraction of time during which two or more notes overlap.

    ``0.0`` means perfectly monophonic; ``1.0`` means always polyphonic.
    Cheap O(n log n) sweep-line implementation.
    """
    notes = sorted(inst.notes, key=lambda n: n.start)
    if not notes:
        return 0.0

    events: List[tuple[float, int]] = []
    for n in notes:
        events.append((n.start, +1))
        events.append((n.end, -1))
    events.sort()

    active = 0
    poly_time = 0.0
    mono_time = 0.0
    prev_t = events[0][0]
    for t, delta in events:
        if active >= 2:
            poly_time += t - prev_t
        elif active >= 1:
            mono_time += t - prev_t
        active += delta
        prev_t = t

    total = poly_time + mono_time
    return poly_time / total if total > 0 else 0.0


def pick_melody_track(
    pm: pretty_midi.PrettyMIDI,
    max_polyphony: float = 0.05,
    min_notes: int = 16,
) -> pretty_midi.Instrument:
    """Choose the best monophonic-ish track from a multi-track MIDI file.

    Heuristic:
      1. Drop drum tracks and ones with too few notes.
      2. Among the remaining, prefer tracks whose ``polyphony_score`` is below
         ``max_polyphony``.
      3. Tie-break by note count (longer tracks usually carry the melody).

    Raises ``MidiParseError`` if no suitable track exists.
    """
    candidates: list[tuple[float, int, pretty_midi.Instrument]] = []
    for inst in pm.instruments:
        if is_drum_instrument(inst):
            continue
        if len(inst.notes) < min_notes:
            continue
        score = polyphony_score(inst)
        candidates.append((score, len(inst.notes), inst))

    if not candidates:
        raise MidiParseError("No usable instrument tracks found.")

    # Prefer monophonic, then most notes.
    candidates.sort(key=lambda x: (x[0] > max_polyphony, x[0], -x[1]))
    return candidates[0][2]


def force_monophonic(notes: Sequence[pretty_midi.Note]) -> List[pretty_midi.Note]:
    """Make a note list strictly monophonic by truncating overlaps.

    When two notes overlap, we trim the earlier note's end to the later note's
    start. Notes whose duration would become non-positive are dropped. This
    preserves the *first* onset and is usually closer to the perceived melody
    than dropping the later note.
    """
    sorted_notes = sorted(notes, key=lambda n: n.start)
    out: List[pretty_midi.Note] = []
    for n in sorted_notes:
        if out and out[-1].end > n.start:
            out[-1].end = n.start
            if out[-1].end <= out[-1].start:
                out.pop()
        if n.end > n.start:
            out.append(n)
    return out


# -----------------------------------------------------------------------------
# Quantization
# -----------------------------------------------------------------------------

def quantize_to_grid(
    notes: Sequence[pretty_midi.Note],
    seconds_per_step: float,
    min_dur_steps: int = 1,
) -> List[Note]:
    """Snap ``pretty_midi.Note``s to a fixed sixteenth-note grid.

    ``seconds_per_step`` is the duration of one grid step. Returned notes are
    in our ``Note`` dataclass with integer step times.

    Pitches outside ``[PITCH_MIN, PITCH_MAX]`` are octave-shifted into range;
    if that still fails (very rare) the note is dropped.
    """
    out: List[Note] = []
    prev_end = -1
    for n in notes:
        start = int(round(n.start / seconds_per_step))
        end = int(round(n.end / seconds_per_step))
        if end - start < min_dur_steps:
            end = start + min_dur_steps
        # Prevent overlap after rounding.
        if start < prev_end:
            start = prev_end
            if end <= start:
                continue

        pitch = n.pitch
        while pitch < PITCH_MIN:
            pitch += 12
        while pitch > PITCH_MAX:
            pitch -= 12
        if not (PITCH_MIN <= pitch <= PITCH_MAX):
            continue

        out.append(Note(pitch=pitch, start=start, end=end))
        prev_end = end
    return out


def estimate_tempo(pm: pretty_midi.PrettyMIDI, default: float = 120.0) -> float:
    """Return a single representative tempo for the file.

    Uses ``estimate_tempo`` from pretty_midi as the primary source, falling
    back to the first tempo change or ``default``. We clamp to a musically
    plausible range to avoid pathological values from corrupted files.
    """
    try:
        bpm = float(pm.estimate_tempo())
    except Exception:
        bpm = 0.0

    if not (30.0 <= bpm <= 300.0):
        # Try the tempo-change list.
        _, tempi = pm.get_tempo_changes()
        if len(tempi) > 0:
            bpm = float(tempi[0])

    if not (30.0 <= bpm <= 300.0):
        bpm = default
    return bpm


def seconds_per_sixteenth(bpm: float) -> float:
    """Convert BPM (quarter-note) to seconds per sixteenth-note step."""
    quarter = 60.0 / bpm
    return quarter / 4.0


# -----------------------------------------------------------------------------
# High-level convenience
# -----------------------------------------------------------------------------

def midi_to_monophonic_notes(
    path: str | Path,
    *,
    max_polyphony: float = 0.05,
    min_notes: int = 16,
    return_bpm: bool = False,
) -> List[Note] | tuple[List[Note], float]:
    """End-to-end: MIDI file → quantized monophonic ``Note`` list.

    Set ``return_bpm=True`` to also receive the estimated tempo (used by the
    Call-and-Response pair builder to tokenize a TEMPO context).
    """
    pm = load_midi(path)
    track = pick_melody_track(pm, max_polyphony=max_polyphony, min_notes=min_notes)
    mono = force_monophonic(track.notes)
    bpm = estimate_tempo(pm)
    spq = seconds_per_sixteenth(bpm)
    notes = quantize_to_grid(mono, seconds_per_step=spq)
    if not notes:
        raise MidiParseError(f"{path}: nothing left after quantization.")
    if return_bpm:
        return notes, bpm
    return notes


# -----------------------------------------------------------------------------
# Bar slicing (used by the call-and-response pair builder)
# -----------------------------------------------------------------------------

def slice_bars(
    notes: Sequence[Note],
    start_bar: int,
    num_bars: int,
) -> List[Note]:
    """Return notes whose ``start`` falls within ``[start_bar, start_bar+num_bars)``,
    rebased so the slice starts at step 0.

    Notes that begin inside the window but extend past it are truncated at the
    window's end.
    """
    win_start = start_bar * STEPS_PER_BAR
    win_end = (start_bar + num_bars) * STEPS_PER_BAR
    out: List[Note] = []
    for n in notes:
        if n.start < win_start or n.start >= win_end:
            continue
        new_start = n.start - win_start
        new_end = min(n.end, win_end) - win_start
        if new_end > new_start:
            out.append(Note(n.pitch, new_start, new_end))
    return out


def num_bars(notes: Sequence[Note]) -> int:
    """Total bar count covered by the note sequence."""
    if not notes:
        return 0
    last_step = max(n.end for n in notes)
    return (last_step + STEPS_PER_BAR - 1) // STEPS_PER_BAR
