"""
Decode helpers for team B.

The model speaks tokens; the UI speaks notes and audio. This module bridges
the two:

  * :func:`decode_tokens_to_notes` — wraps Tokenizer.decode_notes, returned
    as a plain list of ``Note`` dataclasses (already importable from
    ``data.tokenizer``).
  * :func:`notes_to_pretty_midi` — convert ``Note`` list to a
    ``pretty_midi.PrettyMIDI`` so it can be written to disk *or* fed to
    FluidSynth for audio rendering.
  * :func:`notes_to_wav` — one-shot, returns float32 mono audio. Uses
    ``pretty_midi.PrettyMIDI.fluidsynth`` so all sound-font handling lives
    inside pretty_midi and we don't need a separate FluidSynth Python binding.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pretty_midi
import time


from data.tokenizer import STEPS_PER_BAR, Note, Tokenizer


def decode_tokens_to_notes(tokens: Sequence[int], tokenizer: Tokenizer) -> List[Note]:
    """Thin wrapper over ``Tokenizer.decode_notes`` so importers have a single
    obvious entrypoint."""
    return tokenizer.decode_notes(tokens)


def notes_to_pretty_midi(
    notes: Sequence[Note],
    *,
    tempo: float = 120.0,
    program: int = 0,  # Acoustic Grand Piano
) -> pretty_midi.PrettyMIDI:
    """Build a single-instrument :class:`pretty_midi.PrettyMIDI` from notes.

    Step timing assumes a sixteenth-note grid (so 4 steps = 1 quarter note).
    """
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    inst = pretty_midi.Instrument(program=program)

    seconds_per_step = (60.0 / tempo) / 4.0  # quarter / 4 = sixteenth
    for n in notes:
        inst.notes.append(
            pretty_midi.Note(
                velocity=90,
                pitch=int(n.pitch),
                start=float(n.start * seconds_per_step),
                end=float(n.end * seconds_per_step),
            )
        )
    pm.instruments.append(inst)
    return pm


def notes_to_wav(
    notes: Sequence[Note],
    *,
    tempo: float = 120.0,
    sample_rate: int = 22050,
    sound_font: Optional[str] = None,
) -> np.ndarray:
    """Render notes to a mono float32 waveform via FluidSynth.

    ``sound_font`` is the path to a ``.sf2`` file. If ``None``, pretty_midi
    falls back to its bundled default if available; on a fresh machine, install
    one (e.g. ``brew install fluid-synth`` ships GeneralUser GS) and pass the
    path explicitly.
    """
    pm = notes_to_pretty_midi(notes, tempo=tempo)
    t_audio = time.time()

    audio = pm.fluidsynth(fs=sample_rate, sf2_path=sound_font) if sound_font else pm.fluidsynth(fs=sample_rate)
    print("fluidsynth:", time.time() - t_audio)
    # pretty_midi returns float64; downcast for size.
    return audio.astype(np.float32)


def save_notes_as_midi(
    notes: Sequence[Note],
    path: str | Path,
    *,
    tempo: float = 120.0,
) -> None:
    """Convenience: write a ``.mid`` file in one call."""
    notes_to_pretty_midi(notes, tempo=tempo).write(str(path))
