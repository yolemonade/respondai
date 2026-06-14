#!/usr/bin/env python3
"""
Compare RespondAI's fast NumPy synth + bass against FluidSynth + bass.

Outputs the same fixed jazz phrase through both renderers:
  compare_outputs/01_fast_synth_bass.wav
  compare_outputs/02_fluidsynth_bass.wav
  compare_outputs/render_report.txt

Usage:
  python scripts/compare_audio_versions.py \
      --soundfont soundfonts/your_soundfont.sf2

Optional:
  --piano-program 0
  --bass-program 32
  --bpm 92
  --key Dm
  --bass-gain 0.32
  --output-dir compare_outputs
"""

from __future__ import annotations

import argparse
import time
import wave
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from data.tokenizer import Note, STEPS_PER_BAR
from input.piano import synth_notes


SAMPLE_RATE = 22050

_ROOT_PC = {
    "C": 0,
    "C#": 1,
    "D": 2,
    "D#": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "G": 7,
    "G#": 8,
    "A": 9,
    "A#": 10,
    "B": 11,
}


def fixed_jazz_phrase() -> list[Note]:
    """Two-bar D-minor phrase with rests, syncopation, and mixed durations."""
    return [
        Note(62, 0, 3),    # D4
        Note(65, 3, 5),    # F4
        Note(69, 6, 8),    # A4
        Note(72, 8, 11),   # C5
        Note(69, 11, 12),  # A4
        Note(67, 14, 16),  # G4
        Note(65, 16, 18),  # F4
        Note(69, 18, 21),  # A4
        Note(73, 22, 24),  # C#5
        Note(74, 24, 28),  # D5
        Note(72, 29, 31),  # C5
        Note(69, 31, 32),  # A4
    ]


def make_bass_notes(
    melody_notes: Sequence[Note],
    key_token: str,
    *,
    octave: int = 2,
) -> list[Note]:
    """Beat 1 root, beat 3 fifth; slightly detached for a jazzier feel."""
    if not melody_notes or not key_token:
        return []

    root_name = key_token[:-1] if key_token.endswith("m") else key_token
    if root_name not in _ROOT_PC:
        raise ValueError(
            f"Unsupported key {key_token!r}. "
            f"Use one of: {', '.join(_ROOT_PC)} with optional 'm'."
        )

    root_pc = _ROOT_PC[root_name]
    root_midi = 12 * (octave + 1) + root_pc
    fifth_midi = root_midi + 7

    melody_low = min(int(note.pitch) for note in melody_notes)
    while fifth_midi > melody_low - 5 and root_midi - 12 >= 24:
        root_midi -= 12
        fifth_midi -= 12

    span = max(int(note.end) for note in melody_notes)
    number_of_bars = max(
        1,
        (span + STEPS_PER_BAR - 1) // STEPS_PER_BAR,
    )
    half_bar = STEPS_PER_BAR // 2

    bass: list[Note] = []
    for bar_index in range(number_of_bars):
        bar_start = bar_index * STEPS_PER_BAR

        if bar_start < span:
            bass.append(
                Note(
                    root_midi,
                    bar_start,
                    min(bar_start + 6, span),
                )
            )

        fifth_start = bar_start + half_bar
        if fifth_start < span:
            bass.append(
                Note(
                    fifth_midi,
                    fifth_start,
                    min(fifth_start + 6, span),
                )
            )

    return bass


def normalize_mono(audio: np.ndarray, peak_target: float = 0.90) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return np.zeros(1, dtype=np.float32)

    peak = float(np.max(np.abs(audio)))
    if peak > 1e-8:
        audio = audio / peak * peak_target

    return np.clip(audio, -1.0, 1.0).astype(np.float32, copy=False)


def write_mono_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (normalize_mono(audio) * 32767.0).astype(np.int16)

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


def render_fast_version(
    melody_notes: Sequence[Note],
    bass_notes: Sequence[Note],
    *,
    bpm: int,
    bass_gain: float,
    swing_amount: float,
) -> np.ndarray:
    melody = synth_notes(
        list(melody_notes),
        bpm=bpm,
        sample_rate=SAMPLE_RATE,
        role="ai",
        tail_sec=0.20,
        swing_amount=swing_amount,
        humanize_ms=8.0,
    )

    bass = synth_notes(
        list(bass_notes),
        bpm=bpm,
        sample_rate=SAMPLE_RATE,
        role="user",
        tail_sec=0.20,
        swing_amount=0.0,
        humanize_ms=1.0,
    )

    length = max(len(melody), len(bass))
    mixed = np.zeros(length, dtype=np.float32)
    mixed[: len(melody)] += melody
    mixed[: len(bass)] += bass * float(bass_gain)

    return normalize_mono(mixed)


def _stable_jitter_ms(note: Note, index: int, maximum_ms: float) -> float:
    """Deterministic pseudo-humanization so repeated tests are identical."""
    if maximum_ms <= 0:
        return 0.0

    value = (
        int(note.pitch) * 31
        + int(note.start) * 17
        + int(note.end) * 13
        + (index + 1) * 19
    )
    unit = ((value % 1001) / 1000.0) * 2.0 - 1.0
    return unit * maximum_ms


def _swing_delay_seconds(
    step: int,
    seconds_per_step: float,
    swing_amount: float,
) -> float:
    if int(step) % 4 != 2:
        return 0.0
    return (2.0 / 3.0) * float(swing_amount) * seconds_per_step


def _velocity(note: Note, index: int, count: int, base: int) -> int:
    velocity = int(base)
    position = int(note.start) % 16

    if position == 0:
        velocity += 10
    elif position == 8:
        velocity += 5
    elif position in (2, 6, 10, 14):
        velocity -= 3

    if count > 1:
        phrase_position = index / (count - 1)
        arc = 1.0 - abs(phrase_position * 2.0 - 1.0)
        velocity += round(5 * arc)

    return max(1, min(127, velocity))


def _fluid_events(
    notes: Sequence[Note],
    *,
    channel: int,
    base_velocity: int,
    bpm: int,
    sample_rate: int,
    swing_amount: float,
    humanize_ms: float,
) -> list[tuple[int, int, int, int, int]]:
    """Return (sample, order, channel, pitch, velocity); off precedes on."""
    seconds_per_step = (60.0 / float(bpm)) / 4.0
    events: list[tuple[int, int, int, int, int]] = []

    ordered = sorted(
        notes,
        key=lambda note: (int(note.start), int(note.end), int(note.pitch)),
    )

    for index, note in enumerate(ordered):
        jitter = _stable_jitter_ms(note, index, humanize_ms) / 1000.0
        start_seconds = (
            int(note.start) * seconds_per_step
            + _swing_delay_seconds(
                int(note.start),
                seconds_per_step,
                swing_amount,
            )
            + jitter
        )
        start_seconds = max(0.0, start_seconds)

        symbolic_duration = max(
            seconds_per_step * 0.25,
            (int(note.end) - int(note.start)) * seconds_per_step,
        )
        gate_ratio = 0.88 if channel == 0 else 0.84
        end_seconds = start_seconds + symbolic_duration * gate_ratio

        start_sample = max(0, round(start_seconds * sample_rate))
        end_sample = max(start_sample + 1, round(end_seconds * sample_rate))

        events.append(
            (
                start_sample,
                1,
                channel,
                int(note.pitch),
                _velocity(
                    note,
                    index,
                    len(ordered),
                    base_velocity,
                ),
            )
        )
        events.append(
            (
                end_sample,
                0,
                channel,
                int(note.pitch),
                0,
            )
        )

    return events


def render_fluidsynth_version(
    melody_notes: Sequence[Note],
    bass_notes: Sequence[Note],
    *,
    bpm: int,
    soundfont: Path,
    piano_program: int,
    bass_program: int,
    swing_amount: float,
) -> np.ndarray:
    try:
        import fluidsynth
    except ImportError as exc:
        raise RuntimeError(
            "Python package 'pyfluidsynth' is not available. "
            "Install it with: pip install pyfluidsynth"
        ) from exc

    if not soundfont.exists():
        raise FileNotFoundError(f"SoundFont not found: {soundfont}")

    synth = fluidsynth.Synth(
        samplerate=SAMPLE_RATE,
        gain=0.55,
    )

    try:
        soundfont_id = synth.sfload(str(soundfont))
        if soundfont_id < 0:
            raise RuntimeError(f"FluidSynth could not load: {soundfont}")

        if synth.program_select(0, soundfont_id, 0, int(piano_program)) != 0:
            raise RuntimeError(
                f"Could not select piano program {piano_program}."
            )

        if synth.program_select(1, soundfont_id, 0, int(bass_program)) != 0:
            raise RuntimeError(
                f"Could not select bass program {bass_program}. "
                "Your SoundFont may not be GM-compatible."
            )

        # Warm-up prevents the first actual block from paying initialization cost.
        synth.get_samples(256)

        events = []
        events.extend(
            _fluid_events(
                melody_notes,
                channel=0,
                base_velocity=92,
                bpm=bpm,
                sample_rate=SAMPLE_RATE,
                swing_amount=swing_amount,
                humanize_ms=8.0,
            )
        )
        events.extend(
            _fluid_events(
                bass_notes,
                channel=1,
                base_velocity=58,
                bpm=bpm,
                sample_rate=SAMPLE_RATE,
                swing_amount=0.0,
                humanize_ms=1.0,
            )
        )
        events.sort(key=lambda event: (event[0], event[1]))

        if not events:
            return np.zeros(int(0.4 * SAMPLE_RATE), dtype=np.float32)

        chunks: list[np.ndarray] = []
        cursor = 0
        position = 0

        while position < len(events):
            sample_position = events[position][0]
            frame_count = sample_position - cursor

            if frame_count > 0:
                chunks.append(
                    np.asarray(
                        synth.get_samples(frame_count)
                    )
                )

            while (
                position < len(events)
                and events[position][0] == sample_position
            ):
                _, order, channel, pitch, velocity = events[position]
                if order == 0:
                    synth.noteoff(channel, pitch)
                else:
                    synth.noteon(channel, pitch, velocity)
                position += 1

            cursor = sample_position

        # SoundFont release/reverb tail.
        chunks.append(
            np.asarray(
                synth.get_samples(
                    int(0.45 * SAMPLE_RATE)
                )
            )
        )

        raw = np.concatenate(chunks)

    finally:
        try:
            synth.delete()
        except Exception:
            pass

    # pyFluidSynth returns interleaved stereo samples.
    if raw.size % 2:
        raw = raw[:-1]

    stereo = raw.reshape(-1, 2)

    if np.issubdtype(stereo.dtype, np.integer):
        scale = float(np.iinfo(stereo.dtype).max)
        stereo = stereo.astype(np.float32) / scale
    else:
        stereo = stereo.astype(np.float32)

    mono = stereo.mean(axis=1)
    return normalize_mono(mono)


def note_report(
    melody_notes: Sequence[Note],
    bass_notes: Sequence[Note],
) -> str:
    lines = ["MELODY"]
    for note in melody_notes:
        lines.append(
            f"pitch={note.pitch:3d} start={note.start:2d} "
            f"end={note.end:2d} duration={note.end-note.start:2d}"
        )

    lines.append("")
    lines.append("BASS")
    for note in bass_notes:
        lines.append(
            f"pitch={note.pitch:3d} start={note.start:2d} "
            f"end={note.end:2d} duration={note.end-note.start:2d}"
        )

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--soundfont",
        required=True,
        help="Path to a GM-compatible .sf2 file.",
    )
    parser.add_argument("--key", default="Dm")
    parser.add_argument("--bpm", type=int, default=92)
    parser.add_argument("--piano-program", type=int, default=0)
    parser.add_argument("--bass-program", type=int, default=32)
    parser.add_argument("--bass-gain", type=float, default=0.32)
    parser.add_argument("--swing", type=float, default=0.65)
    parser.add_argument(
        "--output-dir",
        default="compare_outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    melody_notes = fixed_jazz_phrase()
    bass_notes = make_bass_notes(
        melody_notes,
        args.key,
    )

    fast_start = time.perf_counter()
    fast_audio = render_fast_version(
        melody_notes,
        bass_notes,
        bpm=args.bpm,
        bass_gain=args.bass_gain,
        swing_amount=args.swing,
    )
    fast_ms = (time.perf_counter() - fast_start) * 1000.0

    fluid_start = time.perf_counter()
    fluid_audio = render_fluidsynth_version(
        melody_notes,
        bass_notes,
        bpm=args.bpm,
        soundfont=Path(args.soundfont),
        piano_program=args.piano_program,
        bass_program=args.bass_program,
        swing_amount=args.swing,
    )
    fluid_ms = (time.perf_counter() - fluid_start) * 1000.0

    fast_path = output_dir / "01_fast_synth_bass.wav"
    fluid_path = output_dir / "02_fluidsynth_bass.wav"
    report_path = output_dir / "render_report.txt"

    write_mono_wav(
        fast_path,
        fast_audio,
        SAMPLE_RATE,
    )
    write_mono_wav(
        fluid_path,
        fluid_audio,
        SAMPLE_RATE,
    )

    report = "\n".join(
        [
            f"bpm={args.bpm}",
            f"key={args.key}",
            f"soundfont={Path(args.soundfont).resolve()}",
            f"piano_program={args.piano_program}",
            f"bass_program={args.bass_program}",
            f"fast_render_ms={fast_ms:.2f}",
            f"fluidsynth_render_ms={fluid_ms:.2f}",
            "",
            note_report(melody_notes, bass_notes),
        ]
    )
    report_path.write_text(report, encoding="utf-8")

    print(f"[OK] Fast synth: {fast_path} ({fast_ms:.1f} ms)")
    print(f"[OK] FluidSynth: {fluid_path} ({fluid_ms:.1f} ms)")
    print(f"[OK] Report: {report_path}")


if __name__ == "__main__":
    main()
