"""Piano input helpers and lightweight audio synthesis.

This module keeps the existing mock response helpers and provides a fast
NumPy-based fallback synthesizer with:
- note-length-aware articulation
- beat accents
- deterministic timing humanization
- optional jazz swing on offbeat eighth notes
- a small room/echo tail

It does not require FluidSynth or an external SoundFont.
"""
from __future__ import annotations

from typing import List

import numpy as np

from data.tokenizer import Note


def mock_ai_response(notes: List[Note]) -> List[Note]:
    """Shift user notes up by one octave, capped at the MIDI range."""
    if not notes:
        return []
    return [
        Note(min(int(note.pitch) + 12, 127), int(note.start), int(note.end))
        for note in notes
    ]


# UI/game-flow fallback melody.
TEMP_AI_TEST_MELODY: List[Note] = [
    Note(60, 0, 4),
    Note(64, 4, 8),
    Note(67, 8, 12),
    Note(72, 12, 16),
    Note(67, 16, 20),
    Note(64, 20, 24),
]


def temp_ai_response_for_testing(user_notes: List[Note]) -> List[Note]:
    """Return a fixed melody for an empty call, otherwise octave-shift input."""
    if not user_notes:
        return list(TEMP_AI_TEST_MELODY)
    return mock_ai_response(user_notes)


def _midi_to_hz(pitch: int) -> float:
    return 440.0 * (2.0 ** ((float(pitch) - 69.0) / 12.0))


def _stable_seed(notes: List[Note], role: str) -> int:
    """Build a deterministic seed so the same phrase keeps the same feel."""
    role_offset = {
        "user": 5,
        "ai": 17,
        "bass": 29,
        "comp": 41,
    }.get(role, 5)
    value = role_offset
    for index, note in enumerate(notes):
        value += (
            (index + 1)
            * (
                int(note.pitch) * 31
                + int(note.start) * 17
                + int(note.end) * 13
            )
        )
    return int(value % (2**32 - 1))


def _swing_delay_seconds(
    step: int,
    seconds_per_step: float,
    swing_amount: float,
) -> float:
    """Delay the offbeat eighth note.

    The symbolic grid has four sixteenth-note steps per quarter note.
    The second eighth-note position is therefore step % 4 == 2.

    swing_amount:
      0.0 -> straight
      1.0 -> approximately triplet swing
    """
    if int(step) % 4 != 2:
        return 0.0

    max_delay_steps = 2.0 / 3.0
    return max_delay_steps * swing_amount * seconds_per_step


def _velocity_gain(
    note: Note,
    note_index: int,
    note_count: int,
    role: str,
) -> float:
    """멜로디에는 표현을 주고, 반주는 일정한 볼륨을 유지한다."""
    beat_position = int(note.start) % 16

    if role == "bass":
        gain = 0.70
        if beat_position == 0:
            gain += 0.045
        elif beat_position == 8:
            gain += 0.020
        return float(np.clip(gain, 0.58, 0.78))

    if role == "comp":
        gain = 0.43
        if beat_position == 0:
            gain += 0.025
        elif beat_position in (2, 6, 10, 14):
            gain -= 0.018
        return float(np.clip(gain, 0.34, 0.50))

    gain = 0.66 if role == "ai" else 0.60

    if beat_position == 0:
        gain += 0.16
    elif beat_position == 8:
        gain += 0.09
    elif beat_position in (4, 12):
        gain += 0.025
    elif beat_position in (2, 6, 10, 14):
        gain -= 0.035

    if note_count > 1:
        phrase_position = note_index / (note_count - 1)
        arc = 1.0 - abs(phrase_position * 2.0 - 1.0)
        gain += 0.055 * arc

    if role == "ai":
        gain += 0.045

    return float(np.clip(gain, 0.34, 0.96))


def _oscillator_mix(
    frequency: float,
    time_axis: np.ndarray,
    role: str,
) -> np.ndarray:
    """Fast piano/electric-piano-like additive oscillator bank."""
    phase = 2.0 * np.pi * frequency * time_axis

    if role == "bass":
        # 기본음 중심의 두껍고 건조한 베이스.
        signal = (
            0.78 * np.sin(phase)
            + 0.16 * np.sin(2.0 * phase + 0.02)
            + 0.045 * np.sin(3.0 * phase)
            + 0.018 * np.sin(4.0 * phase)
        )

    elif role == "comp":
        # 부드러운 전자피아노 계열의 코드 컴핑.
        signal = (
            0.43 * np.sin(phase)
            + 0.23 * np.sin(2.0 * phase + 0.11)
            + 0.105 * np.sin(3.0 * phase + 0.06)
            + 0.045 * np.sin(4.0 * phase)
        )
        signal += 0.065 * np.sin(
            2.0 * np.pi * frequency * 1.006 * time_axis
        )

    elif role == "ai":
        # 밝은 AI 멜로디에 detune과 짧은 hammer 성분을 추가한다.
        signal = (
            0.44 * np.sin(phase)
            + 0.215 * np.sin(2.0 * phase + 0.04)
            + 0.105 * np.sin(3.0 * phase + 0.09)
            + 0.048 * np.sin(4.0 * phase)
            + 0.022 * np.sin(5.0 * phase)
        )
        signal += 0.045 * np.sin(
            2.0 * np.pi * frequency * 0.997 * time_axis
        )
        signal += 0.045 * np.sin(
            2.0 * np.pi * frequency * 1.003 * time_axis
        )
        signal += (
            0.030
            * np.sin(2.0 * np.pi * frequency * 7.25 * time_axis)
            * np.exp(-32.0 * time_axis)
        )

    else:
        # 사용자 미리듣기 음색.
        signal = (
            0.56 * np.sin(phase)
            + 0.22 * np.sin(2.0 * phase + 0.03)
            + 0.09 * np.sin(3.0 * phase)
            + 0.035 * np.sin(4.0 * phase)
        )

    return signal.astype(np.float32, copy=False)


def _piano_envelope(
    time_axis: np.ndarray,
    duration: float,
    sample_rate: int,
    role: str,
) -> np.ndarray:
    """Sharp attack, two-stage decay, and short release."""
    sample_count = len(time_axis)
    if sample_count == 0:
        return np.zeros(0, dtype=np.float32)

    attack_seconds = min(0.012, max(0.003, duration * 0.08))
    attack_samples = min(
        sample_count,
        max(1, int(attack_seconds * sample_rate)),
    )

    decay_rate = {
        "ai": 3.15,
        "user": 3.9,
        "bass": 1.85,
        "comp": 2.65,
    }.get(role, 3.9)
    envelope = (
        0.72 * np.exp(-decay_rate * time_axis / max(duration, 0.04))
        + 0.28 * np.exp(-0.75 * time_axis / max(duration, 0.04))
    )

    envelope[:attack_samples] *= np.linspace(
        0.0,
        1.0,
        attack_samples,
        dtype=np.float32,
    )

    release_seconds = min(0.055, max(0.015, duration * 0.18))
    release_samples = min(
        sample_count,
        max(1, int(release_seconds * sample_rate)),
    )
    envelope[-release_samples:] *= np.linspace(
        1.0,
        0.0,
        release_samples,
        dtype=np.float32,
    )

    return envelope.astype(np.float32, copy=False)


def _apply_room(
    audio: np.ndarray,
    sample_rate: int,
    role: str,
) -> np.ndarray:
    """Apply a few delay taps as a cheap room effect."""
    if audio.size == 0:
        return audio

    wet = {
        "ai": 0.15,
        "user": 0.075,
        "bass": 0.03,
        "comp": 0.19,
    }.get(role, 0.075)
    taps = (
        (0.043, 0.48),
        (0.071, 0.28),
        (0.113, 0.16),
    )

    output = audio.astype(np.float32, copy=True)

    for delay_seconds, tap_gain in taps:
        delay = int(delay_seconds * sample_rate)
        if delay <= 0 or delay >= len(audio):
            continue
        output[delay:] += audio[:-delay] * (wet * tap_gain)

    return output


def synth_notes(
    notes: List[Note],
    bpm: float = 90,
    sample_rate: int = 22050,
    *,
    role: str = "user",
    tail_sec: float = 0.3,
    swing_amount: float = 0.0,
    humanize_ms: float = 0.0,
) -> np.ndarray:
    """Synthesize notes to float32 mono audio.

    Parameters
    ----------
    notes:
        Notes on the project's sixteenth-note grid.
    bpm:
        Quarter-note tempo.
    sample_rate:
        Output sample rate.
    role:
        ``"user"``, ``"ai"``, ``"bass"``, or ``"comp"``.
    tail_sec:
        Extra room for release/reverb.
    swing_amount:
        0.0 is straight; 1.0 approaches triplet swing. A practical jazz range
        is 0.5 to 0.7.
    humanize_ms:
        Deterministic onset jitter in milliseconds. Keep this below about
        10 ms for a tight game feel.
    """
    if not notes:
        return np.zeros(int(0.4 * sample_rate), dtype=np.float32)

    if bpm <= 0:
        raise ValueError("bpm must be greater than zero")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be greater than zero")

    ordered = sorted(
        notes,
        key=lambda note: (int(note.start), int(note.end), int(note.pitch)),
    )

    swing_amount = float(np.clip(swing_amount, 0.0, 1.0))
    humanize_ms = float(np.clip(humanize_ms, 0.0, 20.0))
    seconds_per_step = (60.0 / float(bpm)) / 4.0

    rng = np.random.default_rng(_stable_seed(ordered, role))

    maximum_swing_delay = (2.0 / 3.0) * seconds_per_step * swing_amount
    maximum_jitter = humanize_ms / 1000.0

    total_duration = (
        max(int(note.end) for note in ordered) * seconds_per_step
        + maximum_swing_delay
        + maximum_jitter
        + max(0.0, float(tail_sec))
        + 0.14
    )

    audio = np.zeros(
        max(1, int(np.ceil(total_duration * sample_rate))),
        dtype=np.float32,
    )

    for note_index, note in enumerate(ordered):
        pitch = int(note.pitch)
        start_step = int(note.start)
        end_step = int(note.end)

        if end_step <= start_step:
            continue

        frequency = _midi_to_hz(pitch)

        jitter = 0.0
        if humanize_ms > 0:
            jitter = rng.uniform(-humanize_ms, humanize_ms) / 1000.0

        start_seconds = (
            start_step * seconds_per_step
            + _swing_delay_seconds(
                start_step,
                seconds_per_step,
                swing_amount,
            )
            + jitter
        )
        start_seconds = max(0.0, start_seconds)

        symbolic_duration = (end_step - start_step) * seconds_per_step

        # 역할별 articulation.
        if role == "bass":
            gate_ratio = (
                0.88
                if symbolic_duration <= 2.0 * seconds_per_step
                else 0.94
            )
        elif role == "comp":
            gate_ratio = (
                0.72
                if symbolic_duration <= seconds_per_step
                else 0.84
            )
        elif symbolic_duration <= seconds_per_step:
            gate_ratio = 0.80 if role == "ai" else 0.84
        elif symbolic_duration <= 2.0 * seconds_per_step:
            gate_ratio = 0.88 if role == "ai" else 0.90
        else:
            gate_ratio = 0.94 if role == "ai" else 0.95

        duration = max(0.025, symbolic_duration * gate_ratio)
        sample_count = max(1, int(duration * sample_rate))

        time_axis = (
            np.arange(sample_count, dtype=np.float32)
            / float(sample_rate)
        )

        signal = _oscillator_mix(
            frequency,
            time_axis,
            role,
        )

        envelope = _piano_envelope(
            time_axis,
            duration,
            sample_rate,
            role,
        )

        gain = _velocity_gain(
            note,
            note_index,
            len(ordered),
            role,
        )

        rendered = signal * envelope * gain

        start_index = int(round(start_seconds * sample_rate))
        if start_index >= len(audio):
            continue

        end_index = min(
            len(audio),
            start_index + len(rendered),
        )

        rendered = rendered[: end_index - start_index]
        audio[start_index:end_index] += rendered

    audio = _apply_room(
        audio,
        sample_rate,
        role,
    )

    # Gentle soft clipping followed by normalization.
    audio = np.tanh(audio * 1.15).astype(np.float32, copy=False)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0

    if peak > 1e-6:
        target_peak = {
            "user": 0.86,
            "ai": 0.88,
            "bass": 0.68,
            "comp": 0.50,
        }.get(role, 0.86)
        audio = audio / peak * target_peak

    return audio.astype(np.float32, copy=False)
