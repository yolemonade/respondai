"""Fast jazz backing for RespondAI.

10 bass patterns + 10 comping patterns, rendered with the existing fast
``synth_notes`` engine. Pattern selection is stable for a given key/round/
exchange/melody so replaying the same response does not unexpectedly change.
"""

from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Sequence

import numpy as np

from data.tokenizer import Note, STEPS_PER_BAR
from input.piano import synth_notes


_AUDIO_POOL = ThreadPoolExecutor(max_workers=4)

BASS_GAIN = 0.30
COMP_GAIN = 0.13

_ROOT_PC = {
    "C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5,
    "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11,
}

_CHORD_INTERVALS = {
    "maj7": (0, 4, 7, 11),
    "min7": (0, 3, 7, 10),
    "7": (0, 4, 7, 10),
    "m7b5": (0, 3, 6, 10),
}

_MAJOR_PROGRESSIONS = (
    ((0, "maj7"), (7, "7")),
    ((0, "maj7"), (9, "min7")),
    ((2, "min7"), (7, "7")),
    ((0, "maj7"), (5, "maj7")),
    ((4, "min7"), (9, "min7")),
    ((9, "min7"), (2, "min7")),
    ((5, "maj7"), (7, "7")),
    ((0, "maj7"), (10, "7")),
    ((0, "maj7"), (4, "7")),
    ((2, "min7"), (1, "7")),
)

_MINOR_PROGRESSIONS = (
    ((0, "min7"), (7, "7")),
    ((0, "min7"), (5, "min7")),
    ((2, "m7b5"), (7, "7")),
    ((0, "min7"), (8, "maj7")),
    ((0, "min7"), (10, "7")),
    ((5, "min7"), (7, "7")),
    ((8, "maj7"), (7, "7")),
    ((0, "min7"), (2, "m7b5")),
    ((3, "maj7"), (10, "7")),
    ((0, "min7"), (4, "7")),
)

# (start step in bar, duration, role)
_BASS_PATTERNS = (
    ("two_feel", ((0, 6, "root"), (8, 6, "fifth"))),
    ("walking", ((0, 3, "root"), (4, 3, "third"), (8, 3, "fifth"), (12, 3, "approach_below"))),
    ("syncopated", ((0, 4, "root"), (6, 3, "fifth"), (11, 2, "octave"), (14, 2, "approach_below"))),
    ("pedal", ((0, 3, "root"), (4, 3, "root"), (8, 3, "fifth"), (12, 3, "root"))),
    ("charleston", ((0, 3, "root"), (6, 2, "fifth"), (12, 3, "root"))),
    ("sparse", ((0, 7, "root"),)),
    ("anticipation", ((0, 5, "root"), (7, 3, "fifth"), (14, 2, "approach_below"))),
    ("broken_seventh", ((0, 3, "root"), (4, 3, "fifth"), (8, 3, "seventh"), (12, 3, "third"))),
    ("octave_bounce", ((0, 3, "root"), (4, 3, "octave"), (8, 3, "fifth"), (12, 3, "octave"))),
    ("late_push", ((0, 5, "root"), (9, 3, "fifth"), (13, 3, "approach_above"))),
)

# (start step in bar, duration, voicing)
_COMP_PATTERNS = (
    ("two_stabs", ((0, 5, "shell"), (8, 5, "shell"))),
    ("charleston", ((0, 3, "full"), (6, 2, "upper"), (12, 3, "shell"))),
    ("offbeat_four", ((2, 2, "upper"), (6, 2, "shell"), (10, 2, "upper"), (14, 2, "shell"))),
    ("quarter_shells", ((0, 2, "shell"), (4, 2, "shell"), (8, 2, "shell"), (12, 2, "shell"))),
    ("late_offbeats", ((3, 2, "upper"), (7, 2, "rootless"), (11, 2, "upper"), (15, 1, "shell"))),
    ("sustained_pad", ((0, 7, "pad"), (8, 7, "pad"))),
    ("front_and_back", ((0, 2, "full"), (10, 3, "shell"))),
    ("minimal_rootless", ((4, 2, "rootless"), (12, 2, "rootless"))),
    ("delayed_stabs", ((1, 3, "upper"), (9, 3, "shell"))),
    ("busy_comp", ((0, 3, "shell"), (5, 2, "upper"), (10, 2, "shell"), (13, 3, "upper"))),
)


def _parse_key_token(key_token: str) -> tuple[int, bool]:
    is_minor = bool(key_token and key_token.endswith("m"))
    root_name = key_token[:-1] if is_minor else key_token
    if root_name not in _ROOT_PC:
        return _ROOT_PC["C"], False
    return _ROOT_PC[root_name], is_minor


def _nearest_pitch_class(pitch_class: int, preferred_midi: int) -> int:
    center = preferred_midi - preferred_midi % 12 + pitch_class
    candidates = (center - 12, center, center + 12)
    return min(candidates, key=lambda pitch: abs(pitch - preferred_midi))


def _selection_seed(
    melody_notes: Sequence[Note],
    key_token: str,
    round_num: int,
    exchange_num: int,
) -> int:
    value = sum(
        (index + 1) * ord(character)
        for index, character in enumerate(key_token or "C")
    )
    value += int(round_num) * 10007
    value += int(exchange_num) * 1009
    for index, note in enumerate(melody_notes):
        value += (index + 1) * (
            int(note.pitch) * 31
            + int(note.start) * 17
            + int(note.end) * 13
        )
    return value


def _bass_role_pitch(
    root_midi: int,
    quality: str,
    role: str,
    next_root_midi: int,
) -> int:
    intervals = _CHORD_INTERVALS.get(quality, _CHORD_INTERVALS["min7"])
    if role == "root":
        pitch = root_midi
    elif role == "third":
        pitch = root_midi + intervals[1]
    elif role == "fifth":
        pitch = root_midi + intervals[2]
    elif role == "seventh":
        pitch = root_midi + intervals[3]
    elif role == "octave":
        pitch = root_midi + 12
    elif role == "approach_above":
        pitch = next_root_midi + 1
    else:
        pitch = next_root_midi - 1
    return max(24, min(59, pitch))


def _chord_voicing(
    root_midi: int,
    quality: str,
    voicing_name: str,
) -> list[int]:
    intervals = _CHORD_INTERVALS.get(quality, _CHORD_INTERVALS["min7"])
    root = root_midi
    third = root_midi + intervals[1]
    fifth = root_midi + intervals[2]
    seventh = root_midi + intervals[3]
    ninth = root_midi + 14

    if voicing_name == "shell":
        pitches = [third, seventh]
    elif voicing_name == "upper":
        pitches = [third, fifth, seventh]
    elif voicing_name == "rootless":
        pitches = [third, seventh, ninth]
    elif voicing_name == "pad":
        pitches = [root, third, fifth, seventh]
    else:
        pitches = [root, third, fifth, seventh]

    return [max(43, min(76, pitch)) for pitch in pitches]


def make_random_backing_layers(
    melody_notes: Sequence[Note],
    key_token: str,
    *,
    round_num: int = 1,
    exchange_num: int = 1,
) -> tuple[List[Note], List[Note], dict]:
    if not melody_notes or not key_token:
        return [], [], {}

    key_root_pc, is_minor = _parse_key_token(key_token)
    rng = random.Random(
        _selection_seed(
            melody_notes,
            key_token,
            round_num,
            exchange_num,
        )
    )

    progressions = _MINOR_PROGRESSIONS if is_minor else _MAJOR_PROGRESSIONS
    progression_index = rng.randrange(len(progressions))
    bass_pattern_index = rng.randrange(len(_BASS_PATTERNS))
    comp_pattern_index = rng.randrange(len(_COMP_PATTERNS))

    progression = progressions[progression_index]
    bass_pattern_name, bass_pattern = _BASS_PATTERNS[bass_pattern_index]
    comp_pattern_name, comp_pattern = _COMP_PATTERNS[comp_pattern_index]

    span = max(int(note.end) for note in melody_notes)
    number_of_bars = max(
        1,
        (span + STEPS_PER_BAR - 1) // STEPS_PER_BAR,
    )

    bass_notes: List[Note] = []
    comp_notes: List[Note] = []

    for bar_index in range(number_of_bars):
        chord_offset, chord_quality = progression[bar_index % len(progression)]
        next_offset, _ = progression[(bar_index + 1) % len(progression)]

        chord_root_pc = (key_root_pc + chord_offset) % 12
        next_root_pc = (key_root_pc + next_offset) % 12

        bass_root = _nearest_pitch_class(
            chord_root_pc,
            preferred_midi=36 + key_root_pc,
        )
        next_bass_root = _nearest_pitch_class(
            next_root_pc,
            preferred_midi=bass_root,
        )
        comp_root = _nearest_pitch_class(
            chord_root_pc,
            preferred_midi=48 + key_root_pc,
        )

        bar_start = bar_index * STEPS_PER_BAR

        for local_start, duration, role in bass_pattern:
            start = bar_start + local_start
            if start >= span:
                continue
            end = min(start + duration, span)
            if end <= start:
                continue
            bass_notes.append(
                Note(
                    _bass_role_pitch(
                        bass_root,
                        chord_quality,
                        role,
                        next_bass_root,
                    ),
                    start,
                    end,
                )
            )

        for local_start, duration, voicing_name in comp_pattern:
            start = bar_start + local_start
            if start >= span:
                continue
            end = min(start + duration, span)
            if end <= start:
                continue
            for pitch in _chord_voicing(
                comp_root,
                chord_quality,
                voicing_name,
            ):
                comp_notes.append(Note(pitch, start, end))

    metadata = {
        "progression_index": progression_index,
        "bass_pattern_index": bass_pattern_index,
        "bass_pattern_name": bass_pattern_name,
        "comp_pattern_index": comp_pattern_index,
        "comp_pattern_name": comp_pattern_name,
    }
    return bass_notes, comp_notes, metadata


def _mix_layer(
    destination: np.ndarray,
    source: Optional[np.ndarray],
    gain: float,
) -> None:
    if source is None or len(source) == 0:
        return
    length = min(len(destination), len(source))
    destination[:length] += source[:length] * float(gain)


def build_round_audio(
    user_notes: Sequence[Note],
    ai_notes: Sequence[Note],
    bpm: int,
    *,
    sample_rate: int = 22050,
    ai_response_max_sec: float = 4.0,
    swing_amount: float = 0.60,
    key_token: Optional[str] = None,
    round_num: int = 1,
    exchange_num: int = 1,
    bass_gain: float = BASS_GAIN,
    comp_gain: float = COMP_GAIN,
) -> tuple[int, np.ndarray]:
    """Render user melody, AI melody, bass, and comping."""
    gap = np.zeros(int(0.15 * sample_rate), dtype=np.float32)
    swing_amount = float(np.clip(swing_amount, 0.0, 1.0))

    bass_notes: List[Note] = []
    comp_notes: List[Note] = []
    backing_meta: dict = {}

    if ai_notes and key_token:
        bass_notes, comp_notes, backing_meta = make_random_backing_layers(
            ai_notes,
            key_token,
            round_num=round_num,
            exchange_num=exchange_num,
        )

    user_future = _AUDIO_POOL.submit(
        synth_notes,
        list(user_notes),
        bpm=bpm,
        sample_rate=sample_rate,
        role="user",
        swing_amount=min(swing_amount, 0.52),
        humanize_ms=2.0,
    )
    ai_future = _AUDIO_POOL.submit(
        synth_notes,
        list(ai_notes),
        bpm=bpm,
        sample_rate=sample_rate,
        role="ai",
        tail_sec=0.05,
        swing_amount=swing_amount,
        humanize_ms=8.0,
    )
    bass_future = (
        _AUDIO_POOL.submit(
            synth_notes,
            bass_notes,
            bpm=bpm,
            sample_rate=sample_rate,
            role="user",
            tail_sec=0.05,
            swing_amount=0.12,
            humanize_ms=2.0,
        )
        if bass_notes
        else None
    )
    comp_future = (
        _AUDIO_POOL.submit(
            synth_notes,
            comp_notes,
            bpm=bpm,
            sample_rate=sample_rate,
            role="ai",
            tail_sec=0.08,
            swing_amount=swing_amount,
            humanize_ms=5.0,
        )
        if comp_notes
        else None
    )

    user_wav = user_future.result()
    ai_wav = ai_future.result()
    bass_wav = bass_future.result() if bass_future else None
    comp_wav = comp_future.result() if comp_future else None

    ai_mix_length = max(
        len(ai_wav),
        len(bass_wav) if bass_wav is not None else 0,
        len(comp_wav) if comp_wav is not None else 0,
        1,
    )
    ai_mix = np.zeros(ai_mix_length, dtype=np.float32)

    _mix_layer(ai_mix, ai_wav, 1.0)
    _mix_layer(ai_mix, bass_wav, bass_gain)
    _mix_layer(ai_mix, comp_wav, comp_gain)

    ai_max_samples = int(float(ai_response_max_sec) * int(sample_rate))
    ai_mix = ai_mix[:ai_max_samples]

    combined = np.concatenate([user_wav, gap, ai_mix])
    peak = float(np.max(np.abs(combined))) if len(combined) else 0.0

    if peak < 1e-8:
        combined = gap
        peak = float(np.max(np.abs(gap))) or 1.0

    combined = (combined / peak * 0.92).astype(np.float32)

    if backing_meta:
        print(
            "[BACKING] "
            f"progression={backing_meta['progression_index']} "
            f"bass={backing_meta['bass_pattern_name']} "
            f"comp={backing_meta['comp_pattern_name']}"
        )

    pcm = (np.clip(combined, -1.0, 1.0) * 32767).astype(np.int16)
    return int(sample_rate), pcm
