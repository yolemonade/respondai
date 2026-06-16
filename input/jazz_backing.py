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

try:
    from input.fluid_band import get_shared_renderer
except Exception:
    get_shared_renderer = None

_FLUID_RENDERER = None


_AUDIO_POOL = ThreadPoolExecutor(max_workers=4)

BASS_GAIN = 0.42
COMP_GAIN = 0.18

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
    """마디별 멜로디 강박에 맞는 안전한 재즈 코드를 선택한다."""
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

    span = max(int(note.end) for note in melody_notes)
    number_of_bars = max(
        1,
        (span + STEPS_PER_BAR - 1) // STEPS_PER_BAR,
    )

    if is_minor:
        chord_candidates = (
            (0, "min7"),
            (5, "min7"),
            (7, "7"),
            (8, "maj7"),
            (10, "7"),
        )
        scale_intervals = (0, 2, 3, 5, 7, 8, 10)
    else:
        chord_candidates = (
            (0, "maj7"),
            (2, "min7"),
            (5, "maj7"),
            (7, "7"),
            (9, "min7"),
        )
        scale_intervals = (0, 2, 4, 5, 7, 9, 11)

    scale_pcs = {
        (key_root_pc + interval) % 12
        for interval in scale_intervals
    }

    chord_plan: list[tuple[int, str]] = []
    previous_root: Optional[int] = None

    for bar_index in range(number_of_bars):
        bar_start = bar_index * STEPS_PER_BAR
        bar_end = bar_start + STEPS_PER_BAR
        bar_notes = [
            note
            for note in melody_notes
            if bar_start <= int(note.start) < bar_end
        ]

        def chord_score(candidate: tuple[int, str]) -> float:
            offset, quality = candidate
            root_pc = (key_root_pc + offset) % 12
            chord_pcs = {
                (root_pc + interval) % 12
                for interval in _CHORD_INTERVALS[quality]
            }

            score = 0.0
            for note in bar_notes:
                position = int(note.start) % STEPS_PER_BAR
                weight = (
                    2.7
                    if position in (0, 8)
                    else 1.5
                    if position in (4, 12)
                    else 0.8
                )
                pitch_class = int(note.pitch) % 12

                if pitch_class in chord_pcs:
                    score += 2.3 * weight
                elif pitch_class in scale_pcs:
                    score += 0.18 * weight
                else:
                    score -= 1.2 * weight

            # 빈 마디나 비슷한 점수에서는 tonic을 우선한다.
            if offset == 0:
                score += 0.75

            if previous_root is not None:
                movement = (root_pc - previous_root) % 12
                if movement == 0:
                    score += 0.35
                elif movement in (5, 7):
                    score += 0.50
                elif movement == 6:
                    score -= 0.45

            return score

        chosen = max(
            chord_candidates,
            key=chord_score,
        )
        chord_plan.append(chosen)
        previous_root = (key_root_pc + chosen[0]) % 12

    # 복잡한 walking/busy 패턴 대신 반주 역할에 집중한다.
    safe_bass_indices = (0, 4, 5)
    safe_comp_indices = (0, 6, 7, 8)
    bass_pattern_index = safe_bass_indices[
        rng.randrange(len(safe_bass_indices))
    ]
    comp_pattern_index = safe_comp_indices[
        rng.randrange(len(safe_comp_indices))
    ]

    bass_pattern_name, bass_pattern = _BASS_PATTERNS[
        bass_pattern_index
    ]
    comp_pattern_name, comp_pattern = _COMP_PATTERNS[
        comp_pattern_index
    ]

    bass_notes: List[Note] = []
    comp_notes: List[Note] = []

    for bar_index, (chord_offset, chord_quality) in enumerate(chord_plan):
        next_offset, _ = chord_plan[
            min(bar_index + 1, len(chord_plan) - 1)
        ]

        chord_root_pc = (
            key_root_pc + chord_offset
        ) % 12
        next_root_pc = (
            key_root_pc + next_offset
        ) % 12

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
            preferred_midi=47 + key_root_pc,
        )

        bar_start = bar_index * STEPS_PER_BAR

        for local_start, duration, role in bass_pattern:
            start = bar_start + local_start
            if start >= span:
                continue

            end = min(
                start + duration,
                span,
            )
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

            end = min(
                start + duration,
                span,
            )
            if end <= start:
                continue

            # full/pad 대신 shell/rootless 위주로 멜로디와 충돌을 줄인다.
            safe_voicing = (
                "rootless"
                if voicing_name in ("full", "upper", "pad")
                else voicing_name
            )

            for pitch in _chord_voicing(
                comp_root,
                chord_quality,
                safe_voicing,
            ):
                # 컴핑을 리드보다 아래에 제한한다.
                comp_notes.append(
                    Note(
                        min(pitch, 67),
                        start,
                        end,
                    )
                )

    metadata = {
        "progression_index": -1,
        "chord_plan": [
            f"{offset}:{quality}"
            for offset, quality in chord_plan
        ],
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


def _soft_compress(
    audio: np.ndarray,
    threshold: float = 0.42,
    ratio: float = 3.5,
) -> np.ndarray:
    """여러 화음이 겹칠 때 생기는 순간적인 음량 급증을 완화한다."""
    if audio.size == 0:
        return audio

    magnitude = np.abs(audio)
    compressed = np.where(
        magnitude <= threshold,
        magnitude,
        threshold + (magnitude - threshold) / max(1.0, ratio),
    )
    return (np.sign(audio) * compressed).astype(np.float32, copy=False)


def _shift_notes(notes: Sequence[Note], offset: int) -> List[Note]:
    return [Note(int(note.pitch), int(note.start) + offset, int(note.end) + offset) for note in notes]



def _split_notes_at_step(
    notes: Sequence[Note],
    boundary: int,
) -> tuple[List[Note], List[Note]]:
    """경계를 가로지르는 반주 노트를 잘라 CALL/RESPONSE 음량을 따로 준다."""
    before: List[Note] = []
    after: List[Note] = []

    for note in notes:
        start = int(note.start)
        end = int(note.end)
        if end <= boundary:
            before.append(Note(int(note.pitch), start, end))
        elif start >= boundary:
            after.append(Note(int(note.pitch), start, end))
        else:
            if boundary > start:
                before.append(Note(int(note.pitch), start, boundary))
            if end > boundary:
                after.append(Note(int(note.pitch), boundary, end))

    return before, after


def _cadence_notes(
    key_token: str,
    start_step: int,
    duration_steps: int = 4,
) -> tuple[List[Note], List[Note]]:
    root_pc, is_minor = _parse_key_token(key_token)
    quality = "min7" if is_minor else "maj7"
    bass_root = _nearest_pitch_class(root_pc, 36 + root_pc)
    comp_root = _nearest_pitch_class(root_pc, 48 + root_pc)
    end_step = int(start_step) + max(2, int(duration_steps))
    bass = [Note(bass_root, int(start_step), end_step)]
    comp = [
        Note(pitch, int(start_step), end_step)
        for pitch in _chord_voicing(comp_root, quality, "full")
    ]
    return bass, comp


def _get_fluid_renderer(sample_rate: int):
    global _FLUID_RENDERER

    if get_shared_renderer is None:
        return None

    if _FLUID_RENDERER is None:
        try:
            _FLUID_RENDERER = get_shared_renderer(sample_rate)
            print("[AUDIO] Shared persistent FluidSynth ready")
        except Exception as exc:
            print(
                "[AUDIO] FluidSynth unavailable, "
                f"fallback to fast synth: {exc}"
            )
            _FLUID_RENDERER = False

    return (
        _FLUID_RENDERER
        if _FLUID_RENDERER is not False
        else None
    )




def _soften_ai_piano(
    audio: np.ndarray,
) -> np.ndarray:
    """AI 피아노의 날카로운 고역과 짧은 clipping성 잡음을 줄인다."""
    if audio.size < 3:
        return audio.astype(np.float32, copy=False)

    source = audio.astype(np.float32, copy=False)
    softened = source.copy()
    softened[1:-1] = (
        0.20 * source[:-2]
        + 0.60 * source[1:-1]
        + 0.20 * source[2:]
    )

    # 강한 transient만 부드럽게 제한한다.
    return np.tanh(
        softened * 0.92
    ).astype(
        np.float32,
        copy=False,
    )

def build_round_audio(
    user_notes: Sequence[Note],
    ai_notes: Sequence[Note],
    bpm: int,
    *,
    sample_rate: int = 22050,
    ai_response_max_sec: float = 8.0,
    swing_amount: float = 0.64,
    key_token: Optional[str] = None,
    round_num: int = 1,
    exchange_num: int = 1,
    bass_gain: float = BASS_GAIN,
    comp_gain: float = COMP_GAIN,
) -> tuple[int, np.ndarray]:
    """부드러운 피아노와 마디별 적응형 반주를 stem 단위로 믹스한다."""
    del ai_response_max_sec, bass_gain, comp_gain

    renderer = _get_fluid_renderer(sample_rate)

    if not ai_notes or not key_token:
        if renderer is not None:
            preview = renderer.render(
                [(0, list(user_notes), 86)],
                bpm=bpm,
                swing_amount=0.0,
                tail_sec=0.42,
            )
        else:
            preview = synth_notes(
                list(user_notes),
                bpm=bpm,
                sample_rate=sample_rate,
                role="user",
                swing_amount=0.0,
            )

        return (
            int(sample_rate),
            (
                np.clip(preview, -1.0, 1.0)
                * 32767
            ).astype(np.int16),
        )

    user_end = max(
        (int(note.end) for note in user_notes),
        default=0,
    )

    # 사용자 끝 뒤 최소 한 박의 숨을 두고 다음 beat에서 AI가 시작한다.
    ai_start = ((user_end + 3) // 4) * 4
    if ai_start <= user_end:
        ai_start += 4

    ai_shifted = _shift_notes(
        ai_notes,
        ai_start,
    )

    total_end = max(
        (int(note.end) for note in ai_shifted),
        default=ai_start + STEPS_PER_BAR,
    )

    guide = list(user_notes) + ai_shifted

    bass_notes, comp_notes, meta = make_random_backing_layers(
        guide,
        key_token,
        round_num=round_num,
        exchange_num=exchange_num,
    )

    cadence_bass, cadence_comp = _cadence_notes(
        key_token,
        total_end,
        duration_steps=4,
    )
    bass_notes.extend(cadence_bass)
    comp_notes.extend(cadence_comp)

    bass_call, bass_answer = _split_notes_at_step(
        bass_notes,
        ai_start,
    )
    comp_call, comp_answer = _split_notes_at_step(
        comp_notes,
        ai_start,
    )

    if renderer is not None:
        user_audio = renderer.render(
            [(0, list(user_notes), 84)],
            bpm=bpm,
            swing_amount=0.0,
            tail_sec=0.52,
        )

        ai_audio = renderer.render(
            [(3, ai_shifted, 80)],
            bpm=bpm,
            swing_amount=max(0.64, swing_amount),
            tail_sec=0.58,
        )
        ai_audio = _soften_ai_piano(ai_audio)

        bass_audio = renderer.render(
            [
                (1, bass_call, 68),
                (1, bass_answer, 82),
            ],
            bpm=bpm,
            swing_amount=0.14,
            tail_sec=0.58,
        )

        comp_audio = renderer.render(
            [
                (2, comp_call, 56),
                (2, comp_answer, 70),
            ],
            bpm=bpm,
            swing_amount=max(0.62, swing_amount),
            tail_sec=0.64,
        )

        length = max(
            len(user_audio),
            len(ai_audio),
            len(bass_audio),
            len(comp_audio),
            1,
        )
        audio = np.zeros(
            length,
            dtype=np.float32,
        )

        # 리드는 줄이고 베이스와 컴핑은 실제로 들리게 한다.
        _mix_layer(audio, user_audio, 0.56)
        _mix_layer(audio, ai_audio, 0.58)
        _mix_layer(audio, bass_audio, 0.58)
        _mix_layer(audio, comp_audio, 0.46)

        audio = _soft_compress(
            audio,
            threshold=0.48,
            ratio=3.2,
        )

        peak = (
            float(np.max(np.abs(audio)))
            if audio.size
            else 0.0
        )
        if peak > 1e-8:
            audio = audio / peak * 0.86

        user_rms = float(np.sqrt(np.mean(np.square(user_audio))))
        ai_rms = float(np.sqrt(np.mean(np.square(ai_audio))))
        bass_rms = float(np.sqrt(np.mean(np.square(bass_audio))))
        comp_rms = float(np.sqrt(np.mean(np.square(comp_audio))))

    else:
        user_audio = synth_notes(
            list(user_notes),
            bpm=bpm,
            sample_rate=sample_rate,
            role="user",
            swing_amount=0.0,
        )
        ai_audio = synth_notes(
            ai_shifted,
            bpm=bpm,
            sample_rate=sample_rate,
            role="ai",
            swing_amount=max(0.64, swing_amount),
        )
        bass_audio = synth_notes(
            bass_notes,
            bpm=bpm,
            sample_rate=sample_rate,
            role="bass",
            swing_amount=0.14,
        )
        comp_audio = synth_notes(
            comp_notes,
            bpm=bpm,
            sample_rate=sample_rate,
            role="comp",
            swing_amount=max(0.62, swing_amount),
        )

        length = max(
            len(user_audio),
            len(ai_audio),
            len(bass_audio),
            len(comp_audio),
            1,
        )
        audio = np.zeros(length, dtype=np.float32)
        _mix_layer(audio, user_audio, 0.50)
        _mix_layer(audio, ai_audio, 0.52)
        _mix_layer(audio, bass_audio, 0.64)
        _mix_layer(audio, comp_audio, 0.52)

        audio = _soft_compress(
            audio,
            threshold=0.46,
            ratio=3.2,
        )
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1e-8:
            audio = audio / peak * 0.84

        user_rms = ai_rms = bass_rms = comp_rms = -1.0

    if meta:
        print(
            "[MUSIC V4] "
            f"chords={meta.get('chord_plan')} "
            f"bass={meta['bass_pattern_name']}({len(bass_notes)}) "
            f"comp={meta['comp_pattern_name']}({len(comp_notes)}) "
            f"fluid={renderer is not None} "
            f"rms=user:{user_rms:.4f}/ai:{ai_rms:.4f}/"
            f"bass:{bass_rms:.4f}/comp:{comp_rms:.4f} "
            f"ai_swing={max(0.64, swing_amount):.2f}"
        )

    return (
        int(sample_rate),
        (
            np.clip(audio, -1.0, 1.0)
            * 32767
        ).astype(np.int16),
    )
