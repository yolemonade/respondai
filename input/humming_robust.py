"""Fallback wrapper for sparse humming recognition.

The original PESTO-based recognizer is used first.
Only when it returns too few/too-short notes do we run a librosa pYIN
fallback. The fallback never invents new pitches: it uses median pitches
actually detected from the recorded contour.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from data.tokenizer import Note, STEPS_PER_BAR
from input.humming import humming_to_notes as _original_humming_to_notes


TARGET_SR = 22050
MAX_STEPS = 2 * STEPS_PER_BAR
MIN_USEFUL_NOTES = 3


def _load_audio(audio: Any) -> tuple[np.ndarray, int]:
    import librosa

    if audio is None:
        return np.zeros(0, dtype=np.float32), TARGET_SR

    if isinstance(audio, (str, Path)):
        y, sr = librosa.load(
            str(audio),
            sr=TARGET_SR,
            mono=True,
        )
        return y.astype(np.float32), TARGET_SR

    if (
        isinstance(audio, (tuple, list))
        and len(audio) == 2
    ):
        first, second = audio

        if np.isscalar(first):
            sr = int(first)
            y = np.asarray(second)
        elif np.isscalar(second):
            sr = int(second)
            y = np.asarray(first)
        else:
            raise ValueError("Unsupported audio tuple")

    else:
        sr = TARGET_SR
        y = np.asarray(audio)

    if y.ndim == 2:
        # Gradio may return (samples, channels) or (channels, samples).
        if y.shape[0] <= 2 and y.shape[1] > y.shape[0]:
            y = np.mean(y, axis=0)
        else:
            y = np.mean(y, axis=1)

    if np.issubdtype(y.dtype, np.integer):
        info = np.iinfo(y.dtype)
        y = y.astype(np.float32) / max(
            abs(info.min),
            info.max,
        )
    else:
        y = y.astype(np.float32)

    if sr != TARGET_SR:
        y = librosa.resample(
            y,
            orig_sr=sr,
            target_sr=TARGET_SR,
        )
        sr = TARGET_SR

    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1e-8:
        y = y / peak

    return y.astype(np.float32), sr


def _smooth_midi(
    midi: np.ndarray,
    radius: int = 2,
) -> np.ndarray:
    result = midi.copy()

    for index in range(len(midi)):
        left = max(0, index - radius)
        right = min(len(midi), index + radius + 1)
        values = midi[left:right]
        values = values[np.isfinite(values)]

        if values.size:
            result[index] = float(np.median(values))

    return result


def _target_steps(
    duration_seconds: float,
    bpm: float,
) -> int:
    safe_bpm = max(30.0, float(bpm))
    seconds_per_step = 60.0 / safe_bpm / 4.0

    return max(
        4,
        min(
            MAX_STEPS,
            int(round(duration_seconds / seconds_per_step)),
        ),
    )


def _uniform_contour_fallback(
    midi: np.ndarray,
    voiced: np.ndarray,
    target_steps: int,
) -> list[Note]:
    """긴 녹음이 한 음으로 합쳐졌을 때 박 단위로 다시 나눈다.

    음정을 새로 만들지는 않고 각 구간에서 실제 검출한 pitch median을 쓴다.
    """
    valid = np.flatnonzero(
        voiced & np.isfinite(midi)
    )

    if valid.size < 3:
        return []

    note_count = min(
        8,
        max(
            4,
            int(round(target_steps / 4.0)),
        ),
    )

    frame_edges = np.linspace(
        0,
        len(midi),
        note_count + 1,
    ).round().astype(int)

    step_edges = np.linspace(
        0,
        target_steps,
        note_count + 1,
    ).round().astype(int)

    global_pitch = int(
        round(
            float(
                np.nanmedian(
                    midi[voiced & np.isfinite(midi)]
                )
            )
        )
    )

    notes: list[Note] = []
    previous_pitch = global_pitch

    for index in range(note_count):
        frame_start = frame_edges[index]
        frame_end = max(
            frame_start + 1,
            frame_edges[index + 1],
        )

        values = midi[frame_start:frame_end]
        mask = voiced[frame_start:frame_end]
        values = values[
            mask & np.isfinite(values)
        ]

        if values.size:
            pitch = int(
                round(float(np.median(values)))
            )
            previous_pitch = pitch
        else:
            pitch = previous_pitch

        start = int(step_edges[index])
        next_start = int(step_edges[index + 1])

        if next_start <= start:
            continue

        # 음절이 분리되어 들리도록 약간의 gate를 둔다.
        width = next_start - start
        end = min(
            target_steps,
            start + max(1, width - 1),
        )

        if end > start:
            notes.append(
                Note(
                    pitch,
                    start,
                    end,
                )
            )

    return notes


def _pyin_fallback(
    audio: Any,
    bpm: float,
) -> list[Note]:
    import librosa

    y, sr = _load_audio(audio)

    if y.size < int(0.4 * sr):
        return []

    y, _ = librosa.effects.trim(
        y,
        top_db=38,
    )

    if y.size < int(0.35 * sr):
        return []

    duration = len(y) / float(sr)
    target_steps = _target_steps(
        duration,
        bpm,
    )

    hop_length = 256

    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C6"),
        sr=sr,
        frame_length=2048,
        hop_length=hop_length,
        fill_na=np.nan,
    )

    if f0 is None or len(f0) == 0:
        return []

    midi = librosa.hz_to_midi(f0)
    midi = _smooth_midi(midi)

    if voiced_flag is None:
        voiced_flag = np.isfinite(midi)

    if voiced_prob is None:
        voiced_prob = np.ones(
            len(midi),
            dtype=np.float32,
        )

    voiced = (
        np.asarray(voiced_flag, dtype=bool)
        & np.isfinite(midi)
        & (np.asarray(voiced_prob) >= 0.18)
    )

    if np.count_nonzero(voiced) < 3:
        return []

    onset_frames = librosa.onset.onset_detect(
        y=y,
        sr=sr,
        hop_length=hop_length,
        backtrack=True,
        units="frames",
        delta=0.06,
        wait=2,
    )

    boundaries = {
        0,
        len(midi),
    }

    # 실제 발음 어택을 음 경계로 사용.
    for frame in onset_frames:
        frame = int(frame)
        if 1 <= frame < len(midi) - 1:
            boundaries.add(frame)

    # 무음 구간 경계.
    for index in range(1, len(voiced)):
        if voiced[index] != voiced[index - 1]:
            boundaries.add(index)

    # 지속되는 반음 이상의 pitch 변화 경계.
    for index in range(3, len(midi) - 3):
        before = midi[index - 3:index]
        after = midi[index:index + 3]

        before = before[np.isfinite(before)]
        after = after[np.isfinite(after)]

        if (
            before.size >= 2
            and after.size >= 2
            and abs(
                float(np.median(after))
                - float(np.median(before))
            ) >= 0.65
        ):
            boundaries.add(index)

    ordered_boundaries = sorted(boundaries)

    # 너무 가까운 경계 제거.
    filtered = [ordered_boundaries[0]]

    for boundary in ordered_boundaries[1:]:
        if boundary - filtered[-1] >= 4:
            filtered.append(boundary)

    if filtered[-1] != len(midi):
        filtered.append(len(midi))

    notes: list[Note] = []

    for left, right in zip(
        filtered,
        filtered[1:],
    ):
        if right - left < 3:
            continue

        values = midi[left:right]
        mask = voiced[left:right]
        values = values[
            mask & np.isfinite(values)
        ]

        if values.size < 2:
            continue

        pitch = int(
            round(float(np.median(values)))
        )

        start = int(
            round(
                left
                / len(midi)
                * target_steps
            )
        )

        end = int(
            round(
                right
                / len(midi)
                * target_steps
            )
        )

        start = max(
            0,
            min(start, target_steps - 1),
        )

        end = max(
            start + 1,
            min(end, target_steps),
        )

        notes.append(
            Note(
                pitch,
                start,
                end,
            )
        )

    # 일반 분할이 실패한 경우 긴 contour를 박 단위로 재분할.
    if len(notes) < MIN_USEFUL_NOTES and duration >= 1.8:
        uniform = _uniform_contour_fallback(
            midi,
            voiced,
            target_steps,
        )

        if len(uniform) > len(notes):
            notes = uniform

    return notes



def _expand_sparse_humming(
    notes: list[Note],
    target_steps: int,
) -> list[Note]:
    """희소한 허밍을 녹음 길이에 맞추고 긴 음을 같은 pitch로 분할한다.

    새로운 음정을 만들지 않는다. 원래 검출된 pitch 순서만 반복한다.
    """
    if not notes or target_steps <= 0:
        return []

    ordered = sorted(
        notes,
        key=lambda note: (
            int(note.start),
            int(note.end),
            int(note.pitch),
        ),
    )

    origin = min(
        int(note.start)
        for note in ordered
    )

    source_end = max(
        int(note.end)
        for note in ordered
    )

    source_span = max(
        1,
        source_end - origin,
    )

    # 우선 기존 상대 위치와 음 길이를 목표 길이에 맞춘다.
    fitted: list[Note] = []

    for note in ordered:
        start = int(
            round(
                (
                    int(note.start)
                    - origin
                )
                / source_span
                * target_steps
            )
        )

        end = int(
            round(
                (
                    int(note.end)
                    - origin
                )
                / source_span
                * target_steps
            )
        )

        start = max(
            0,
            min(
                start,
                target_steps - 1,
            ),
        )

        end = max(
            start + 1,
            min(
                end,
                target_steps,
            ),
        )

        fitted.append(
            Note(
                int(note.pitch),
                start,
                end,
            )
        )

    # 마지막 검출 음이 전체 허밍 길이까지 이어지도록 한다.
    last = fitted[-1]

    fitted[-1] = Note(
        int(last.pitch),
        min(
            int(last.start),
            target_steps - 1,
        ),
        target_steps,
    )

    # 길이에 따라 4~8개 박스를 목표로 한다.
    desired_count = min(
        8,
        max(
            4,
            int(
                round(
                    target_steps / 4.0
                )
            ),
        ),
    )

    if len(fitted) >= desired_count:
        return fitted

    # 검출된 pitch 순서를 유지한 채 균등한 리듬 구간으로 재분할한다.
    edges = np.linspace(
        0,
        target_steps,
        desired_count + 1,
    ).round().astype(int)

    expanded: list[Note] = []

    for index in range(desired_count):
        source_index = min(
            len(fitted) - 1,
            int(
                index
                * len(fitted)
                / desired_count
            ),
        )

        pitch = int(
            fitted[source_index].pitch
        )

        start = int(
            edges[index]
        )

        next_start = int(
            edges[index + 1]
        )

        if next_start <= start:
            continue

        width = next_start - start

        # 각 음 사이에 최대 한 step의 작은 호흡을 둔다.
        if index < desired_count - 1 and width >= 3:
            end = next_start - 1
        else:
            end = next_start

        end = max(
            start + 1,
            min(
                end,
                target_steps,
            ),
        )

        expanded.append(
            Note(
                pitch,
                start,
                end,
            )
        )

    if expanded:
        last = expanded[-1]

        expanded[-1] = Note(
            int(last.pitch),
            int(last.start),
            target_steps,
        )

    return expanded

def humming_to_notes(
    audio: Any,
    bpm: float = 90,
    **kwargs,
) -> list[Note]:
    original: list[Note] = []

    try:
        original = _original_humming_to_notes(
            audio,
            bpm=bpm,
            **kwargs,
        )
    except Exception as exc:
        print(
            "[HUMMING ROBUST] original failed:",
            repr(exc),
        )

    original_span = max(
        (note.end for note in original),
        default=0,
    )

    # 기존 결과가 충분하면 그대로 사용.
    if (
        len(original) >= MIN_USEFUL_NOTES
        and original_span >= 8
    ):
        print(
            "[HUMMING ROBUST] "
            f"source=original notes={len(original)} "
            f"span={original_span}"
        )
        return original

    fallback: list[Note] = []

    try:
        fallback = _pyin_fallback(
            audio,
            bpm,
        )
    except Exception as exc:
        print(
            "[HUMMING ROBUST] fallback failed:",
            repr(exc),
        )

    fallback_span = max(
        (note.end for note in fallback),
        default=0,
    )

    if (
        len(fallback) > len(original)
        or fallback_span > original_span
    ):
        chosen = fallback
        source = "pyin"
    else:
        chosen = original
        source = "original"

    # 실제 녹음 길이를 게임의 최대 2마디 범위로 변환한다.
    try:
        raw_audio, raw_sr = _load_audio(audio)

        if raw_audio.size:
            import librosa

            trimmed_audio, _ = librosa.effects.trim(
                raw_audio,
                top_db=45,
            )

            duration_seconds = (
                len(trimmed_audio)
                / float(raw_sr)
            )
        else:
            duration_seconds = 0.0

    except Exception as exc:
        print(
            "[HUMMING ROBUST] duration check failed:",
            repr(exc),
        )

        duration_seconds = 0.0

    if duration_seconds > 0:
        target_steps = _target_steps(
            duration_seconds,
            bpm,
        )
    else:
        target_steps = max(
            4,
            min(
                MAX_STEPS,
                max(
                    (
                        int(note.end)
                        for note in chosen
                    ),
                    default=4,
                ),
            ),
        )

    expanded = _expand_sparse_humming(
        chosen,
        target_steps,
    )

    print(
        "[HUMMING ROBUST] "
        f"source={source} "
        f"duration={duration_seconds:.2f}s "
        f"target={target_steps} "
        f"original={len(original)}/{original_span} "
        f"fallback={len(fallback)}/{fallback_span} "
        f"chosen={len(chosen)} "
        f"expanded={len(expanded)} "
        f"end={max((note.end for note in expanded), default=0)} "
        f"pitches={[int(note.pitch) for note in expanded]}"
    )

    return expanded
