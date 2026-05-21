"""Phase 1 mock AI response and audio synthesis fallback."""
from __future__ import annotations

import numpy as np
from typing import List

from data.tokenizer import Note


def mock_ai_response(notes: List[Note]) -> List[Note]:
    """Shift user notes up by one octave (pitch +12), cap at 127."""
    if not notes:
        return []
    return [Note(min(n.pitch + 12, 127), n.start, n.end) for n in notes]


# ── [TEST / 임시] UI·게임오버 플레이스루 테스트용 ─────────────────────────────
# 실제 inference.generate() 연동(Phase 2) 시 아래 블록 전체를 제거하고
# app.py에서 mock_ai_response / generate()만 사용할 것.

TEMP_AI_TEST_MELODY: List[Note] = [
    Note(60, 0, 4),
    Note(64, 4, 8),
    Note(67, 8, 12),
    Note(72, 12, 16),
    Note(67, 16, 20),
    Note(64, 20, 24),
]


def temp_ai_response_for_testing(user_notes: List[Note]) -> List[Note]:
    """[TEST / 임시] Phase 1 UI 테스트용 AI 노트.

    - 유저 입력 없음(빈 확정): 고정 C major 아르페지오 멜로디
    - 유저 입력 있음: mock_ai_response (옥타브 +12)와 동일
    """
    if not user_notes:
        return list(TEMP_AI_TEST_MELODY)
    return mock_ai_response(user_notes)


def synth_notes(
    notes: List[Note],
    bpm: float = 90,
    sample_rate: int = 22050,
    *,
    role: str = "user",
) -> np.ndarray:
    """Synthesize notes to float32 mono audio using additive sine synthesis.

    Used as a FluidSynth fallback.
    """
    if not notes:
        return np.zeros(int(0.4 * sample_rate), dtype=np.float32)

    seconds_per_step = (60.0 / bpm) / 4.0
    total_dur = max(n.end for n in notes) * seconds_per_step + 0.3
    audio = np.zeros(int(total_dur * sample_rate), dtype=np.float32)

    for note in notes:
        freq = 440.0 * (2.0 ** ((note.pitch - 69) / 12.0))
        t_start = note.start * seconds_per_step
        t_end = note.end * seconds_per_step
        dur = t_end - t_start
        n_samples = int(dur * sample_rate)
        t = np.linspace(0.0, dur, n_samples, endpoint=False)

        if role == "ai":
            # [TEST / 임시] AI 구간은 더 밝은 톤으로 구분 (테스트 시 청취 확인용)
            wave = (
                np.sin(2 * np.pi * freq * t) * 0.35
                + np.sin(2 * np.pi * 2 * freq * t) * 0.40
                + np.sin(2 * np.pi * 3 * freq * t) * 0.20
            )
        else:
            wave = (
                np.sin(2 * np.pi * freq * t) * 0.50
                + np.sin(2 * np.pi * 2 * freq * t) * 0.25
                + np.sin(2 * np.pi * 3 * freq * t) * 0.12
                + np.sin(2 * np.pi * 4 * freq * t) * 0.06
            )
        # Piano-like envelope: sharp attack, exponential decay
        attack = min(0.01, dur * 0.05)
        attack_samples = max(1, int(attack * sample_rate))
        envelope = np.exp(-4.0 * t / max(dur, 0.01))
        envelope[:attack_samples] *= np.linspace(0, 1, attack_samples)
        wave *= envelope

        idx_start = int(t_start * sample_rate)
        idx_end = idx_start + n_samples
        if idx_end > len(audio):
            wave = wave[: len(audio) - idx_start]
            idx_end = len(audio)
        audio[idx_start:idx_end] += wave

    peak = np.abs(audio).max()
    if peak > 1e-6:
        audio /= peak * 1.1
    return audio
