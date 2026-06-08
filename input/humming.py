"""Humming mode (Phase 3, Beta) — mic audio → quantized Note sequence.

Pipeline (PROJECT.md 공통 파이프라인 진입 직전까지):

    mic audio (gr.Audio microphone)
        ↓ _to_mono_float   16kHz mono, normalize
        ↓ librosa.effects.trim   무음 구간 제거
        ↓ PESTO (ViT pitch detection)   연속 f0 + confidence
        ↓ confidence threshold (0.8)
        ↓ Hz → MIDI, median filter   pitch contour 스무딩
        ↓ _segment_notes   onset 경계 검출 → 음 분할
        ↓ _quantize   16분음표 그리드 nearest-neighbor snap
        ↓ _octave_correct   가창 음역대 보정
    List[Note]  (data.tokenizer.Note, 16분음표 스텝 단위)

정확도 한계는 PROJECT.md에 명시: f0→이산 음표 양자화는 음 경계·비브라토·
리듬 스냅 난이도가 높아 오류율이 있을 수 있다 (Beta 기능).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from data.tokenizer import Note, PITCH_MIN, PITCH_MAX

# ─── Constants ────────────────────────────────────────────────────────────────

PESTO_MODEL_NAME = "mir-1k_g7"     # PESTO 사전학습 체크포인트 (가창/허밍용)
STEP_MS          = 20.0            # PESTO 프레임 간격 (ms)
TARGET_SR        = 16000           # PROJECT.md: 허밍 샘플링 16kHz
CONF_THRESHOLD   = 0.8             # PROJECT.md: confidence 0.8 이상만 사용
MEDIAN_KERNEL    = 5               # pitch contour median filter 프레임 수 (홀수)
MIN_NOTE_SEC     = 0.08            # 이보다 짧은 음은 잡음으로 간주하고 폐기
TRIM_TOP_DB      = 30              # librosa.effects.trim 무음 임계
MAX_NOTES        = 16             # app.MAX_NOTES 와 동일 상한
MAX_BARS         = 4
STEPS_PER_BAR    = 16
# 가창 음역대 보정 목표 — median pitch 를 이 구간(대략 G3~G5)으로 옥타브 이동
VOCAL_LOW, VOCAL_HIGH = 55, 79


# ─── PESTO model (lazy, cached) ───────────────────────────────────────────────

_PESTO_MODEL = None


def _get_pesto():
    """PESTO 모델을 1회 로드해 캐시한다 (predict()는 매 호출 재로딩하므로 회피)."""
    global _PESTO_MODEL
    if _PESTO_MODEL is None:
        import pesto
        _PESTO_MODEL = pesto.load_model(
            PESTO_MODEL_NAME, step_size=STEP_MS, sampling_rate=TARGET_SR
        )
    return _PESTO_MODEL


# ─── Audio preprocessing ──────────────────────────────────────────────────────

def _to_mono_float(audio, sr_hint: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """gr.Audio 입력(또는 경로/배열)을 16kHz mono float32 로 정규화.

    허용 입력:
      - (sr, np.ndarray)   ← gr.Audio(type="numpy") 기본 반환
      - np.ndarray         ← sr_hint 필요
      - str (파일 경로)     ← librosa.load
    """
    import librosa

    if isinstance(audio, tuple):
        sr, y = audio
        y = np.asarray(y)
    elif isinstance(audio, str):
        y, sr = librosa.load(audio, sr=TARGET_SR, mono=True)
        return y.astype(np.float32), TARGET_SR
    else:
        y = np.asarray(audio)
        sr = sr_hint or TARGET_SR

    if y.ndim > 1:                      # stereo → mono
        y = y.mean(axis=1)
    y = y.astype(np.float32)

    # int16/int32 PCM → [-1, 1]
    peak = np.abs(y).max() if y.size else 0.0
    if peak > 1.0:
        y = y / 32768.0
        peak = np.abs(y).max() if y.size else 0.0
    if peak > 1e-6:
        y = y / peak                    # 정규화

    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
    return y.astype(np.float32), TARGET_SR


# ─── f0 extraction (PESTO) ────────────────────────────────────────────────────

def _extract_f0(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PESTO 로 프레임별 (시간[sec], MIDI pitch[float], confidence) 추출.

    load_model() 의 raw 모델 호출은 (pitch_MIDI, confidence, volume, activations)
    순서로 반환한다 (predict() 와 달리 timesteps 없음, pitch 는 이미 MIDI 단위).
    """
    import torch

    model = _get_pesto()
    x = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))
    with torch.no_grad():
        pitch_midi, conf, *_ = model(x)

    midi = pitch_midi.detach().cpu().numpy().astype(np.float32)
    c = conf.detach().cpu().numpy().astype(np.float32)
    t_sec = np.arange(len(midi), dtype=np.float32) * (STEP_MS / 1000.0)
    return t_sec, midi, c


# ─── Note segmentation ────────────────────────────────────────────────────────

def _smooth_contour(midi: np.ndarray, conf: np.ndarray,
                    conf_threshold: float) -> np.ndarray:
    """confidence 가 낮거나 무음인 프레임을 NaN 으로, 나머지는 median filter."""
    from scipy.signal import medfilt

    out = midi.copy()
    out[conf < conf_threshold] = np.nan
    out[np.isnan(midi)] = np.nan

    # NaN 을 건너뛰고 유효 프레임에만 median filter (잡음 점프 제거)
    valid = ~np.isnan(out)
    if valid.sum() >= MEDIAN_KERNEL:
        rounded = np.round(out)
        filled = rounded.copy()
        filled[~valid] = 0.0
        smoothed = medfilt(filled, kernel_size=MEDIAN_KERNEL)
        out = np.where(valid, smoothed, np.nan)
    else:
        out = np.round(out)
    return out


def _segment_notes(t_sec: np.ndarray, contour: np.ndarray
                   ) -> List[Tuple[int, float, float]]:
    """스무딩된 contour 를 (pitch, t_start, t_end) 음 단위로 분할.

    연속한 같은 반음 프레임을 하나의 음으로 묶고, 무음/피치 변화에서 끊는다.
    """
    if len(t_sec) < 2:
        return []
    frame_dt = float(np.median(np.diff(t_sec)))

    segments: List[Tuple[int, float, float]] = []
    cur_pitch: Optional[int] = None
    cur_start = 0.0

    def _close(end_t: float):
        nonlocal cur_pitch
        if cur_pitch is not None and (end_t - cur_start) >= MIN_NOTE_SEC:
            segments.append((cur_pitch, cur_start, end_t))
        cur_pitch = None

    for i, val in enumerate(contour):
        t = float(t_sec[i])
        if np.isnan(val):
            _close(t)
            continue
        p = int(val)
        if cur_pitch is None:
            cur_pitch, cur_start = p, t
        elif p != cur_pitch:
            _close(t)
            cur_pitch, cur_start = p, t
    _close(float(t_sec[-1]) + frame_dt)
    return segments


# ─── Quantization to 16th-note grid ───────────────────────────────────────────

def _quantize(segments: List[Tuple[int, float, float]], bpm: float) -> List[Note]:
    """초 단위 음 구간을 16분음표 스텝 그리드로 nearest-neighbor snap."""
    if not segments:
        return []
    sec_per_step = (60.0 / bpm) / 4.0
    t0 = segments[0][1]                       # 첫 음을 0 스텝으로 정렬

    notes: List[Note] = []
    for pitch, s, e in segments:
        start = int(round((s - t0) / sec_per_step))
        end = int(round((e - t0) / sec_per_step))
        if end <= start:
            end = start + 1                   # 최소 1 스텝 보장
        # 직전 음과 겹치면 경계 정리 (monophonic)
        if notes and start < notes[-1].end:
            start = notes[-1].end
            if end <= start:
                end = start + 1
        notes.append(Note(pitch, start, end))
    return notes


def _octave_correct(notes: List[Note]) -> List[Note]:
    """전체 멜로디 median pitch 를 가창 음역대로 옥타브 이동 + 피아노 범위 클램프."""
    if not notes:
        return notes
    med = float(np.median([n.pitch for n in notes]))
    shift = 0
    while med + shift < VOCAL_LOW:
        shift += 12
    while med + shift > VOCAL_HIGH:
        shift -= 12

    out: List[Note] = []
    for n in notes:
        p = int(np.clip(n.pitch + shift, PITCH_MIN, PITCH_MAX))
        out.append(Note(p, n.start, n.end))
    return out


def _cap_length(notes: List[Note]) -> List[Note]:
    """음 개수·총 길이를 게임 입력 상한(MAX_NOTES, MAX_BARS)으로 제한."""
    max_span = MAX_BARS * STEPS_PER_BAR
    capped = [n for n in notes if n.start < max_span][:MAX_NOTES]
    return [Note(n.pitch, n.start, min(n.end, max_span)) for n in capped]


# ─── Public API ───────────────────────────────────────────────────────────────

def humming_to_notes(
    audio,
    bpm: float = 90,
    *,
    sr_hint: Optional[int] = None,
    conf_threshold: float = CONF_THRESHOLD,
) -> List[Note]:
    """허밍 마이크 입력을 양자화된 Note 시퀀스로 변환 (피아노 모드와 동일 포맷).

    Parameters
    ----------
    audio : (sr, np.ndarray) | np.ndarray | str
        gr.Audio(microphone) 가 넘기는 (sr, ndarray) 튜플 권장.
    bpm : float
        현재 라운드 BPM — 16분음표 그리드 스냅 기준.

    Returns
    -------
    List[Note]   비어 있으면 인식 실패 (호출부에서 안내 처리).
    """
    if audio is None:
        return []
    y, _ = _to_mono_float(audio, sr_hint)
    if y.size < TARGET_SR // 10:              # 0.1초 미만이면 무시
        return []

    import librosa
    y, _ = librosa.effects.trim(y, top_db=TRIM_TOP_DB)
    if y.size < TARGET_SR // 10:
        return []

    t_sec, midi, conf = _extract_f0(y)
    contour = _smooth_contour(midi, conf, conf_threshold)
    segments = _segment_notes(t_sec, contour)
    notes = _quantize(segments, bpm)
    notes = _octave_correct(notes)
    notes = _cap_length(notes)
    return notes


# ─── Self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    # 합성 "허밍" — C4, E4, G4 를 각 0.5초씩 (간단한 아르페지오)
    sr = TARGET_SR
    bpm = 90
    seq_hz = [261.63, 329.63, 392.00]        # C4 E4 G4
    parts = []
    for f in seq_hz:
        t = np.linspace(0, 0.5, int(0.5 * sr), endpoint=False)
        # 살짝의 비브라토 + 배음으로 목소리 흉내
        vib = 1.0 + 0.01 * np.sin(2 * np.pi * 5 * t)
        tone = (0.6 * np.sin(2 * np.pi * f * vib * t)
                + 0.3 * np.sin(2 * np.pi * 2 * f * t)
                + 0.1 * np.sin(2 * np.pi * 3 * f * t))
        env = np.minimum(1.0, np.minimum(t / 0.02, (0.5 - t) / 0.05))
        parts.append((tone * np.clip(env, 0, 1)).astype(np.float32))
        parts.append(np.zeros(int(0.05 * sr), dtype=np.float32))  # 짧은 쉼표
    y = np.concatenate(parts)

    notes = humming_to_notes((sr, y), bpm=bpm)
    print(f"입력: C4 E4 G4 (각 0.5초, bpm={bpm})")
    print(f"인식된 음 {len(notes)}개:")
    for n in notes:
        name = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"][n.pitch % 12]
        octave = n.pitch // 12 - 1
        print(f"  {name}{octave} (midi {n.pitch})  steps {n.start}–{n.end}")
    pitches_pc = [n.pitch % 12 for n in notes]
    expected = [0, 4, 7]                      # C E G pitch-class
    ok = pitches_pc == expected
    print(f"\n결과: {'✅ PASS' if ok else '⚠️  pitch-class ' + str(pitches_pc) + ' != ' + str(expected)}")
