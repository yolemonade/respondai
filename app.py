"""RespondAI — Phase 2 Gradio app (real model)."""
from __future__ import annotations

import base64
import io
import os
import random
import time
import wave as _wave
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm

# Use a Korean-capable font if available, else fall back gracefully
_KO_FONT = next(
    (f.name for f in _fm.fontManager.ttflist
     if "Nanum Gothic" in f.name or "Apple SD Gothic" in f.name),
    None,
)
if _KO_FONT:
    plt.rcParams["font.family"] = _KO_FONT
import matplotlib.patches as mpatches
import numpy as np
import gradio as gr

from data.tokenizer import Note, STEPS_PER_BAR, PITCH_MIN, PITCH_MAX
from analysis.scoring import score_response, session_summary, key_consistency, grade_from_total
from input.piano import synth_notes, mock_ai_response
from input.humming import humming_to_notes
from inference.generate import load_model_for_inference, generate_candidates
from analysis.rhythm import rhythm_similarity
from analysis.motif import motif_overlap
#from inference.decode import notes_to_wav

from input.jazz_backing import (
    build_round_audio as build_jazz_round_audio,
)

# ─── Constants ───────────────────────────────────────────────────────────────

MAX_EXCHANGES  = 3
MAX_NOTES      = 16
DEFAULT_DURATION = 2   # sixteenth-note steps per note slot
TOTAL_ROUNDS   = 2  # 테스트용 (발표 전 5로 복원) — MAX_EXCHANGES=3 고정
AI_MODE        = "model"

def _get_checkpoint_path() -> str:
    local = "checkpoints/best_inference.pt"
    if os.path.exists(local):
        return local
    from huggingface_hub import hf_hub_download
    return hf_hub_download(
        repo_id="uuyeong/respondai-model",
        filename="best_inference.pt",
        local_dir="checkpoints",
    )

CHECKPOINT_PATH = _get_checkpoint_path()

SAMPLE_RATE    = 22050

KEYS = [
    "C major", "G major", "D major", "F major", "Bb major",
    "A minor", "E minor", "D minor", "G minor", "B minor",
]
BPM_CHOICES = [70, 80, 90, 100, 110]

KEY_TO_TOKEN = {
    "C major": "C",  "G major": "G",  "D major": "D",  "F major": "F",
    "Bb major": "A#","A minor": "Am", "E minor": "Em", "D minor": "Dm",
    "G minor": "Gm", "B minor": "Bm",
}

ROUND_CONDITIONS = {
    1: "Play freely. This becomes your motif.",
    2: "Respond to the AI's melody.",
    3: "Recall the motif from Round 1.",
    4: "Focus on rhythm.",
    5: "Close with your opening motif.",
}

EXCHANGE_STEPS = (MAX_NOTES * DEFAULT_DURATION) + 8   # x-width per exchange slot

# AI 응답 길이: 유저 입력에 맞추되 재생은 AI_RESPONSE_MAX_SEC 로 상한
AI_MAX_BARS_CAP = 2
AI_RESPONSE_MAX_SEC = 4.0
AI_MAX_NOTES_EMPTY = 4

# 피아노 입력 타임라인 상한: 최대 2마디(16분음표 그리드)
INPUT_MAX_STEPS = AI_MAX_BARS_CAP * STEPS_PER_BAR

# 키보드 입력 확장: 흰 건반(a s d f g h j k l), 검은 건반(w e r t y u i o)
KB_WHITE_SEMIS = [0, 2, 4, 5, 7, 9, 11, 12, 14]          # C4..D5
KB_BLACK_SEMIS = [1, 3, 6, 8, 10, 13, 15, 18]            # C#4..F#5
KB_ALLOWED_MIDIS = sorted({60 + s for s in (KB_WHITE_SEMIS + KB_BLACK_SEMIS)})

# 건반 위에 표시할 키보드 단축키 레이블 (영문/한글)
# 흰 a(60) s(62) d(64) f(65) g(67) h(69) j(71) k(72) l(74)
# 검 w(61) e(63) r(66) t(68) y(70) u(73) i(75) o(78)
MIDI_KEY_LABEL = {
    60: "A/ㅁ", 61: "W/ㅈ", 62: "S/ㄴ", 63: "E/ㄷ", 64: "D/ㅇ",
    65: "F/ㄹ", 66: "R/ㄱ", 67: "G/ㅎ", 68: "T/ㅅ", 69: "H/ㅗ",
    70: "Y/ㅛ", 71: "J/ㅓ", 72: "K/ㅏ", 73: "U/ㅕ", 74: "L/ㅣ",
    75: "I/ㅑ", 78: "O/ㅐ",
}


def _step_sec(bpm: int) -> float:
    return (60.0 / bpm) / 4.0


def ai_timeline_cap_steps(bpm: int) -> int:
    """AI 노트 타임라인 상한(16분음표 스텝) — 약 4초."""
    return max(4, int(AI_RESPONSE_MAX_SEC / _step_sec(bpm)))


def _user_time_span(user_notes: List[Note]) -> int:
    if not user_notes:
        return ai_timeline_cap_steps(90)
    return max(n.end for n in user_notes)


def generation_limits(user_notes: List[Note]) -> tuple[int, int]:
    """(max_bars, max_new_tokens) — CALL 길이에 비례."""
    if not user_notes:
        return 1, 40
    span = _user_time_span(user_notes)
    user_bars = max(1, (span + STEPS_PER_BAR - 1) // STEPS_PER_BAR)
    max_bars = max(1, min(AI_MAX_BARS_CAP, user_bars))
    max_new_tokens = min(80, 20 + len(user_notes) * 8)
    return max_bars, max_new_tokens


def trim_note_timeline(
    notes: List[Note],
    max_span: int,
    max_notes: int,
) -> List[Note]:
    """첫 음을 0으로 옮기되 모델이 만든 쉼표와 리듬은 유지한다."""
    if not notes:
        return []

    ordered = sorted(
        notes,
        key=lambda n: (n.start, n.end, n.pitch),
    )

    origin = ordered[0].start
    out: List[Note] = []

    for note in ordered:
        start = note.start - origin
        end = note.end - origin

        if start >= max_span:
            break

        end = min(end, max_span)

        if end <= start:
            continue

        out.append(
            Note(
                pitch=note.pitch,
                start=start,
                end=end,
            )
        )

        if len(out) >= max_notes:
            break

    return out


def fit_ai_response_to_user(
    ai_notes: List[Note],
    user_notes: List[Note],
    bpm: int,
) -> List[Note]:
    if not ai_notes:
        return []

    cap_steps = ai_timeline_cap_steps(bpm)
    user_span = (
        max(n.end for n in user_notes)
        if user_notes
        else cap_steps // 2
    )

    if user_notes:
        max_notes = min(
            MAX_NOTES,
            max(6, len(user_notes) + 4),
        )
        max_span = min(
            user_span + 8,
            cap_steps,
        )
    else:
        max_notes = AI_MAX_NOTES_EMPTY
        max_span = cap_steps

    return trim_note_timeline(
        ai_notes,
        max_span=max_span,
        max_notes=max_notes,
    )


def align_attn_to_notes(attn_scores: List[float], notes: List[Note]) -> List[float]:
    if not attn_scores or not notes:
        return attn_scores or []
    n = len(notes)
    if len(attn_scores) >= n:
        step = len(attn_scores) / n
        return [attn_scores[int(i * step)] for i in range(n)]
    return attn_scores + [attn_scores[-1]] * (n - len(attn_scores))


# ─── Model (앱 시작 시 1회 로드) ─────────────────────────────────────────────

print(f"[RespondAI] Loading model from {CHECKPOINT_PATH} ...")
_MODEL, _TOKENIZER, _DEVICE = load_model_for_inference(CHECKPOINT_PATH)
print(f"[RespondAI] Model ready on {_DEVICE} ({sum(p.numel() for p in _MODEL.parameters()):,} params)")


NUM_CANDIDATES = 4  # 배치로 동시 생성할 후보 수 (비용은 1개 생성과 거의 동일)


def _copy_ratio(cand: List[Note], user_notes: List[Note]) -> float:
    """후보가 CALL 을 위치까지 그대로 베낀 정도 (0~1).

    같은 인덱스의 pitch 일치 비율 — 모티프를 *다른 위치/변형*으로 재사용하면
    낮게, 처음부터 그대로 따라 치면 높게 나온다.
    """
    a = [n.pitch for n in user_notes]
    b = [n.pitch for n in cand]
    if not a or not b:
        return 0.0
    k = min(len(a), len(b))
    return sum(x == y for x, y in zip(a, b)) / k

def _target_score(
    value: float,
    target: float,
    tolerance: float,
) -> float:
    return max(
        0.0,
        1.0 - abs(value - target) / tolerance,
    )


def _melodic_variety(notes: List[Note]) -> float:
    if len(notes) < 2:
        return 0.0

    pitches = [n.pitch for n in notes]
    intervals = [
        b - a
        for a, b in zip(pitches, pitches[1:])
    ]

    unique_pitch_ratio = (
        len(set(pitches)) / len(pitches)
    )

    unique_interval_ratio = (
        len(set(intervals)) / len(intervals)
        if intervals else 0.0
    )

    pitch_range_score = min(
        1.0,
        (max(pitches) - min(pitches)) / 12.0,
    )

    repeated_rate = sum(
        a == b
        for a, b in zip(pitches, pitches[1:])
    ) / max(1, len(pitches) - 1)

    large_leap_rate = sum(
        abs(interval) >= 12
        for interval in intervals
    ) / max(1, len(intervals))

    return (
        0.35 * unique_pitch_ratio
        + 0.35 * unique_interval_ratio
        + 0.30 * pitch_range_score
        - 0.75 * repeated_rate
        - 0.25 * large_leap_rate
    )

def _jazz_rhythm_score(notes: List[Note]) -> float:
    """싱코페이션·쉼표·길이 변화가 있는 후보에 작은 보너스를 준다."""
    if len(notes) < 2:
        return 0.0

    ordered = sorted(notes, key=lambda n: (n.start, n.end, n.pitch))
    starts = [n.start for n in ordered]
    durations = [max(1, n.end - n.start) for n in ordered]

    offbeat_rate = sum(start % 4 != 0 for start in starts) / len(starts)
    offbeat_target = max(0.0, 1.0 - abs(offbeat_rate - 0.42) / 0.42)

    gaps = [
        max(0, right.start - left.end)
        for left, right in zip(ordered, ordered[1:])
    ]
    gap_rate = sum(gap > 0 for gap in gaps) / max(1, len(gaps))
    duration_variety = min(1.0, len(set(durations)) / 4.0)

    return (
        0.45 * offbeat_target
        + 0.30 * duration_variety
        + 0.25 * min(1.0, gap_rate * 2.0)
    )


def _rank_candidate(
    cand: List[Note],
    user_notes: List[Note],
    key_token: str,
    previous_ai_notes: Optional[List[Note]] = None,
) -> float:
    if not cand:
        return -1.0

    key_score = key_consistency(
        cand,
        key_token,
    )

    if not user_notes:
        return (
            1.5 * key_score
            + 0.8 * _melodic_variety(cand)
        )

    rhythm = max(
        0.0,
        rhythm_similarity(user_notes, cand),
    )
    motif = motif_overlap(
        user_notes,
        cand,
    )

    # 완전히 같은 리듬보다 적당히 관련된 변형을 선호
    rhythm_relation = _target_score(
        rhythm,
        target=0.50,
        tolerance=0.55,
    )

    # 모티프도 100% 복사보다 30~50% 활용을 선호
    motif_relation = _target_score(
        motif,
        target=0.40,
        tolerance=0.40,
    )

    copy_penalty = _copy_ratio(
        cand,
        user_notes,
    )

    score = (
        1.50 * key_score
        + 0.65 * rhythm_relation
        + 0.85 * motif_relation
        + 0.80 * _melodic_variety(cand)
        + 0.45 * _jazz_rhythm_score(cand)
        - 1.10 * copy_penalty
    )

    # 이전 AI 응답과 또 비슷한 결과가 나오는 것도 억제
    if previous_ai_notes:
        previous_motif = motif_overlap(
            previous_ai_notes,
            cand,
        )
        previous_rhythm = max(
            0.0,
            rhythm_similarity(
                previous_ai_notes,
                cand,
            ),
        )

        score -= 0.55 * previous_motif
        score -= 0.25 * previous_rhythm

    return score


def generate_ai_notes(
    user_notes: List[Note],
    key_token: str,
    bpm: int,
    previous_ai_notes: Optional[List[Note]] = None,
):
    max_bars, max_new_tokens = generation_limits(
        user_notes,
    )

    candidates = generate_candidates(
        _MODEL,
        _TOKENIZER,
        user_notes,
        key=key_token,
        tempo=bpm,
        num_candidates=NUM_CANDIDATES,
        max_bars=max_bars,
        max_new_tokens=max_new_tokens,

        # 후보마다 안정적 → 모험적인 순서
        temperature=(0.82, 0.95, 1.08, 1.18),
        top_p=0.95,

        # pitch 반복만 완만하게 억제
        repetition_penalty=1.10,
        rep_window=8,
    )

    scored = []

    for candidate in candidates:
        fitted = fit_ai_response_to_user(
            candidate,
            user_notes,
            bpm,
        )

        if not fitted:
            continue

        score = _rank_candidate(
            fitted,
            user_notes,
            key_token,
            previous_ai_notes,
        )

        scored.append((score, fitted))

    if scored:
        scored.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        # 1등과 2등 차이가 작으면 항상 같은 안전한 후보를
        # 선택하지 않고 75:25로 선택
        if (
            len(scored) >= 2
            and scored[0][0] - scored[1][0] < 0.25
        ):
            chosen = random.choices(
                scored[:2],
                weights=[0.75, 0.25],
                k=1,
            )[0]
        else:
            chosen = scored[0]

        return chosen[1], None

    fallback = (
        mock_ai_response(user_notes)
        if user_notes
        else []
    )

    return (
        fit_ai_response_to_user(
            fallback,
            user_notes,
            bpm,
        )
        or fallback,
        None,
    )

# ─── State helpers ───────────────────────────────────────────────────────────

def init_state() -> dict:
    return {
        "screen": "S1",
        "round": 1,
        "exchange": 1,
        "phase": "user_input",
        "current_notes": [],
        "ai_notes": [],
        "exchange_log": [],   # [{user_notes, ai_notes, score}, ...]
        "round_results": [],  # [{round_num, key, bpm, total, exchange_scores, grade}, ...]
        "r1_motif": [],
        "key": "C major",
        "bpm": 90,
        "mode": "piano",
        "total_score": 0,

        "cursor_step": 0,
        "note_duration": 2,
        "input_events": [],
        "swing_amount": 0.60,
    }


def round_grade(score: int) -> str:
    if score >= 950: return "⭐⭐⭐ Perfect!"
    if score >= 800: return "⭐⭐ Well played!"
    if score >= 600: return "⭐ Nicely done!"
    return "A little short — keep going!"


def reset_note_input(st: dict) -> None:
    """현재 교환의 입력 내용만 초기화한다.

    note_duration은 사용자가 UI에서 고른 값을 유지한다.
    """
    st["current_notes"] = []
    st["input_events"] = []
    st["cursor_step"] = 0
    st.setdefault("note_duration", DEFAULT_DURATION)


def _notes_from_input_events(events: List[dict]) -> List[Note]:
    """입력 이벤트 중 note 이벤트만 Note 목록으로 변환한다."""
    notes: List[Note] = []
    for event in events:
        if event.get("kind") != "note":
            continue
        notes.append(
            Note(
                int(event["pitch"]),
                int(event["start"]),
                int(event["end"]),
            )
        )
    return notes


def _event_cursor(events: List[dict]) -> int:
    """마지막 note/rest 이벤트가 끝난 스텝을 반환한다."""
    if not events:
        return 0
    return int(events[-1]["end"])


# ─── Scoring ─────────────────────────────────────────────────────────────────

def _note_count_scale(n: int) -> float:
    """입력 음 개수에 따른 스케일 팩터 — 짧은 입력은 낮게, 충분한 입력은 만점 허용."""
    if n <= 0: return 0.0
    if n == 1: return 0.08
    if n == 2: return 0.20
    if n == 3: return 0.38
    if n == 4: return 0.55
    if n == 5: return 0.68
    if n <= 7: return 0.80
    if n <= 9: return 0.92
    return 1.0


def compute_exchange_score(
    user_notes: List[Note],
    ai_prev_notes: List[Note],
    r1_motif: List[Note],
    round_num: int,
    exchange_num: int,
    key_token: str,
) -> dict:
    """Compute per-exchange score according to prompt.md spec."""
    if not user_notes:
        return {
            "key_consistency": 0, "rhythm_similarity": 0,
            "motif_usage": 0, "creativity_bonus": 0,
            "total": 0, "feedback": "No input received.",
            "raw": {"key_consistency": 0.0, "rhythm_pearson": 0.0, "motif_overlap": 0.0},
        }

    scale = _note_count_scale(len(user_notes))

    # R1/E1: auto 300 for rhythm + motif; only key computed
    if round_num == 1 and exchange_num == 1:
        kc = int(key_consistency(user_notes, key_token) * 300)
        rhythm_score = int(300 * scale)
        motif_score  = int(300 * scale)
        cb = int(50 * scale)
        return {
            "key_consistency": kc, "rhythm_similarity": rhythm_score,
            "motif_usage": motif_score, "creativity_bonus": cb,
            "total": kc + rhythm_score + motif_score + cb,
            "feedback": "Motif saved as your reference.",
            "raw": {"key_consistency": kc / 300, "rhythm_pearson": scale, "motif_overlap": scale},
        }

    # Rhythm: compare previous AI notes (or scaled auto if E1)
    if exchange_num == 1 or not ai_prev_notes:
        rhythm_score = int(300 * scale)
        rhythm_raw = scale
    else:
        tmp = score_response(ai_prev_notes, user_notes, key_token)
        rhythm_score = int(tmp["rhythm_similarity"] * scale)
        rhythm_raw = tmp["raw"]["rhythm_pearson"]

    # Key consistency: always computable (not scaled — pitch choices always count)
    base = score_response(ai_prev_notes or user_notes, user_notes, key_token)
    key_score = base["key_consistency"]

    # Creativity: comparing user vs AI-prev; auto-award if no AI response yet
    if not ai_prev_notes or exchange_num == 1:
        creativity = int(50 * scale)
    else:
        creativity = int(base["creativity_bonus"] * scale)

    # Motif: compare against r1_motif (or ai_prev if no motif yet)
    # r1_motif needs ≥3 notes to form n-grams (n=3 → 2 intervals); scaled auto if too short
    if r1_motif and len(r1_motif) >= 3:
        motif_res = score_response(r1_motif, user_notes, key_token)
        motif_score = int(motif_res["motif_usage"] * scale)
        motif_raw = motif_res["raw"]["motif_overlap"]
    elif r1_motif and len(r1_motif) < 3:
        motif_score = int(300 * scale)
        motif_raw = scale
    else:
        motif_score = int(base["motif_usage"] * scale)
        motif_raw = base["raw"]["motif_overlap"]

    print(f"[SCORE] R{round_num}/E{exchange_num} | scale={scale:.2f} key={key_score} rhythm={rhythm_score} motif={motif_score} creativity={creativity} | r1_motif_len={len(r1_motif)} user_len={len(user_notes)} ai_prev_len={len(ai_prev_notes)} | motif_raw={motif_raw:.3f}")

    total = key_score + rhythm_score + motif_score + creativity

    return {
        "key_consistency": key_score,
        "rhythm_similarity": rhythm_score,
        "motif_usage": motif_score,
        "creativity_bonus": creativity,
        "total": total,
        "feedback": base["feedback"],
        "raw": {
            "key_consistency": base["raw"]["key_consistency"],
            "rhythm_pearson": rhythm_raw,
            "motif_overlap": motif_raw,
        },
    }


# ─── Audio ───────────────────────────────────────────────────────────────────

def audio_duration_ms(audio: tuple) -> int:
    """합성된 오디오 버퍼 실제 길이(ms) — synth_notes tail 포함."""
    sr, samples = audio
    n = len(samples)
    if n <= 0:
        return 800
    return int(n / sr * 1000) + 350


def estimate_playback_ms(user_notes: List[Note], ai_notes: List[Note], bpm: int) -> int:
    """오디오 재생 길이 추정(ms) — audio_duration_ms 폴백용."""
    step_sec = (60.0 / bpm) / 4.0
    gap = 0.25
    tail = 0.35  # synth_notes per-segment release

    def _end(notes: List[Note], empty: float) -> float:
        if not notes:
            return empty
        return max(n.end for n in notes) * step_sec + tail

    sec = _end(user_notes, 0.35) + gap + _end(ai_notes, 0.35) + 0.15
    return int(sec * 1000)


def audio_to_html(audio_tuple, *, notify_done: bool = False, playback_ms: int = 0) -> str:
    """Convert (sample_rate, int16_ndarray) to an autoplay HTML audio element.

    Uses a native <audio> element to bypass WaveSurfer's retained playback
    position bug in gr.Audio — each call creates a fresh DOM element at t=0.

    notify_done=True 이면 재생 종료(onended) 시 숨김 버튼 #btn-audio-done 을
    클릭해 서버에 '재생 끝' 이벤트를 전달한다. playback_ms 는 data-ms 속성에
    실려 confirm 체인의 워치독(WATCHDOG_JS) 타임아웃 계산에 쓰인다.
    """
    sample_rate, data = audio_tuple
    buf = io.BytesIO()
    with _wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(data.tobytes())
    b64 = base64.b64encode(buf.getvalue()).decode()
    if not notify_done:
        return (
            f'<audio autoplay src="data:audio/wav;base64,{b64}" '
            f'style="display:none" id="s3-audio-el"></audio>'
        )
    _click = (
        "clearTimeout(window._raWd);"
        "(function(){var b=document.getElementById('btn-audio-done');"
        "var t=b?(b.querySelector('button')||b):null;if(t)t.click();})();"
    )
    return (
        f'<audio autoplay src="data:audio/wav;base64,{b64}" '
        f'style="display:none" id="s3-audio-el" data-ms="{int(playback_ms)}" '
        f'onended="{_click}" onerror="{_click}"></audio>'
    )


def build_round_audio(
    user_notes: List[Note],
    ai_notes: List[Note],
    bpm: int,
    swing_amount: float = 0.60,
    key_token: Optional[str] = None,
    round_num: int = 1,
    exchange_num: int = 1,
) -> tuple:
    """Delegate rendering to the random jazz backing module."""
    return build_jazz_round_audio(
        user_notes,
        ai_notes,
        bpm,
        sample_rate=SAMPLE_RATE,
        ai_response_max_sec=AI_RESPONSE_MAX_SEC,
        swing_amount=swing_amount,
        key_token=key_token,
        round_num=round_num,
        exchange_num=exchange_num,
    )


# ─── Piano roll rendering ────────────────────────────────────────────────────

ROLL_DPI        = 200
NOTE_USER_COLOR = "#2563EB"
NOTE_AI_COLOR   = "#F0594B"


def _gradient_image(h: int = 440, w: int = 1320) -> np.ndarray:
    """Soft diagonal gradient — lavender (top-left) → sky → teal (bottom-right)."""
    lav  = np.array([0.922, 0.906, 0.973])
    sky  = np.array([0.639, 0.804, 0.882])
    teal = np.array([0.145, 0.404, 0.475])
    yy, xx = np.mgrid[0:h, 0:w]
    d = (xx / (w - 1)) * 0.42 + (yy / (h - 1)) * 0.58

    def _smooth(t):
        t = np.clip(t, 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    lo_w = _smooth(d / 0.52)                 # lavender → sky
    hi_w = _smooth((d - 0.52) / 0.48)        # sky → teal
    img = np.empty((h, w, 3), dtype=np.float32)
    for k in range(3):
        lo = lav[k] + (sky[k] - lav[k]) * lo_w
        hi = sky[k] + (teal[k] - sky[k]) * hi_w
        img[..., k] = np.where(d < 0.52, lo, hi)
    return img


_GRADIENT_IMG = _gradient_image()

# 피아노롤 표시용 기본 세로 범위 (입력 없을 때)
ROLL_PITCH_DEFAULT_LO, ROLL_PITCH_DEFAULT_HI = 55, 80


def _clip_notes_to_slot(notes: List[Note], slot_width: int = EXCHANGE_STEPS) -> List[Note]:
    """EX 슬롯 밖으로 그려지지 않도록 타임라인 클립."""
    clipped: List[Note] = []
    for n in notes:
        if n.start >= slot_width:
            continue
        s = max(0, n.start)
        e = max(s + 1, min(n.end, slot_width))
        clipped.append(Note(n.pitch, s, e))
    return clipped


def _roll_pitch_range(note_groups: List[List[Note]]) -> tuple[int, int]:
    """실제 노트 피치에 맞춘 y축 (건반·모델 범위 내)."""
    pitches = [n.pitch for group in note_groups for n in group]
    if not pitches:
        return ROLL_PITCH_DEFAULT_LO, ROLL_PITCH_DEFAULT_HI
    lo = max(PITCH_MIN, min(pitches) - 4)
    hi = min(PITCH_MAX, max(pitches) + 4)
    if hi - lo < 12:
        mid = (lo + hi) // 2
        lo = max(PITCH_MIN, mid - 6)
        hi = min(PITCH_MAX, mid + 6)
    return lo, hi


def _current_exchange_index(state: dict) -> int:
    return max(0, min(int(state.get("exchange", 1)) - 1, MAX_EXCHANGES - 1))


def _finalize_roll_figure(fig, ax) -> None:
    """피그마 여백 제거 — 상단 검은 띠(투명 fig + dark 패널) 방지."""
    fig.patch.set_alpha(1.0)
    fig.patch.set_facecolor("#141414")
    ax.set_position([0.0, 0.0, 1.0, 1.0])
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)


def _draw_roll_notes(ax, notes, x_offset, color, alpha=0.95, linewidth=0.9):
    for n in notes:
        width = max(n.end - n.start, 1)
        rect = mpatches.FancyBboxPatch(
            (x_offset + n.start, n.pitch - 0.42), width, 0.84,
            boxstyle="round,pad=0.1",
            facecolor=color, edgecolor="white",
            linewidth=linewidth, alpha=alpha,
            transform=ax.transData, zorder=4,
        )
        ax.add_patch(rect)


def render_piano_roll(state: dict) -> plt.Figure:
    plt.close("all")
    fig, ax = plt.subplots(figsize=(9.8, 2.95), dpi=ROLL_DPI)

    exchange_log = state.get("exchange_log", [])
    current_notes = state.get("current_notes", [])
    cur_ex_idx = _current_exchange_index(state)
    all_groups: List[List[Note]] = []
    for entry in exchange_log:
        all_groups.append(entry.get("user_notes") or [])
        all_groups.append(entry.get("ai_notes") or [])
    all_groups.append(current_notes)

    pitch_lo, pitch_hi = _roll_pitch_range(all_groups)
    x0, x1 = 0, MAX_EXCHANGES * EXCHANGE_STEPS

    ax.imshow(_GRADIENT_IMG, extent=[x0, x1, pitch_lo, pitch_hi],
              aspect="auto", origin="upper", zorder=0,
              interpolation="bilinear")

    # Exchange separators
    for ex in range(1, MAX_EXCHANGES):
        ax.axvline(ex * EXCHANGE_STEPS, color="white", linewidth=1.1,
                   linestyle=(0, (4, 4)), alpha=0.5, zorder=2)

    # Exchange labels
    for ex in range(MAX_EXCHANGES):
        x = ex * EXCHANGE_STEPS
        ax.text(x + EXCHANGE_STEPS / 2, pitch_hi - 2.4,
                f"EX {ex+1}", ha="center", va="top",
                color="#3A3A55", fontsize=9, fontweight="bold",
                alpha=0.7, zorder=3)

    for i, entry in enumerate(exchange_log):
        off = i * EXCHANGE_STEPS
        _draw_roll_notes(
            ax, _clip_notes_to_slot(entry.get("user_notes") or []), off, NOTE_USER_COLOR,
        )
        _draw_roll_notes(
            ax, _clip_notes_to_slot(entry.get("ai_notes") or []), off, NOTE_AI_COLOR,
        )

    # 입력 중(미확정) 유저 노트 — 현재 EX 슬롯, 반투명
    _draw_roll_notes(
        ax, _clip_notes_to_slot(current_notes), cur_ex_idx * EXCHANGE_STEPS,
        NOTE_USER_COLOR, alpha=0.5,
    )

    ax.set_xlim(x0, x1)
    ax.set_ylim(pitch_lo, pitch_hi)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    user_patch = mpatches.Patch(facecolor=NOTE_USER_COLOR, edgecolor="white", label="You")
    ai_patch   = mpatches.Patch(facecolor=NOTE_AI_COLOR, edgecolor="white", label="AI")
    leg = ax.legend(
        handles=[user_patch, ai_patch],
        loc="lower right",
        bbox_to_anchor=(0.99, 0.02),
        bbox_transform=ax.transAxes,
        facecolor="white",
        edgecolor="none",
        labelcolor="#2A2A40",
        fontsize=8.5,
        framealpha=0.82,
        borderpad=0.6,
        handlelength=1.1,
    )
    leg.set_zorder(6)

    _finalize_roll_figure(fig, ax)
    return fig


def render_full_history_roll(round_results: List[dict]) -> plt.Figure:
    """Piano roll showing all rounds (for S4/S5)."""
    plt.close("all")
    fig, ax = plt.subplots(figsize=(9.8, 2.6), dpi=ROLL_DPI)

    round_width = MAX_EXCHANGES * EXCHANGE_STEPS + 16
    total_width = max(len(round_results) * round_width, 1)
    all_groups: List[List[Note]] = []
    for rr in round_results:
        for entry in rr.get("exchange_log", []):
            all_groups.append(entry.get("user_notes") or [])
            all_groups.append(entry.get("ai_notes") or [])
    pitch_lo, pitch_hi = _roll_pitch_range(all_groups)

    ax.imshow(_GRADIENT_IMG, extent=[0, total_width, pitch_lo, pitch_hi],
              aspect="auto", origin="upper", zorder=0,
              interpolation="bilinear")

    for r_idx, rr in enumerate(round_results):
        r_offset = r_idx * round_width
        if r_idx > 0:
            ax.axvline(r_offset, color="white", linewidth=1.3,
                       alpha=0.6, zorder=2)
        ax.text(r_offset + round_width / 2, pitch_hi - 2.4,
                f"R{rr['round_num']}", ha="center", va="top",
                color="#3A3A55", fontsize=9, fontweight="bold",
                alpha=0.7, zorder=3)
        for ex_idx, entry in enumerate(rr["exchange_log"]):
            ex_offset = r_offset + ex_idx * EXCHANGE_STEPS
            _draw_roll_notes(
                ax, _clip_notes_to_slot(entry.get("user_notes") or []), ex_offset,
                NOTE_USER_COLOR, linewidth=0.6,
            )
            _draw_roll_notes(
                ax, _clip_notes_to_slot(entry.get("ai_notes") or []), ex_offset,
                NOTE_AI_COLOR, linewidth=0.6,
            )

    ax.set_xlim(0, total_width)
    ax.set_ylim(pitch_lo, pitch_hi)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    _finalize_roll_figure(fig, ax)
    return fig


# ─── SVG energy visualization ─────────────────────────────────────────────────

def render_energy_svg(
    notes: List[Note],
    role: str = "player",
    attn_scores: Optional[List[float]] = None,
) -> str:
    """Circular energy visualization (VIZ-01 / VIZ-02).

    Player: pitch → hue (blue→purple), spoke length.
    AI (VIZ-02): attn_scores → spoke length + hue (orange→red), lightness tracks generation progress.
    """
    cx, cy, r_base = 80, 80, 45

    if not notes and not attn_scores:
        return (
            f'<svg width="160" height="160" style="background:#1C1C1C;border-radius:50%;">'
            f'<circle cx="{cx}" cy="{cy}" r="45" fill="none" '
            f'stroke="{"#2a2a3a" if role == "player" else "#3a2a2a"}" stroke-width="2"/>'
            f'<text x="{cx}" y="{cy+5}" text-anchor="middle" fill="#555" font-size="11">'
            f'{"🎹" if role == "player" else "🤖"}</text></svg>'
        )

    slices = []

    if role == "ai" and attn_scores:
        # VIZ-02: attention scores → spoke
        n = len(attn_scores)
        for i, score in enumerate(attn_scores):
            angle = (i / n) * 360
            norm = max(0.0, min(1.0, float(score)))
            progress = i / max(n - 1, 1)          # 0→1 as generation proceeds
            hue = 40 - norm * 40                   # orange(40°) → red(0°)
            sat = 70
            lightness = 30 + progress * 40         # 30→70% as per spec
            radius = r_base + norm * 28
            rad = np.radians(angle)
            x2 = cx + radius * np.cos(rad)
            y2 = cy + radius * np.sin(rad)
            slices.append(
                f'<line x1="{cx}" y1="{cy}" x2="{x2:.1f}" y2="{y2:.1f}" '
                f'stroke="hsl({hue:.0f},{sat}%,{lightness:.0f}%)" stroke-width="2.5" stroke-linecap="round"/>'
            )
    else:
        # VIZ-01: pitch → hue (blue 240° → purple 270°)
        n = len(notes)
        for i, note in enumerate(notes):
            angle = (i / n) * 360
            norm_pitch = max(0.0, min(1.0, (note.pitch - 40) / 50))
            radius = r_base + norm_pitch * 25
            hue = 240 + norm_pitch * 30
            sat = 70 + norm_pitch * 30
            rad = np.radians(angle)
            x2 = cx + radius * np.cos(rad)
            y2 = cy + radius * np.sin(rad)
            slices.append(
                f'<line x1="{cx}" y1="{cy}" x2="{x2:.1f}" y2="{y2:.1f}" '
                f'stroke="hsl({hue:.0f},{sat:.0f}%,50%)" stroke-width="2.5" stroke-linecap="round"/>'
            )

    inner_r = r_base - 10
    label = "🎹" if role == "player" else "🤖"
    return (
        f'<svg width="160" height="160" style="background:#1C1C1C;border-radius:50%;">'
        + "".join(slices)
        + f'<circle cx="{cx}" cy="{cy}" r="{inner_r}" fill="#1C1C1C" stroke="none"/>'
        + f'<text x="{cx}" y="{cy+5}" text-anchor="middle" fill="#888" font-size="14">{label}</text>'
        + f'</svg>'
    )


# ─── JS: elem_id nk-{midi} / btn-undo Gradio 버튼 클릭 ─────────────────────

KEYBOARD_JS = """() => {
  if (window.__respondAIKeyboardV === 12) return;
  window.__respondAIKeyboardV = 12;

  /* 확장 건반: 흰 a s d f g h j k l / 검 w e r t y u i o */
  const WHITE_SEMI = { KeyA:0, KeyS:2, KeyD:4, KeyF:5, KeyG:7, KeyH:9, KeyJ:11, KeyK:12, KeyL:14 };
  const BLACK_SEMI = { KeyW:1, KeyE:3, KeyR:6, KeyT:8, KeyY:10, KeyU:13, KeyI:15, KeyO:18 };
  const KO_WHITE = { 'ㅁ':0,'ㄴ':2,'ㅇ':4,'ㄹ':5,'ㅎ':7,'ㅗ':9,'ㅓ':11,'ㅏ':12,'ㅣ':14 };
  const KO_BLACK = { 'ㅈ':1,'ㄷ':3,'ㄱ':6,'ㅅ':8,'ㅛ':10,'ㅕ':13,'ㅑ':15,'ㅐ':18 };
  let baseOctave = 4;
  const activeKeys = new Set();

  function midiFromEvent(e) {
    if (WHITE_SEMI[e.code] !== undefined) return (baseOctave + 1) * 12 + WHITE_SEMI[e.code];
    if (BLACK_SEMI[e.code] !== undefined) return (baseOctave + 1) * 12 + BLACK_SEMI[e.code];
    const k = e.key.length === 1 ? e.key.toLowerCase() : e.key;
    if (k in {a:1,s:1,d:1,f:1,g:1,h:1,j:1,k:1,l:1}) {
      const map = {a:0,s:2,d:4,f:5,g:7,h:9,j:11,k:12,l:14};
      return (baseOctave + 1) * 12 + map[k];
    }
    if (k in {w:1,e:1,r:1,t:1,y:1,u:1,i:1,o:1}) {
      const map = {w:1,e:3,r:6,t:8,y:10,u:13,i:15,o:18};
      return (baseOctave + 1) * 12 + map[k];
    }
    if (KO_WHITE[e.key] !== undefined) return (baseOctave + 1) * 12 + KO_WHITE[e.key];
    if (KO_BLACK[e.key] !== undefined) return (baseOctave + 1) * 12 + KO_BLACK[e.key];
    return null;
  }

  function keyToken(e) {
    return e.code || e.key;
  }

  function readPhase() {
    if (window._raPhase) return window._raPhase;
    const hud = document.querySelector('.panel-s3:not(.hide) .ra-hud[data-phase]')
      || document.querySelector('.ra-hud[data-phase]');
    return hud ? (hud.dataset.phase || '') : '';
  }

  function isInputLocked() {
    const p = readPhase();
    return p === 'ai_response' || p === 'ai_playing';
  }

  function isVisible(el) {
    if (!el) return false;
    const cs = window.getComputedStyle ? getComputedStyle(el) : null;
    if (cs && (cs.display === 'none' || cs.visibility === 'hidden')) return false;
    const r = el.getBoundingClientRect ? el.getBoundingClientRect() : null;
    return !!(r && r.width > 0 && r.height > 0);
  }

  // 건반 입력(소리 포함)은 오직 실제 라운드의 피아노 입력 중에만 허용:
  // S3 화면 + 피아노 모드(화면 피아노가 보임) + AI 응답 중이 아님.
  // → 메인/점수/허밍 화면에서는 키보드를 눌러도 건반 소리가 나지 않음.
  function isGameInputActive() {
    if (isInputLocked()) return false;
    const roots = [document];
    document.querySelectorAll('gradio-app, .gradio-container').forEach(h => {
      if (h.shadowRoot) roots.push(h.shadowRoot);
    });
    for (const root of roots) {
      const host = root.querySelector('#s3-piano-host');
      if (host && isVisible(host)) return true;
    }
    return false;
  }

  function gradioBtn(id) {
    const roots = [document];
    document.querySelectorAll('gradio-app, .gradio-container').forEach(h => {
      if (h.shadowRoot) roots.push(h.shadowRoot);
    });
    let fallback = null;
    for (const root of roots) {
      const nodes = root.querySelectorAll('[id="' + id + '"]');
      for (const el of nodes) {
        const btn = el.tagName === 'BUTTON' ? el : el.querySelector('button');
        if (!btn) continue;
        if (!fallback) fallback = btn;
        if (isVisible(btn)) return btn;
      }
    }
    return fallback;
  }

  let audioCtx = null;
  let masterGain = null;
  function ensureAudioCtx() {
    if (!audioCtx) {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      masterGain = audioCtx.createGain();
      masterGain.gain.value = 0.9;
      masterGain.connect(audioCtx.destination);
    }
    if (audioCtx.state === 'suspended') {
      try { audioCtx.resume(); } catch (_) {}
    }
    return audioCtx;
  }

  function unlockAudio() {
    try {
      const ctx = ensureAudioCtx();
      if (ctx.state === 'suspended') ctx.resume();
    } catch (_) {}
  }

  function playPreviewTone(midi) {
    if (midi < 21 || midi > 108) return;
    try {
      const ctx = ensureAudioCtx();
      if (!ctx || !masterGain) return;
      const freq = 440 * Math.pow(2, (midi - 69) / 12);
      const osc = ctx.createOscillator();
      const osc2 = ctx.createOscillator();
      const gain = ctx.createGain();
      const t0 = ctx.currentTime;
      const dur = 0.26;
      osc.type = 'triangle';
      osc2.type = 'sine';
      osc.frequency.value = freq;
      osc2.frequency.value = freq * 2;
      gain.gain.setValueAtTime(0.0001, t0);
      gain.gain.exponentialRampToValueAtTime(0.55, t0 + 0.012);
      gain.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
      osc.connect(gain);
      osc2.connect(gain);
      gain.connect(masterGain);
      osc.start(t0);
      osc2.start(t0);
      osc.stop(t0 + dur + 0.04);
      osc2.stop(t0 + dur + 0.04);
    } catch (_) {}
  }

  function sendNote(midi) {
    if (midi === -1) {
      const u = gradioBtn('btn-undo');
      if (u) u.click();
      return;
    }
    playPreviewTone(midi);
    const btn = gradioBtn('nk-' + midi);
    if (btn) btn.click();
    else console.warn('[RespondAI] nk-' + midi + ' missing');
  }

  function flashKey(midi) {
    const kEl = document.querySelector('[data-midi="' + midi + '"]');
    if (!kEl) return;
    kEl.classList.add('active');
    setTimeout(() => kEl.classList.remove('active'), 150);
  }

  function clickConfirm() {
    const btn = gradioBtn('btn-confirm');
    if (btn && !btn.disabled) btn.click();
  }
  window.raClickConfirm = clickConfirm;

  // ── 허밍 모드 헬퍼 ─────────────────────────────────────────────────────
  // S3 마이크 호스트가 보이면 허밍 모드(녹음 입력) 상태.
  function hummingMicHost() {
    const roots = [document];
    document.querySelectorAll('gradio-app, .gradio-container').forEach(h => {
      if (h.shadowRoot) roots.push(h.shadowRoot);
    });
    for (const root of roots) {
      const host = root.querySelector('#s3-mic-host');
      if (host && isVisible(host)) return host;
    }
    return null;
  }
  // gr.Audio 의 "정지(Stop)" 버튼 — 녹음 중일 때만 존재(aria-label) 한다.
  function micStopButton() {
    const host = hummingMicHost();
    if (!host) return null;
    return host.querySelector('button[aria-label="Stop recording"]')
        || host.querySelector('button[aria-label*="Stop"]')
        || host.querySelector('.stop-button, button.stop-button')
        || null;
  }
  function isMicRecording() {
    return !!window.raIsRecording || !!micStopButton();
  }
  // Enter(허밍): 녹음 중이면 정지→인식→자동 전송, 정지 상태면 바로 전송.
  window.addEventListener('keydown', function(e) {
    if (e.key !== 'Enter' || e.repeat) return;
    if (!hummingMicHost()) return;           // 허밍 S3 화면이 아닐 때는 무시
    if (isInputLocked()) return;             // AI 응답 중에는 무시
    e.preventDefault();
    e.stopImmediatePropagation();
    if (isMicRecording()) {
      const stop = micStopButton();
      if (stop) {
        window.raHummingAutoConfirm = true;  // 인식 완료 후 자동 전송 플래그
        stop.click();                        // → stop_recording → PESTO 인식
      } else {
        clickConfirm();
      }
    } else {
      clickConfirm();                        // 이미 정지 상태 → 바로 전송
    }
  }, true);

  /* ─── UI 효과음 (Web Audio) ─── */
  function playBell(freq, t0, dur, peak, type) {
    const ctx = ensureAudioCtx();
    if (!ctx || !masterGain) return;
    const osc = ctx.createOscillator();
    const osc2 = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = type || 'sine';
    osc2.type = 'sine';
    osc.frequency.value = freq;
    osc2.frequency.value = freq * 2;
    gain.gain.setValueAtTime(0.0001, t0);
    gain.gain.exponentialRampToValueAtTime(peak, t0 + 0.012);
    gain.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
    osc.connect(gain); osc2.connect(gain); gain.connect(masterGain);
    osc.start(t0); osc2.start(t0);
    osc.stop(t0 + dur + 0.05); osc2.stop(t0 + dur + 0.05);
  }

  // 버튼 효과음 — 짧고 예쁜 "띵" (밝은 벨 한 음)
  function chime() {
    try {
      const ctx = ensureAudioCtx();
      const t0 = ctx.currentTime + 0.01;
      playBell(1318.5, t0, 0.45, 0.5, 'triangle');   // E6
    } catch (_) {}
  }

  // 라운드 결과 멜로디 — 'success'(신나는 상승) / 'fail'(짧은 하강 "뚜루루")
  function resultCue(kind) {
    try {
      const ctx = ensureAudioCtx();
      const t0 = ctx.currentTime + 0.04;
      if (kind === 'fail') {
        const seq = [392.00, 349.23, 311.13, 261.63];   // G4 F4 D#4 C4 (하강)
        seq.forEach((f, i) => playBell(f, t0 + i * 0.16, 0.34, 0.38, 'triangle'));
      } else {
        const seq = [523.25, 659.25, 783.99, 1046.50];  // C5 E5 G5 C6 (상승 아르페지오)
        seq.forEach((f, i) => playBell(f, t0 + i * 0.13, 0.40, 0.46, 'triangle'));
      }
    } catch (_) {}
  }

  window.respondAI = { sendNote, flashKey, chime, resultCue };

  // S4 결과 카드의 data-cue 를 읽어 성공/실패 멜로디 1회 재생 (화면 전환 JS에서 호출)
  window.raPlayResultCue = function() {
    try {
      const roots = [document];
      document.querySelectorAll('gradio-app, .gradio-container').forEach((h) => {
        if (h.shadowRoot) roots.push(h.shadowRoot);
      });
      for (const root of roots) {
        const panel = root.querySelector('.game-panel.panel-s4:not(.hide)');
        if (!panel) continue;
        const card = panel.querySelector('.ra-result-card[data-cue]:not([data-cue-done])');
        if (card) {
          card.setAttribute('data-cue-done', '1');
          window.respondAI && window.respondAI.resultCue && window.respondAI.resultCue(card.getAttribute('data-cue'));
        }
      }
    } catch (_) {}
  };

  // Simple modal controller for S1 nav popups (lives globally so onclick attrs work)
  if (!window.respondAIModal) {
    window.respondAIModal = {
      open(name) {
        const el = document.getElementById('ra-modal-' + name);
        if (!el) return;
        el.classList.add('open');
        document.documentElement.classList.add('ra-modal-locked');
      },
      close() {
        document.querySelectorAll('.ra-modal.open').forEach(m => m.classList.remove('open'));
        document.documentElement.classList.remove('ra-modal-locked');
      },
      howtoTab(name, btn) {
        const card = btn.closest('.ra-modal-card');
        if (!card) return;
        card.querySelectorAll('.ra-howto-tab').forEach(t => t.classList.toggle('active', t === btn));
        card.querySelectorAll('.ra-howto-pane').forEach(p => p.classList.toggle('active', p.dataset.pane === name));
      },
    };
    window.addEventListener('keydown', function(e) {
      if (e.key === 'Escape' && document.querySelector('.ra-modal.open')) {
      e.preventDefault();
        window.respondAIModal.close();
      }
    }, true);
  }

  window.addEventListener('pointerdown', unlockAudio, { capture: true });
  window.addEventListener('touchstart', unlockAudio, { capture: true });

  document.addEventListener('mousedown', function(e) {
    if (!isGameInputActive()) return;
    const key = e.target.closest && e.target.closest('[data-midi]');
    if (!key || !key.dataset.midi) return;
      e.preventDefault();
    const midi = parseInt(key.dataset.midi, 10);
    if (!isNaN(midi)) { sendNote(midi); flashKey(midi); }
  }, true);

  function isGameKey(e) {
    if (e.key === 'Enter' || e.key === 'Backspace') return true;
    return midiFromEvent(e) !== null;
  }

  // 옥타브 이동: Shift + ↑ / ↓ (입력 잠금 중이 아닐 때)
  window.addEventListener('keydown', function(e) {
    if (!e.shiftKey || (e.key !== 'ArrowUp' && e.key !== 'ArrowDown')) return;
    if (!isGameInputActive()) return;
    e.preventDefault();
    e.stopImmediatePropagation();
    if (e.key === 'ArrowUp')   baseOctave = Math.min(6, baseOctave + 1);
    if (e.key === 'ArrowDown') baseOctave = Math.max(2, baseOctave - 1);
  }, true);

  window.addEventListener('keydown', function(e) {
    if (!isGameKey(e)) return;
    if (!isGameInputActive()) return;
    if (e.repeat) return;
    e.preventDefault();
    e.stopImmediatePropagation();
    if (e.key === 'Backspace') { sendNote(-1); return; }
    if (e.key === 'Enter') { clickConfirm(); return; }
    const midi = midiFromEvent(e);
    const tok = keyToken(e);
    if (midi !== null && !activeKeys.has(tok)) {
      activeKeys.add(tok);
      if (midi >= 21 && midi <= 108) {
      sendNote(midi);
        flashKey(midi);
      }
    }
  }, true);

  window.addEventListener('keyup', function(e) {
    activeKeys.delete(keyToken(e));
  }, true);

  /* ─── Hero orbit controller — mouse-distance based parallax spin ─── */
  (function setupOrbit() {
    function attach() {
      const cosmos = document.querySelector('.ra-hero-cosmos');
      if (!cosmos) return false;
      const outer = cosmos.querySelector('.ra-orbit-outer');
      const inner = cosmos.querySelector('.ra-orbit-inner');
      if (!outer || !inner) return false;
      const outerNotes = Array.from(outer.querySelectorAll('.ra-orbit-note'));
      const innerNotes = Array.from(inner.querySelectorAll('.ra-orbit-note'));

      let activity = 0, target = 0, rotO = 0, rotI = 0;
      let last = performance.now();
      let reduced = false;
      try {
        const mq = window.matchMedia('(prefers-reduced-motion: reduce)');
        reduced = !!mq.matches;
        mq.addEventListener && mq.addEventListener('change', e => { reduced = !!e.matches; });
      } catch (_) {}

      function onMove(e) {
        const r = cosmos.getBoundingClientRect();
        if (!r.width || !r.height) return;
        const cx = r.left + r.width / 2;
        const cy = r.top  + r.height / 2;
        const d  = Math.hypot(e.clientX - cx, e.clientY - cy);
        const inR  = r.width * 0.30;   // fully active inside this radius
        const outR = r.width * 0.95;   // fully idle past this radius
        let t = d <= inR ? 1 : d >= outR ? 0 : 1 - (d - inR) / (outR - inR);
        // ease-out cubic on the linear ramp so transitions feel softer
        t = t <= 0 ? 0 : t >= 1 ? 1 : 1 - Math.pow(1 - t, 3);
        target = t;
      }
      function onLeave(e) {
        if (!e.relatedTarget && !e.toElement) target = 0;
      }
      window.addEventListener('mousemove', onMove, { passive: true });
      window.addEventListener('mouseout', onLeave, true);
      window.addEventListener('blur', () => { target = 0; });
      document.addEventListener('visibilitychange', () => {
        if (document.hidden) target = 0;
      });

      function applyTransform(layer, rot, notes) {
        layer.style.transform = `translate(-50%, -50%) rotate(${rot.toFixed(3)}deg)`;
        for (const g of notes) {
          const x = g.dataset.x, y = g.dataset.y, s = g.dataset.scale || '1';
          g.setAttribute('transform', `translate(${x},${y}) rotate(${(-rot).toFixed(3)}) scale(${s})`);
        }
      }
      function tick(now) {
        const dt = Math.min(0.05, (now - last) / 1000);
        last = now;
        // Smoothly ease activity toward target
        activity += (target - activity) * Math.min(1, dt * 4.5);
        const eff = reduced ? activity * 0.15 : activity;
        // Outer: slow clockwise   (baseline 0.4 → up to 7 deg/s at full activity)
        // Inner: faster counter-clockwise (baseline -0.6 → up to -14 deg/s) — parallax
        const speedO =  0.4 + eff * 6.6;
        const speedI = -(0.6 + eff * 13.4);
        rotO = (rotO + speedO * dt) % 360;
        rotI = (rotI + speedI * dt) % 360;
        applyTransform(outer, rotO, outerNotes);
        applyTransform(inner, rotI, innerNotes);
        requestAnimationFrame(tick);
      }
      requestAnimationFrame(tick);
      return true;
    }
    if (!attach()) {
      const obs = new MutationObserver(() => { if (attach()) obs.disconnect(); });
      obs.observe(document.body, { childList: true, subtree: true });
    }
  })();

  console.log('[RespondAI] keyboard ready');
}"""

STOP_AUDIO_JS = """() => {
  document.querySelectorAll('audio').forEach(a => {
    try { a.pause(); a.currentTime = 0; } catch (_) {}
  });
}"""

FOCUS_GAME_JS = """() => {
  const ae = document.activeElement;
  if (ae && (ae.tagName === 'INPUT' || ae.tagName === 'TEXTAREA')) ae.blur();
}"""

# 허밍 녹음 시작 — JS 플래그만 세팅 (서버 왕복 없음)
MIC_START_JS = """() => { window.raIsRecording = true; }"""

# 허밍 인식(stop_recording) 완료 후 — Enter 로 정지했으면 자동 전송
MIC_AFTER_STOP_JS = """() => {
  window.raIsRecording = false;
  if (window.raHummingAutoConfirm) {
    window.raHummingAutoConfirm = false;
    if (window.raClickConfirm) window.raClickConfirm();
  }
}"""

UNLOCK_INPUT_JS = """(phase) => {
  if (phase) window._raPhase = String(phase).trim();
}"""

# 패널 표시 — Python gr.update(visible/elem_classes) 대신 DOM .hide 토글
SHOW_SCREEN_JS = """(screenId) => {
  const id = String(screenId || 'S1').trim().toUpperCase();
  const target = 'panel-' + id.toLowerCase();
  const roots = [document];
  document.querySelectorAll('gradio-app, .gradio-container').forEach((h) => {
    if (h.shadowRoot) roots.push(h.shadowRoot);
  });
  for (const root of roots) {
    root.querySelectorAll('.game-stage .game-panel, .game-panel.panel-s1, .game-panel.panel-s2, .game-panel.panel-s3, .game-panel.panel-s4, .game-panel.panel-s5').forEach((el) => {
      if (el.classList.contains(target)) el.classList.remove('hide');
      else el.classList.add('hide');
    });
  }
  window.raPlayResultCue && window.raPlayResultCue();
}"""

# 버튼 클릭 효과음 ("띵")
CHIME_JS = """() => { try { window.respondAI && window.respondAI.chime && window.respondAI.chime(); } catch (_) {} }"""


# ─── Piano keyboard HTML (pure HTML/CSS, NO <script> — onclick calls window.respondAI) ──

def render_piano_html(base_octave: int = 4) -> str:
    NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    white_midis = [60 + s for s in KB_WHITE_SEMIS]
    black_midis = [60 + s for s in KB_BLACK_SEMIS]

    WW, WH = 64, 152
    BW, BH = 36, 92
    total_w = len(white_midis) * WW

    whites, blacks = [], []
    for wi, midi in enumerate(white_midis):
        left = wi * WW
        name = NOTE_NAMES[midi % 12] + str(midi // 12 - 1)
        kb_label = MIDI_KEY_LABEL.get(midi, "")
        whites.append(
            f'<div data-midi="{midi}"'
            f' class="ra-key ra-key-white"'
            f' style="position:absolute;left:{left}px;width:{WW-2}px;height:{WH}px;'
            f'cursor:pointer;display:flex;flex-direction:column;align-items:center;'
            f'justify-content:flex-end;padding-bottom:4px;box-sizing:border-box;">'
            f'<span class="ra-key-label">{name}</span>'
            + (f'<span class="ra-key-shortcut">{kb_label}</span>' if kb_label else '')
            + f'</div>'
        )

    for bi, midi in enumerate(black_midis):
        left = (bi + 1) * WW - BW // 2 - 1
        name = NOTE_NAMES[midi % 12] + str(midi // 12 - 1)
        kb_label = MIDI_KEY_LABEL.get(midi, "")
        blacks.append(
            f'<div data-midi="{midi}"'
            f' class="ra-key ra-key-black"'
            f' style="position:absolute;left:{left}px;top:0;width:{BW}px;height:{BH}px;'
            f'border-radius:0 0 3px 3px;cursor:pointer;z-index:2;'
            f'display:flex;flex-direction:column;align-items:center;'
            f'justify-content:flex-end;padding-bottom:3px;box-sizing:border-box;">'
            f'<span class="ra-key-label ra-key-label-black">{name}</span>'
            + (f'<span class="ra-key-shortcut ra-key-shortcut-black">{kb_label}</span>' if kb_label else '')
            + f'</div>'
        )

    return (
        f'<div class="ra-piano-wrap">'
        f'<div class="ra-piano-keys" style="position:relative;width:{total_w}px;height:{WH}px;">'
        + ''.join(whites)
        + ''.join(blacks)
        + '</div></div>'
    )


# ─── HUD / info HTML helpers ─────────────────────────────────────────────────

def hud_html(state: dict) -> str:
    phase = state["phase"]
    if phase == "user_input":
        badge = '<div class="ra-turn-badge ra-turn-player">● YOUR TURN</div>'
    elif phase == "ai_response":
        badge = '<div class="ra-turn-badge ra-turn-ai">● AI RESPONDING</div>'
    elif phase == "ai_playing":
        badge = '<div class="ra-turn-badge ra-turn-ai">● AI PLAYING</div>'
    elif phase == "round_result":
        badge = '<div class="ra-turn-badge ra-turn-result">● ROUND RESULT</div>'
    else:
        badge = '<div class="ra-turn-badge ra-turn-result">● SESSION END</div>'
    ex_done = len(state.get("exchange_log", []))
    return f"""
<div class="ra-hud" data-phase="{phase}" data-exchange-done="{ex_done}">
  <div class="ra-hud-group">
    <span class="ra-hud-round">R{state['round']}/{TOTAL_ROUNDS}</span>
    <span class="ra-hud-exchange">EX {state['exchange']}/{MAX_EXCHANGES}</span>
  </div>
  <div class="ra-hud-group">
    <span class="ra-chip">🎼 {state['key']}</span>
    <span class="ra-chip">♩ {state['bpm']} BPM</span>
  </div>
  <div class="ra-hud-group">
    <span class="ra-hud-score">{state['total_score']} pts</span>
  </div>
  {badge}
</div>
"""


def note_list_html(notes: List[Note]) -> str:
    if not notes:
        return '<div class="ra-note-empty">Press keys to add notes</div>'
    NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    names = [f"{NOTE_NAMES[n.pitch % 12]}{n.pitch // 12 - 1}" for n in notes]
    items = "".join(
        f'<span class="ra-note-chip">{name}</span>'
        for name in names
    )
    return (
        f'<div class="ra-note-list">'
        f'{items}'
        f'<span class="ra-note-count">{len(notes)}/{MAX_NOTES}</span>'
        f'</div>'
    )


def score_bar_html(label: str, value: int, max_val: int, color: str) -> str:
    pct = min(100, int(value / max_val * 100))
    return f"""
<div class="ra-score-row">
  <div class="ra-score-head">
    <span>{label}</span><span>{value}/{max_val}</span>
  </div>
  <div class="ra-score-track">
    <div style="background:{color};width:{pct}%;height:4px;border-radius:2px;transition:width .4s ease;"></div>
  </div>
</div>
"""


def s4_html(state: dict) -> str:
    results = state["round_results"]
    if not results:
        return ""
    rr = results[-1]
    exs = rr["exchange_scores"]
    round_total = int(sum(s["total"] for s in exs) / len(exs))
    grade = round_grade(round_total)

    bars = (
        score_bar_html("Key consistency",  exs[-1]["key_consistency"],   300, "#9B8FD4")
        + score_bar_html("Rhythm similarity", exs[-1]["rhythm_similarity"],300, "#7B9FD4")
        + score_bar_html("Motif usage",       exs[-1]["motif_usage"],       300, "#A8D8E8")
        + score_bar_html("Creativity",        exs[-1]["creativity_bonus"], 100, "#C8C5E8")
    )
    r5_bonus = ""
    if rr["round_num"] == 5 and rr.get("r5_motif_bonus"):
        r5_bonus = '<div class="ra-r5-bonus">● R5 motif bonus &nbsp;+150</div>'

    feedback = exs[-1].get("feedback", "")
    cue = "success" if round_total >= 600 else "fail"
    return f"""
<div class="ra-result-card" data-cue="{cue}">
  <div class="ra-round-label">Round {rr['round_num']} Result</div>
  <div class="ra-result-grade">{grade}</div>
  <div class="ra-result-total">{round_total} / 1000</div>
  <div class="ra-result-feedback">{feedback}</div>
  {bars}
  {r5_bonus}
</div>
"""


def s5_html(state: dict) -> str:
    round_results = state["round_results"]
    if not round_results:
        return ""

    # Compute final score
    # Collect per-round average scores for session_summary
    per_round_scores = []
    r5_motif_bonus = 0
    for rr in round_results:
        exs = rr["exchange_scores"]
        avg_total = int(sum(s["total"] for s in exs) / len(exs))
        # Rebuild a summary-compatible dict (scoring.session_summary takes list of exchange dicts)
        # We use the last exchange score as representative
        per_round_scores.append({**exs[-1], "total": avg_total})

    summary = session_summary(per_round_scores)

    # Add R5 motif bonus if applicable
    if round_results[-1].get("r5_motif_bonus"):
        r5_motif_bonus = 150

    final_total = summary["total"] + r5_motif_bonus
    grade = grade_from_total(final_total)

    grade_colors = {"S": "#FFD27A", "A": "#E8E2F2", "B": "#C8C5E8", "C": "#DCD7F0"}
    gc = grade_colors.get(grade, "#fff")

    bonus_detail = ""
    if summary["bonus_score"] or r5_motif_bonus:
        bonus_detail = (
            f'<div class="ra-final-bonus">'
            f'Combo / key bonus &nbsp;+{summary["bonus_score"]} &nbsp;·&nbsp; '
            f'R5 motif bonus &nbsp;+{r5_motif_bonus}'
            f'</div>'
        )

    # ── 좌측: 라운드별 결과 행 ────────────────────────────────────────────────
    def _stars(score: int) -> str:
        if score >= 950: return "⭐⭐⭐"
        if score >= 800: return "⭐⭐"
        if score >= 600: return "⭐"
        return "—"

    round_rows = ""
    for rr in round_results:
        rn = rr["round_num"]
        cond = ROUND_CONDITIONS.get(rn, "")
        round_rows += (
            f'<div class="s5-round-row">'
            f'<span class="s5-round-rn">R{rn}</span>'
            f'<span class="s5-round-cond">{cond}</span>'
            f'<span class="s5-round-stars">{_stars(rr["total"])}</span>'
            f'<span class="s5-round-score">{rr["total"]}</span>'
            f'</div>'
        )

    # ── 우측: 3개 항목 평균 게이지 ────────────────────────────────────────────
    all_ex = [e for rr in round_results for e in rr["exchange_scores"]]
    n_ex = max(1, len(all_ex))

    def _avg(metric: str) -> float:
        return sum(e[metric] for e in all_ex) / n_ex

    def _gauge(icon, label, desc, value, color, hint):
        pct = min(100, int(value / 300 * 100))
        hint_html = (
            f'<div class="s5-gauge-hint">{hint}</div>' if pct < 50 and hint else ""
        )
        return (
            f'<div class="s5-gauge">'
            f'<div class="s5-gauge-head"><span>{label}</span><span>{pct}%</span></div>'
            f'<div class="s5-gauge-desc">{desc}</div>'
            f'<div class="s5-gauge-track"><div class="s5-gauge-fill" style="width:{pct}%;background:{color};"></div></div>'
            f'{hint_html}'
            f'</div>'
        )

    gauges = (
        _gauge("", "조성 일관성", "라운드의 Key 스케일 안에서 연주한 비율",
               _avg("key_consistency"), "#7B9FD4",
               "다음엔 Key 스케일 안의 음 위주로 연주해보세요")
        + _gauge("", "리듬 호응", "AI의 리듬 패턴에 얼마나 맞춰 응답했는가",
                 _avg("rhythm_similarity"), "#6FBF8F",
                 "AI 응답의 박자감을 다음 입력에 반영해보세요")
        + _gauge("", "모티프 활용", "첫 라운드 멜로디를 얼마나 기억하고 활용했는가",
                 _avg("motif_usage"), "#9B8FD4",
                 "1라운드 첫 멜로디를 다시 사용해보세요")
    )

    flavor = '<div class="s5-flavor">당신의 연주 세션이 끝났습니다.</div>'

    return f"""
<div class="s5-stage s5-stage-v2">
  <div class="s5-label">● FINAL RESULT</div>
  <div class="s5-grade" style="color:{gc};">{grade}</div>
  <div class="s5-total">{final_total} <span class="s5-of">/ 5000</span></div>
  {flavor}
  {bonus_detail}
  <div class="s5-cols">
    <div class="s5-col s5-col-rounds">
      <div class="s5-col-cap">라운드별 결과</div>
      {round_rows}
    </div>
    <div class="s5-col s5-col-metrics">
      <div class="s5-col-cap">항목별 분석 · 5라운드 평균</div>
      {gauges}
    </div>
  </div>
  <div class="s5-roll-cap">● 세션 전체 피아노롤 · <span style="color:#7B9FD4;">파랑 You</span> / <span style="color:#FF4A6E;">빨강 AI</span></div>
</div>
"""


# ─── CSS ─────────────────────────────────────────────────────────────────────

APP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500&display=swap');

/* 흰색 배경 위 연한 회색 글씨 → 어둡게 강제 */
.gradio-container label,
.gradio-container label span,
.gradio-container legend,
.gradio-container fieldset > span,
.gradio-container .label-wrap,
.gradio-container .label-wrap span,
.gradio-container [class*="label"],
.gradio-container span.text-sm,
.gradio-container span.text-xs,
.gradio-container .caption-lg,
.gradio-container .gap > span,
.gradio-container .wrap > span,
.gradio-container p,
.gradio-container .prose p,
.gradio-container .block > span {
  color: #1A1A2E !important;
}
/* 마이크/오디오 컴포넌트 — 라이트 패널(S1/S2)에서만 어둡게 */
.panel-s1 #s3-mic-input *,
.panel-s1 #s3-mic-input,
.panel-s1 .s3-mic-host *,
.panel-s1 .s3-mic-host,
.panel-s2 #s3-mic-input *,
.panel-s2 #s3-mic-input,
.panel-s2 .s3-mic-host *,
.panel-s2 .s3-mic-host {
  color: #1A1A2E !important;
}
/* Gradio Audio 컴포넌트 내부 select/option/button — 라이트 패널 한정 */
.panel-s1 select, .panel-s2 select,
.panel-s1 select option, .panel-s2 select option,
.panel-s1 option, .panel-s2 option,
.panel-s1 [class*="select"] span, .panel-s2 [class*="select"] span,
.panel-s1 [class*="dropdown"] span, .panel-s2 [class*="dropdown"] span,
.panel-s1 [class*="device"] span, .panel-s2 [class*="device"] span,
.panel-s1 [class*="audio"] *, .panel-s2 [class*="audio"] *,
.panel-s1 [class*="waveform"] *, .panel-s2 [class*="waveform"] *,
.panel-s1 [class*="microphone"] *, .panel-s2 [class*="microphone"] * {
  color: #1A1A2E !important;
}

/* ── S3 다크 패널 — 모든 텍스트/레이블 흰색 강제 ──────────────────── */
/* 전역 #1A1A2E important를 같은 specificity로 덮으려면 panel-s3 prefix 사용 */
.panel-s3 label,
.panel-s3 label span,
.panel-s3 legend,
.panel-s3 fieldset > span,
.panel-s3 .label-wrap,
.panel-s3 .label-wrap span,
.panel-s3 [class*="label"],
.panel-s3 span.text-sm,
.panel-s3 span.text-xs,
.panel-s3 .caption-lg,
.panel-s3 .gap > span,
.panel-s3 .wrap > span,
.panel-s3 p,
.panel-s3 .prose p,
.panel-s3 .block > span {
  color: #F0F0F0 !important;
}
/* Note length Radio 박스 — 다크 배경 + 흰 글씨 (라이트 모드에서도 강제) */
.panel-s3 .rhythm-input-controls,
.panel-s3 .rhythm-input-controls * {
  background: #1C1C1C !important;
  color: #F0F0F0 !important;
}
/* Radio 선택 원형 인디케이터는 배경 투명 유지 */
.panel-s3 .rhythm-input-controls input[type="radio"] {
  background: transparent !important;
  accent-color: #9B8FD4;
}
.panel-s3 .s3-mic-host,
.panel-s3 .s3-mic-host *,
#s3-mic-host,
#s3-mic-host * {
  background-color: #1C1C1C !important;
  color: #F0F0F0 !important;
}
.panel-s3 .s3-mic-host button,
#s3-mic-host button {
  background: #2A2A3A !important;
  color: #F0F0F0 !important;
  border: 1px solid rgba(255,255,255,0.18) !important;
}
/* Audio 내부 타임코드·select·waveform — 다크 패널 흰색 */
.panel-s3 select,
.panel-s3 select option,
.panel-s3 option,
.panel-s3 [class*="select"] span,
.panel-s3 [class*="dropdown"] span,
.panel-s3 [class*="device"] span,
.panel-s3 [class*="audio"] *,
.panel-s3 [class*="waveform"] *,
.panel-s3 [class*="microphone"] * {
  background-color: #1C1C1C !important;
  color: #F0F0F0 !important;
}


/* 화면 밖 숨김 버튼 (audio onended 브리지) — display:none 이면 일부 환경에서
   programmatic click 이 무시되므로 위치만 밖으로 뺀다. */
.ra-hidden-btn {
  position: absolute !important;
  left: -9999px !important;
  width: 1px !important;
  height: 1px !important;
  min-width: 0 !important;
  padding: 0 !important;
  opacity: 0 !important;
  pointer-events: none !important;
}

:root {
  --light-bg:    #FAFAF9;
  --light-h:     #1A1A1A;
  --light-sub:   #8A8A8A;
  --light-body:  #6B6B6B;
  --dark-bg:     #141414;
  --dark-card:   #1C1C1C;
  --dark-h:      #F0F0F0;
  --dark-sub:    #707070;
  --dark-body:   #666666;
  --accent:      #9B8FD4;
  --accent-deep: #7B6FBF;
  --note-user:   #4A90D9;
  --note-ai:     #C0544A;
  --orb-grad: radial-gradient(ellipse at 30% 25%,
    #DCD7F0 0%,
    #C8C5E8 18%,
    #B5E0F0 38%,
    #7B9FD4 60%,
    #1A7A8A 82%,
    #0A4F5E 100%);
}

html, body, gradio-app {
  height: 100% !important;
  margin: 0 !important;
  overflow: hidden !important;
  background: var(--light-bg) !important;
  color: var(--light-h) !important;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
  -webkit-font-smoothing: antialiased !important;
  transition: background 0.3s ease !important;
}
/* Auto-switch the outer page background to match the visible panel */
html:has(.panel-s3:not(.hide)),
html:has(.panel-s4:not(.hide)),
html:has(.panel-s5:not(.hide)),
body:has(.panel-s3:not(.hide)),
body:has(.panel-s4:not(.hide)),
body:has(.panel-s5:not(.hide)),
gradio-app:has(.panel-s3:not(.hide)),
gradio-app:has(.panel-s4:not(.hide)),
gradio-app:has(.panel-s5:not(.hide)) {
  background: #0E0E0E !important;
}
.gradio-container {
  max-width: 1000px !important;
  width: 100% !important;
  height: 100vh !important;
  max-height: 100vh !important;
  margin: 0 auto !important;
  padding: 6px 10px 8px !important;
  overflow: hidden !important;
  box-sizing: border-box !important;
  background: transparent !important;
}
footer, .footer { display: none !important; }
#main, .app {
  height: 100% !important;
  overflow: hidden !important;
}

/* 건반 브리지: 화면 레이아웃에 안 보이게 (DOM·클릭은 유지) */
.note-input-layer {
  height: 0 !important;
  max-height: 0 !important;
  min-height: 0 !important;
  overflow: hidden !important;
  opacity: 0 !important;
  padding: 0 !important;
  margin: 0 !important;
  border: none !important;
  pointer-events: none !important;
}
.note-input-layer * { pointer-events: auto !important; }

/* 항상 visible인 컨테이너 — 자식 패널만 .hide 토글 (Gradio 6 중첩 Column 버그 회피) */
.game-stage {
  flex: 1 1 auto !important;
  min-height: 0 !important;
  display: flex !important;
  flex-direction: column !important;
  overflow: hidden !important;
}
.game-panel {
  flex: 1 1 auto !important;
  min-height: 0 !important;
  display: flex !important;
  flex-direction: column !important;
  justify-content: space-between !important;
  box-sizing: border-box !important;
  overflow: auto !important;
  padding: 16px !important;
  border-radius: 12px !important;
  position: relative !important;
}
/* Light screens: S1, S2 */
.panel-s1, .panel-s2 {
  background: var(--light-bg) !important;
  border: 1px solid rgba(0,0,0,0.07) !important;
  color: var(--light-h) !important;
}
/* S1 hero must never scroll — the orbit SVGs intentionally overflow the cosmos box */
.panel-s1 {
  overflow: hidden !important;
}
.panel-s1 .screen-body {
  overflow: hidden !important;
}
/* Dark screens: S3, S4, S5 */
.panel-s3, .panel-s4, .panel-s5 {
  background: var(--dark-bg) !important;
  border: 1px solid rgba(255,255,255,0.06) !important;
  color: var(--dark-h) !important;
}
.panel-s3 *, .panel-s4 *, .panel-s5 * { color: inherit; }
.game-stage > .hide {
  display: none !important;
}
.screen-body {
  flex: 1 1 auto !important;
  min-height: 0 !important;
  overflow: auto !important;
  display: flex !important;
  flex-direction: column !important;
  gap: 4px !important;
}
.screen-actions, .quick-notes {
  flex: 0 0 auto !important;
  flex-shrink: 0 !important;
  padding-top: 6px !important;
}
.screen-actions:last-of-type { margin-top: 0 !important; }
.screen-actions button:disabled:not(.mode-card) {
  display: none !important;
}
.mode-card:disabled {
  opacity: 0.45 !important;
  cursor: not-allowed !important;
  display: flex !important;
}
.quick-notes button { min-width: 1.8rem !important; padding: 3px 4px !important; font-size: 11px !important; }
.gradio-button, button {
  white-space: nowrap !important;
  font-family: 'Inter', -apple-system, sans-serif !important;
  font-size: 12px !important;
  font-weight: 400 !important;
  letter-spacing: 0.3px !important;
  border-radius: 20px !important;
  border: 1px solid rgba(0,0,0,0.12) !important;
  background: #FFFFFF !important;
  color: var(--light-h) !important;
  transition: background 0.15s ease, opacity 0.15s ease !important;
  box-shadow: none !important;
}
.gradio-button:hover, button:hover {
  opacity: 0.8 !important;
  box-shadow: none !important;
}
button.primary, .primary button, [data-testid="primary"] {
  background: var(--accent) !important;
  color: #FFFFFF !important;
  border: none !important;
}
button.primary:hover, .primary button:hover, [data-testid="primary"]:hover {
  background: var(--accent-deep) !important;
  opacity: 1 !important;
}
/* Dark panel buttons */
.panel-s3 button, .panel-s4 button, .panel-s5 button {
  background: var(--dark-card) !important;
  color: var(--dark-h) !important;
  border: 1px solid rgba(255,255,255,0.1) !important;
}
.panel-s3 button.primary, .panel-s4 button.primary, .panel-s5 button.primary,
.panel-s3 [data-testid="primary"], .panel-s4 [data-testid="primary"], .panel-s5 [data-testid="primary"] {
  background: var(--accent) !important;
  color: #FFFFFF !important;
  border: none !important;
}

/* S3: 고정 높이 레이아웃 — 건반 입력 시 스크롤·검정 깜빡임 방지 */
.panel-s3.game-panel {
  overflow: hidden !important;
}
.panel-s3 .screen-body.s3-main {
  overflow: hidden !important;
  min-height: 0 !important;
}
.s3-main {
  overflow: hidden !important;
  min-height: 0 !important;
}
.s3-main .viz-side { display: none !important; }
.s3-roll-row {
  justify-content: center !important;
  align-items: center !important;
  gap: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
}
.s3-main .piano-roll-host {
  /* figsize 9.8×2.95 — 표시 높이 222px에 맞춘 너비 (letterbox 방지) */
  --roll-h: 222px;
  --roll-w: calc(var(--roll-h) * 9.8 / 2.95);
  flex: 0 0 auto !important;
  width: var(--roll-w) !important;
  max-width: min(100%, var(--roll-w)) !important;
  margin: 0 auto !important;
  min-width: 0 !important;
  height: var(--roll-h) !important;
  min-height: var(--roll-h) !important;
  max-height: var(--roll-h) !important;
  overflow: hidden !important;
  padding: 0 !important;
  gap: 0 !important;
  background: #141414 !important;
  border: none !important;
  line-height: 0 !important;
}
.s3-main .piano-roll-host .block,
.s3-main .piano-roll-host .form,
.s3-main .piano-roll-host .column,
.s3-main .piano-roll-host .plot-container {
  width: 100% !important;
  height: var(--roll-h) !important;
  min-height: var(--roll-h) !important;
  max-height: var(--roll-h) !important;
  margin: 0 !important;
  padding: 0 !important;
  gap: 0 !important;
  background: #141414 !important;
  border: none !important;
  overflow: hidden !important;
  line-height: 0 !important;
}
.s3-main .piano-roll-host .plot-container > *,
.s3-main .piano-roll-host canvas,
.s3-main .piano-roll-host img,
.s3-main .piano-roll-host svg {
  width: 100% !important;
  height: var(--roll-h) !important;
  min-height: var(--roll-h) !important;
  max-height: var(--roll-h) !important;
  display: block !important;
  object-fit: fill !important;
  object-position: center top !important;
  margin: 0 !important;
  padding: 0 !important;
  vertical-align: top !important;
}
.s3-main .piano-roll-host .empty,
.s3-main .piano-roll-host label {
  display: none !important;
  height: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
}
.panel-s3 .ra-piano-wrap {
  flex: 0 0 auto !important;
}
.panel-s3 .ra-note-list-host {
  flex: 0 0 48px !important;
  min-height: 48px !important;
  max-height: 48px !important;
  overflow: hidden !important;
}
.panel-s3 .ra-note-list-host .ra-note-list,
.panel-s3 .ra-note-list-host .ra-note-empty {
  min-height: 40px !important;
  max-height: 40px !important;
  overflow: hidden !important;
  margin: 0 !important;
}
/* AI 생성/재생 안내 메시지 — 48px 노트리스트 영역에 맞춰 컴팩트하게 */
.panel-s3 .ra-note-list-host .ra-round-done {
  padding: 7px 14px !important;
  margin: 0 !important;
  font-size: 12.5px !important;
  max-height: 46px !important;
  box-sizing: border-box !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
}
/* 건반 입력마다 Gradio 블록 재렌더 시 깜빡임/검정 화면 방지 */
.panel-s3 .block, .panel-s3 .form,
.panel-s3 .gr-html, .panel-s3 .html-container {
  transition: none !important;
  animation: none !important;
}
.panel-s4 .piano-roll-host, .panel-s5 .piano-roll-host {
  width: calc(190px * 9.8 / 2.6) !important;
  max-width: min(100%, calc(190px * 9.8 / 2.6)) !important;
  margin: 0 auto !important;
  height: 190px !important;
  overflow: hidden !important;
  padding: 0 !important;
}
.panel-s4 .plot-container, .panel-s5 .plot-container,
.panel-s4 .piano-roll-host canvas, .panel-s5 .piano-roll-host canvas,
.panel-s4 .piano-roll-host img, .panel-s5 .piano-roll-host img {
  max-height: 190px !important;
  height: 190px !important;
  width: 100% !important;
  object-fit: fill !important;
  display: block !important;
}

/* S1 nav (amra-style) */
.s1-body { padding: 0 !important; }
.ra-nav {
  display: flex; align-items: center; justify-content: space-between;
  padding: 22px 28px 18px;
  border-bottom: none;
  color: var(--light-h);
  min-height: 40px;
}
.ra-nav-logo {
  font-family: 'Inter', -apple-system, sans-serif;
  font-size: 22px; font-weight: 500; color: var(--light-h);
  letter-spacing: -0.4px;
  line-height: 1;
  display: inline-flex; align-items: center;
}
.ra-nav-links {
  display: inline-flex; align-items: center; gap: 28px;
  font-size: 11px; letter-spacing: 1.4px;
  text-transform: uppercase; color: var(--light-h); font-weight: 500;
  line-height: 1;
}
.ra-nav-links span,
.ra-nav-links a.ra-nav-link {
  cursor: pointer;
  transition: opacity 0.15s ease;
  color: var(--light-h);
  text-decoration: none;
  line-height: 1;
  display: inline-flex; align-items: center;
}
.ra-nav-links .ra-nav-link:hover { opacity: 0.55; }
.ra-nav-cta {
  font-size: 10px; letter-spacing: 1.2px; text-transform: uppercase;
  padding: 6px 14px; border-radius: 16px;
  background: var(--accent); color: #fff;
  cursor: default;
}

/* S1 hero */
.ra-title-wrap {
  position: relative;
  display: flex; flex-direction: column; justify-content: flex-start; align-items: center;
  text-align: center; padding: 18px 24px 0;
  min-height: 0;
}
.ra-hero-head {
  font-size: clamp(20px, 3.2vw, 30px);
  font-weight: 400;
  color: var(--light-h);
  letter-spacing: -0.5px;
  line-height: 1.25;
  margin-top: 8px;
}
.ra-hero-sub {
  margin-top: 10px;
  font-size: 13px; font-weight: 300;
  color: var(--light-sub);
  line-height: 1.55;
  max-width: 480px;
}
.ra-hero-orb {
  width: min(380px, 60vh);
  height: min(380px, 60vh);
  border-radius: 50%;
  background: var(--orb-grad);
  margin: 8px auto 0;
  pointer-events: none;
  animation: ra-orb-float 6s ease-in-out infinite;
}
@keyframes ra-orb-float {
  0%, 100% { transform: translateY(0px); }
  50%      { transform: translateY(-6px); }
}

/* Hero cosmos — two SVG layers (outer slow CW + inner faster CCW) orbiting the planet.
   Rotation is driven by JS (mouse-distance based activity factor) for smooth parallax. */
.ra-hero-cosmos {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  width: min(380px, 60vh);
  aspect-ratio: 1;
  margin: 12px auto 0;
  cursor: default;
}
.ra-hero-cosmos .ra-hero-orb {
  position: relative;
  margin: 0 !important;
  width: 100% !important;
  height: 100% !important;
  z-index: 1;
}
.ra-orbit-outer,
.ra-orbit-inner {
  position: absolute;
  top: 50%; left: 50%;
  pointer-events: none;
  transform-origin: 50% 50%;
  transform: translate(-50%, -50%);
  will-change: transform;
}
.ra-orbit-outer { width: 122%;  height: 122%; z-index: 2; }
.ra-orbit-inner { width: 96%;   height: 96%;  z-index: 3; }
.ra-orbit-note  { transition: opacity 0.25s ease; }

/* S1 start buttons → mini-player pill style */
.s1-actions {
  justify-content: center !important;
  gap: 14px !important;
  padding-bottom: 18px !important;
}
.mode-card { min-height: 0 !important; }
.pill-cta {
  border-radius: 28px !important;
  padding: 8px 22px 8px 18px !important;
  display: inline-flex !important;
  align-items: center !important;
  gap: 12px !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  letter-spacing: -0.1px !important;
  flex: 0 0 auto !important;
  width: auto !important;
  min-width: 200px !important;
  height: 48px !important;
  justify-content: flex-start !important;
  box-shadow: 0 4px 20px rgba(0,0,0,0.08) !important;
  position: relative !important;
}
.s1-actions .pill-cta { min-width: 220px !important; }
.s2-actions .pill-cta { min-width: 260px !important; }
.s5-actions .pill-cta { min-width: 200px !important; }
/* music-note glyph instead of orb thumbnail */
.pill-cta::before {
  content: "♪\ufe0e";
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 22px; height: 32px;
  background: none;
  border-radius: 0;
  color: inherit;
  font-size: 26px;
  line-height: 1;
  font-weight: 500;
  flex: 0 0 22px;
  font-variant-emoji: text;
  text-rendering: optimizeLegibility;
  transform: translateY(-1px);
}
.pill-cta::after { display: none !important; }
.pill-cta-primary,
button.pill-cta.pill-cta-primary {
  background: #8B7FC8 !important;
  border: 1px solid transparent !important;
  color: #FFFFFF !important;
  box-shadow: 0 6px 22px rgba(139,127,200,0.32) !important;
}
.pill-cta-primary:hover,
button.pill-cta.pill-cta-primary:hover {
  background: #7B6FBF !important;
  opacity: 1 !important;
}
.pill-cta-ghost {
  background: transparent !important;
  border: 1px solid rgba(0,0,0,0.12) !important;
  color: var(--light-sub) !important;
  box-shadow: none !important;
}
.pill-cta-ghost-light {
  background: rgba(255,255,255,0.06) !important;
  border: 1px solid rgba(255,255,255,0.15) !important;
  color: var(--dark-h) !important;
  box-shadow: none !important;
}
.pill-cta-ghost:disabled { opacity: 0.4 !important; }

.ra-help-dot { color: var(--accent); margin-right: 6px; }

.ra-round-card, .ra-result-card, .ra-final-card {
  margin: 0 auto; width: min(96%, 640px);
  background: var(--dark-card);
  border: 1px solid rgba(255,255,255,0.07);
  border-radius: 12px;
  padding: 20px 24px;
}
.ra-round-label {
  display: flex; align-items: center; gap: 7px;
  font-size: 11px; font-weight: 400; letter-spacing: 2px;
  text-transform: uppercase; color: var(--accent); margin-bottom: 10px;
}
.ra-round-label::before {
  content: '●'; font-size: 7px; color: var(--accent);
}
.ra-round-title, .ra-result-title {
  margin: 0;
  color: var(--dark-h);
  font-size: 22px;
  font-weight: 400;
  letter-spacing: -0.3px;
}
.ra-round-meta { margin: 10px 0 6px; color: var(--dark-h); font-size: 15px; font-weight: 300; }
.ra-round-cond, .ra-result-feedback { color: var(--dark-sub); font-size: 13px; font-weight: 400; margin-top: 8px; line-height: 1.6; }

.ra-hud {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 8px;
  align-items: center;
  padding: 10px 14px;
  border-radius: 10px;
  border: 1px solid rgba(255,255,255,0.07);
  background: var(--dark-card);
}
.ra-hud-group { text-align: center; }
.ra-hud-round { font-family: 'Inter', sans-serif; font-size: 11px; font-weight: 500; letter-spacing: 1.5px; text-transform: uppercase; color: var(--accent); }
.ra-hud-exchange { margin-left: 10px; color: var(--dark-h); font-size: 12px; font-weight: 300; }
.ra-chip {
  display: inline-block; padding: 3px 8px; margin: 0 4px;
  border-radius: 12px; background: rgba(155,143,212,0.12); border: 1px solid rgba(155,143,212,0.25);
  color: var(--accent); font-size: 11px;
}
.ra-hud-score { font-size: 18px; font-weight: 400; color: var(--dark-h); }
.ra-hud-phase { font-weight: 400; text-align: center; font-size: 12px; color: var(--dark-sub); }

/* Turn badge */
.ra-turn-badge {
  font-family: 'Inter', sans-serif;
  font-size: 11px;
  font-weight: 400;
  letter-spacing: 1px;
  padding: 6px 14px;
  border-radius: 20px;
  text-align: center;
  text-transform: uppercase;
  animation: ra-turn-pop 0.35s ease-out;
}
.ra-turn-player {
  background: rgba(155,143,212,0.12);
  border: 1px solid rgba(155,143,212,0.35);
  color: var(--accent);
}
.ra-turn-ai {
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.15);
  color: var(--dark-h);
}
.ra-turn-result {
  background: rgba(155,143,212,0.08);
  border: 1px solid rgba(155,143,212,0.2);
  color: var(--dark-sub);
}
@keyframes ra-turn-pop {
  0%   { transform: translateY(4px); opacity: 0; }
  100% { transform: translateY(0); opacity: 1; }
}

/* S3 하단 액션 바 — Undo/Preview(좌) · Confirm 등(우), 겹침 방지 */
.panel-s3 .s3-actions-bar {
  display: flex !important;
  flex-direction: row !important;
  flex-wrap: wrap !important;
  align-items: center !important;
  justify-content: space-between !important;
  gap: 10px 14px !important;
  width: 100% !important;
  box-sizing: border-box !important;
}
.s3-actions-bar {
  justify-content: center !important;
  align-items: center !important;
  gap: 10px !important;
  flex-wrap: nowrap !important;
}
.s3-actions-bar > * {
  flex-shrink: 1 !important;
  min-width: 0 !important;
}
.s3-side-btn button {
  width: 100% !important;
  min-height: 40px !important;
  font-size: 12px !important;
}

/* 건반 단축키 레이블 */
.ra-key-shortcut {
  font-size: 9px;
  color: rgba(80,80,80,0.7);
  pointer-events: none;
  line-height: 1.2;
  text-align: center;
  margin-bottom: 2px;
  font-family: 'Inter', monospace;
  white-space: nowrap;
}
.ra-key-shortcut-black {
  color: rgba(220,220,220,0.7);
  font-size: 8px;
}
.ra-key-label-black {
  color: rgba(220,220,220,0.55) !important;
  font-size: 7px !important;
  margin-bottom: 1px;
}

/* Round done 메시지 */
.ra-round-done {
  text-align: center;
  font-size: 14px;
  font-weight: 300;
  color: var(--dark-h);
  padding: 14px;
  border: 1px solid rgba(155,143,212,0.25);
  border-radius: 10px;
  background: rgba(155,143,212,0.06);
  margin: 8px 0;
  animation: ra-turn-pop 0.35s ease-out;
}

/* S2 카드 크기 증가 */
.ra-round-card {
  padding: 28px 24px !important;
}
.ra-round-title { font-size: 28px !important; }
.ra-round-meta  { font-size: 17px !important; margin: 14px 0 10px !important; }
.ra-round-cond  { font-size: 14px !important; margin-top: 12px !important; }

/* S2 layout (Round info card with subtle orb) */
.s2-wrap {
  display: flex; flex-direction: column; align-items: center; text-align: center;
  padding: 24px 20px 0;
  position: relative;
}
.s2-headline {
  font-size: clamp(26px, 4vw, 38px);
  font-weight: 400; letter-spacing: -0.5px;
  color: var(--light-h);
  margin-top: 6px;
}
.s2-meta {
  font-size: 14px; font-weight: 300;
  color: var(--light-sub);
  letter-spacing: 1px; text-transform: uppercase;
  margin-top: 4px;
}
.s2-cond {
  margin-top: 14px;
  font-size: 13px; font-weight: 400;
  color: var(--light-body);
  max-width: 460px; line-height: 1.6;
}
.s2-orb {
  width: min(240px, 42vh) !important;
  height: min(240px, 42vh) !important;
  margin-top: 20px !important;
}
.s2-actions {
  justify-content: center !important;
  padding-bottom: 24px !important;
}

/* S5 — game over (image 5) */
.panel-s5 {
  padding: 0 !important;
  background: #0E0E0E !important;
  border: none !important;
  overflow: hidden !important;
  position: relative !important;
}
.panel-s5::after {
  content: '';
  position: absolute;
  left: 50%;
  bottom: -88%;
  transform: translateX(-50%);
  width: min(560px, 72%);
  aspect-ratio: 1 / 1;
  border-radius: 50%;
  background: radial-gradient(circle at 50% 32%,
    #F4F1FB 0%,
    #DCD7F0 12%,
    #C8C5E8 26%,
    #B5E0F0 42%,
    #7B9FD4 58%,
    rgba(26,122,138,0.55) 74%,
    rgba(13,92,106,0.0) 88%);
  pointer-events: none;
  z-index: 0;
  opacity: 0.95;
}
.panel-s5 > * { position: relative; z-index: 1; }
.s5-roll-hidden { display: none !important; }
.s5-body {
  position: relative !important;
  overflow-y: auto !important;
  padding: 0 !important;
}
.s5-stage {
  position: relative;
  display: flex; flex-direction: column; align-items: center;
  width: 100%; min-height: 100%;
  background: transparent;
  padding: 40px 32px 28px;
  box-sizing: border-box;
  text-shadow: 0 2px 18px rgba(14,14,14,0.55);
}
.s5-grade { text-shadow: 0 3px 22px rgba(14,14,14,0.7) !important; }
.s5-label {
  font-size: 11px;
  font-weight: 400;
  letter-spacing: 3px;
  text-transform: uppercase;
  color: var(--accent);
}
.s5-headline {
  margin-top: 20px;
  font-size: clamp(24px, 3.5vw, 34px);
  font-weight: 400;
  letter-spacing: -0.5px;
  line-height: 1.3;
  color: #F0F0F0;
  text-align: center;
}
.s5-grade {
  margin-top: 20px;
  font-size: 96px;
  font-weight: 300;
  letter-spacing: 8px;
  line-height: 1;
}
.s5-total {
  margin-top: 10px;
  font-size: 28px; font-weight: 400;
  color: var(--accent-deep);
}
.s5-of { color: var(--accent); }
.s5-halfsphere {
  position: absolute;
  left: 50%;
  bottom: -55%;
  transform: translateX(-50%);
  width: min(720px, 100%);
  height: min(720px, 100%);
  border-radius: 50%;
  background: linear-gradient(to bottom,
    #F4F1FB 0%,
    #DCD7F0 4%,
    #C8C5E8 7%,
    #B5E0F0 11%,
    #7B9FD4 14%,
    rgba(26,122,138,0.55) 17%,
    rgba(13,92,106,0.0) 22%);
  filter: blur(0.3px);
  pointer-events: none;
  z-index: 0;
}
.s5-stage > * { position: relative; z-index: 1; }

/* S5 v2 — 정보형 결과 레이아웃 */
.s5-stage-v2 { padding: 2px 34px 18px; justify-content: flex-start; }
.s5-stage-v2 .s5-grade { margin-top: 12px; font-size: 72px; }
.s5-stage-v2 .s5-total { margin-top: 6px; font-size: 24px; }
.s5-flavor { margin-top: 8px; font-size: 13px; color: var(--accent); letter-spacing: 0.3px; }
.s5-cols {
  display: flex; gap: 22px; width: 100%;
  max-width: 760px; margin: 22px auto 6px;
  text-align: left; flex-wrap: wrap;
}
.s5-col { flex: 1 1 320px; min-width: 260px; }
.s5-col-cap {
  font-size: 10.5px; font-weight: 600; letter-spacing: 1.4px;
  text-transform: uppercase; color: var(--accent);
  margin-bottom: 10px;
}
.s5-round-row {
  display: flex; align-items: center; gap: 10px;
  padding: 7px 0; border-bottom: 1px solid rgba(255,255,255,0.08);
  font-size: 12.5px; color: #E8E6F0;
}
.s5-round-rn { flex: 0 0 28px; font-weight: 600; color: var(--accent); }
.s5-round-cond {
  flex: 1 1 auto; color: #B8B4C8; font-size: 11.5px;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.s5-round-stars { flex: 0 0 auto; font-size: 11px; letter-spacing: -1px; }
.s5-round-score { flex: 0 0 48px; text-align: right; font-weight: 600; color: #F0EEFA; }
.s5-gauge { margin-bottom: 14px; }
.s5-gauge-head {
  display: flex; justify-content: space-between; align-items: baseline;
  font-size: 12.5px; font-weight: 500; color: #ECEAF4;
}
.s5-gauge-desc { font-size: 11px; color: #9D99AE; margin: 2px 0 5px; }
.s5-gauge-track {
  height: 6px; border-radius: 3px;
  background: rgba(255,255,255,0.1); overflow: hidden;
}
.s5-gauge-fill { height: 6px; border-radius: 3px; transition: width 0.6s ease; }
.s5-gauge-hint { font-size: 10.5px; color: #E8C26A; margin-top: 4px; }
.s5-roll-cap {
  width: 100%; max-width: 760px; margin: 14px auto 0;
  text-align: left; font-size: 10.5px; letter-spacing: 0.6px;
  color: #9D99AE;
}
.s5-roll-host { display: block !important; margin: 6px auto 0; max-width: 820px; width: 100%; }

.s5-actions {
  position: relative !important;
  z-index: 3 !important;
  padding: 16px 24px 30px !important;
  justify-content: center !important;
  background: transparent !important;
}
.s5-actions .game-pill {
  flex: 0 0 auto !important;
  width: auto !important;
  min-width: 0 !important;
  max-width: 240px !important;
  padding: 10px 26px 10px 20px !important;
}
.panel-s5 .ra-final-bonus {
  position: relative; z-index: 2;
  color: var(--accent); font-size: 13px;
  margin-top: 12px; letter-spacing: 0.5px;
}

.ra-piano-wrap {
  user-select: none; padding: 10px 8px; border-radius: 10px; text-align: center;
  background: var(--dark-card); border: 1px solid rgba(255,255,255,0.08);
}
.ra-piano-keys { display: inline-block; position: relative; }
.ra-key { transition: transform .08s ease, opacity .12s ease; }
.ra-key-white {
  background: #F4F4F2;
  border: 1px solid #C8C8C6;
  border-radius: 0 0 6px 6px;
}
.ra-key-black {
  background: #181818;
  border: 1px solid #000;
  box-shadow: inset 0 -6px 8px rgba(0,0,0,.4);
}
.ra-key.active {
  transform: translateY(2px) scale(0.99);
  background: var(--accent) !important;
  opacity: 0.85;
}
.ra-key-label { font-size: 8px; color: #8A8A8A; pointer-events: none; }

.ra-note-empty { color: var(--dark-sub); font-size: 12px; padding: 6px; }
.ra-note-list { display: flex; flex-wrap: wrap; padding: 4px; align-items: center; }
.ra-note-chip {
  background: rgba(155,143,212,0.1); border: 1px solid rgba(155,143,212,0.3);
  border-radius: 999px; padding: 2px 8px; margin: 2px; font-size: 12px; color: var(--accent);
}
.ra-note-count { color: var(--dark-sub); font-size: 11px; margin: 4px; }

.ra-score-row { margin: 6px 0; }
.ra-score-head { display: flex; justify-content: space-between; font-size: 12px; font-weight: 400; color: var(--dark-sub); margin-bottom: 4px; }
.ra-score-track { background: rgba(255,255,255,0.07); border-radius: 999px; height: 4px; border: none; }
.ra-result-grade {
  font-size: 28px; font-weight: 300; letter-spacing: 4px;
  margin: 12px 0 2px; color: var(--dark-h);
}
.ra-result-total {
  font-size: 16px; font-weight: 300;
  color: var(--dark-sub); margin-bottom: 14px;
  letter-spacing: 0.5px;
}
.ra-r5-bonus {
  color: var(--accent); font-size: 11px;
  letter-spacing: 2px; text-transform: uppercase;
  margin-top: 12px;
}
.ra-result-feedback {
  font-style: normal !important;
}
.ra-result-feedback::before {
  content: '— ';
  color: var(--accent);
}
.ra-round-done-cta {
  color: var(--accent);
  text-transform: uppercase;
  letter-spacing: 1.5px;
  font-size: 11px;
}

/* In-game pill button (S3/S4/S5) — white pill, black icon, dark text */
.game-pill {
  background: #FFFFFF !important;
  color: var(--light-h) !important;
  border: 1px solid rgba(0,0,0,0.06) !important;
  border-radius: 999px !important;
  padding: 10px 22px 10px 18px !important;
  font-size: 13.5px !important;
  font-weight: 500 !important;
  letter-spacing: 0.1px !important;
  box-shadow: 0 4px 18px rgba(0,0,0,0.10) !important;
  display: inline-flex !important;
  align-items: center !important;
  gap: 10px !important;
  min-width: 0 !important;
  width: auto !important;
  height: auto !important;
  min-height: 40px !important;
  white-space: nowrap !important;
  font-variant-emoji: text;
  text-rendering: optimizeLegibility;
  transition: background 0.15s ease, opacity 0.15s ease, box-shadow 0.15s ease !important;
}
.game-pill:hover:not(:disabled) {
  background: #FFFFFF !important;
  opacity: 0.92 !important;
  box-shadow: 0 6px 22px rgba(0,0,0,0.14) !important;
}
.game-pill:disabled {
  opacity: 0.4 !important;
  cursor: not-allowed !important;
}
.game-pill-sm {
  padding: 7px 16px 7px 12px !important;
  font-size: 12px !important;
  gap: 7px !important;
  min-height: 30px !important;
  box-shadow: 0 2px 10px rgba(0,0,0,0.08) !important;
}
/* Force white pill style even inside dark panels (overrides earlier dark-button rule) */
.panel-s3 .game-pill,
.panel-s4 .game-pill,
.panel-s5 .game-pill {
  background: #FFFFFF !important;
  color: var(--light-h) !important;
  border: 1px solid rgba(0,0,0,0.06) !important;
}
/* The leading character before the double-space acts as the icon — keep it boldly black */
.game-pill { color: #0E0E0E !important; }
/* Primary in-game pill — same white pill, just slightly heavier weight + softer shadow */
.game-pill-primary {
  font-weight: 600 !important;
  box-shadow: 0 6px 22px rgba(0,0,0,0.14) !important;
}
.game-pill-primary:hover:not(:disabled) {
  box-shadow: 0 8px 26px rgba(0,0,0,0.18) !important;
}
/* Confirm — 우측 그룹, 약 2배 크기 */
#btn-confirm {
  flex: 0 0 auto !important;
  min-width: 0 !important;
  width: auto !important;
}
#btn-confirm,
#btn-confirm button {
  padding: 14px 32px 14px 24px !important;
  font-size: 16px !important;
  font-weight: 600 !important;
  min-height: 52px !important;
  gap: 12px !important;
  width: auto !important;
  max-width: max-content !important;
  margin: 0 !important;
  box-shadow: 0 6px 24px rgba(0,0,0,0.14) !important;
}

/* Modals (S1 nav popups) */
.ra-modal {
  position: fixed; inset: 0;
  display: none;
  align-items: center; justify-content: center;
  background: rgba(20, 22, 32, 0.45);
  backdrop-filter: blur(6px);
  -webkit-backdrop-filter: blur(6px);
  z-index: 9999;
  padding: 24px;
  box-sizing: border-box;
  animation: ra-modal-fade 0.18s ease-out;
}
.ra-modal.open { display: flex; }
@keyframes ra-modal-fade {
  from { opacity: 0; }
  to   { opacity: 1; }
}
.ra-modal-card {
  width: min(560px, 100%);
  max-height: 88vh;
  overflow: auto;
  background: #FFFFFF;
  border-radius: 18px;
  padding: 28px 30px 26px;
  box-shadow: 0 24px 80px rgba(0,0,0,0.18);
  position: relative;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Apple SD Gothic Neo', 'Nanum Gothic', sans-serif;
  color: var(--light-h);
  animation: ra-modal-pop 0.22s cubic-bezier(0.2, 0.7, 0.3, 1);
}
@keyframes ra-modal-pop {
  from { transform: translateY(10px); opacity: 0; }
  to   { transform: translateY(0); opacity: 1; }
}
.ra-modal-head {
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 12px;
}
.ra-modal-eyebrow {
  font-size: 22px; font-weight: 500;
  letter-spacing: 0.4px; text-transform: uppercase;
  color: var(--accent);
}
.ra-modal-close {
  background: transparent !important;
  border: none !important;
  color: var(--light-sub) !important;
  font-size: 22px !important;
  line-height: 1 !important;
  padding: 4px 8px !important;
  margin: -4px -8px 0 0 !important;
  cursor: pointer !important;
  border-radius: 50% !important;
  width: auto !important; min-width: 0 !important; height: auto !important;
  box-shadow: none !important;
  transition: color 0.15s ease, background 0.15s ease !important;
}
.ra-modal-close:hover {
  color: var(--light-h) !important;
  background: rgba(0,0,0,0.04) !important;
  opacity: 1 !important;
}
.ra-modal-title { display: none; }
.ra-modal-body {
  font-size: 13.5px;
  line-height: 1.65;
  color: #2E2E36;
}
.ra-modal-body p { margin: 0 0 12px; }
.ra-modal-body b { color: var(--light-h); font-weight: 500; }
.ra-modal-list {
  margin: 6px 0 14px;
  padding-left: 20px;
}
.ra-modal-list li { margin-bottom: 8px; }
.ra-modal-list-bullets { list-style: none; padding-left: 0; }
.ra-modal-list-bullets li {
  position: relative;
  padding-left: 16px;
}
.ra-modal-list-bullets li::before {
  content: '●';
  position: absolute; left: 0; top: 0;
  color: var(--accent);
  font-size: 7px;
  line-height: 1.8;
}
.ra-modal-body kbd {
  display: inline-block;
  background: #F2F1ED;
  border: 1px solid rgba(0,0,0,0.08);
  border-radius: 4px;
  padding: 1px 6px;
  font-size: 11px;
  font-family: 'Inter', monospace;
  color: var(--light-h);
  margin: 0 1px;
}
.ra-modal-foot {
  margin-top: 14px;
  font-size: 12px;
  color: #565660;
  font-style: normal;
}
.ra-modal-locked { overflow: hidden !important; }

/* How to Play — 탭형 모달 */
.ra-howto-card { width: min(600px, 100%); }
.ra-howto-tabs {
  display: flex; gap: 6px;
  margin: 2px 0 14px;
  padding-bottom: 4px;
  border-bottom: 1px solid rgba(0,0,0,0.08);
}
.ra-howto-tab {
  appearance: none; background: transparent; border: none;
  padding: 8px 12px 10px; margin-bottom: -1px;
  font-family: inherit; font-size: 13px; font-weight: 500;
  color: var(--light-sub); cursor: pointer;
  border-bottom: 2px solid transparent;
  transition: color 0.15s ease, border-color 0.15s ease;
}
.ra-howto-tab:hover { color: var(--light-h); }
.ra-howto-tab.active { color: var(--accent-deep); border-bottom-color: var(--accent); }
.ra-howto-pane { display: none; animation: ra-modal-fade 0.18s ease-out; }
.ra-howto-pane.active { display: block; }
.ra-howto-h { margin: 4px 0 10px; font-size: 15px; font-weight: 600; color: var(--light-h); }
.ra-howto-sub { margin: 0 0 14px; font-size: 12.5px; color: #565660; }
.ra-howto-dim { color: #5A5560; font-size: 12.5px; }
/* Gradio 기본 prose 색이 자식(p/li/span)을 덮어써 연하게 보이는 문제 방지 —
   모달 본문 전체 글씨를 진하게 강제하고, 의도적으로 연한 보조 텍스트만 예외 처리 */
.ra-modal-card { background: #FFFFFF !important; }
.ra-modal-body,
.ra-modal-body p,
.ra-modal-body li,
.ra-modal-body span,
.ra-modal-body b,
.ra-modal-body div { color: #2A2A30 !important; }
/* 의도적으로 연한 보조 텍스트 예외 */
.ra-modal-body .ra-howto-dim { color: #5A5560 !important; }
.ra-modal-body .ra-howto-sub { color: #565660 !important; }
.ra-modal-body .ra-score-item-head span { color: #6A6A72 !important; }
.ra-modal-body .ra-grade-row span:last-child { color: #6A6A72 !important; }
.ra-modal-body .ra-grade-cap { color: var(--accent-deep) !important; }

/* 키보드 단축키 표 — 코드블록 스타일 */
.ra-key-table {
  margin-top: 14px; padding: 12px 14px;
  background: #F4F3EF; border: 1px solid rgba(0,0,0,0.07);
  border-radius: 10px; font-size: 12.5px;
}
.ra-key-row { display: flex; align-items: center; gap: 10px; padding: 3px 0; }
.ra-key-label {
  flex: 0 0 96px; color: #565660; font-size: 11.5px;
  letter-spacing: 0.3px;
}
.ra-key-row kbd {
  display: inline-block; background: #FFFFFF;
  border: 1px solid rgba(0,0,0,0.12); border-bottom-width: 2px;
  border-radius: 4px; padding: 1px 6px; margin: 0 2px;
  font-size: 11px; color: #2E2E36; font-family: 'Inter', monospace;
}

/* 점수 항목 — 좌측 색상 포인트 바 */
.ra-score-item {
  position: relative; padding: 8px 0 8px 14px; margin-bottom: 4px;
}
.ra-score-item::before {
  content: ''; position: absolute; left: 0; top: 8px; bottom: 8px;
  width: 4px; border-radius: 2px; background: var(--pt, var(--accent));
}
.ra-score-item-head {
  display: flex; align-items: baseline; justify-content: space-between;
  margin-bottom: 3px;
}
.ra-score-item-head b { color: var(--light-h); font-weight: 600; font-size: 13.5px; }
.ra-score-item-head span { color: #7A7A84; font-size: 11.5px; }
.ra-score-item p { margin: 0; font-size: 12.5px; line-height: 1.6; color: #3A3A42; }

/* 고득점 팁 */
.ra-tip-list { list-style: none; margin: 4px 0 0; padding: 0; }
.ra-tip-list li { display: flex; gap: 10px; margin-bottom: 12px; font-size: 12.5px; line-height: 1.55; color: #3A3A42; }
.ra-tip-ico { flex: 0 0 auto; font-size: 16px; }
.ra-tip-list b { color: var(--light-h); font-weight: 600; }

/* 등급 기준 표 */
.ra-grade-tables { display: flex; gap: 14px; margin-top: 18px; flex-wrap: wrap; }
.ra-grade-block { flex: 1 1 200px; }
.ra-grade-cap { font-size: 11px; font-weight: 600; letter-spacing: 0.4px; color: var(--accent-deep); margin-bottom: 6px; text-transform: uppercase; }
.ra-grade-row {
  display: flex; justify-content: space-between;
  padding: 4px 0; font-size: 12.5px; color: #3A3A42;
  border-bottom: 1px solid rgba(0,0,0,0.05);
}
.ra-grade-row span:last-child { color: #7A7A84; }
"""


# ─── Gradio app ──────────────────────────────────────────────────────────────

def _stop_audio():
    """AI 재생 종료 — 내 턴으로 돌아올 때 이전 오디오가 반복되지 않게."""
    return gr.update(value="")


def _screens(*visible_ids):
    """
    Python visible 토글은 DOM unmount → 하얀 화면 유발.
    패널 표시는 SHOW_SCREEN_JS(.hide)만 사용. visible_ids 는 state 기록용.
    """
    return tuple(gr.update() for _ in range(5))


def _nav(st: dict, screen: str) -> str:
    st["screen"] = screen
    return screen


def _phase(st: dict) -> str:
    return st.get("phase", "user_input")


def _result_panel_updates(st: dict):
    """S4/S5 결과 HTML — 화면 전환 직전/직후 채움."""
    sid = st.get("screen", "S3")
    if sid == "S4":
        return s4_html(st), gr.update(), gr.update()
    if sid == "S5":
        return gr.update(), s5_html(st), render_full_history_roll(st["round_results"])
    return gr.update(), gr.update(), gr.update()


_GR_MAJOR = int(gr.__version__.split(".")[0])
_blocks_kwargs = {"title": "RespondAI"}
if _GR_MAJOR < 6:
    _blocks_kwargs["css"] = APP_CSS
    _blocks_kwargs["js"] = KEYBOARD_JS

with gr.Blocks(**_blocks_kwargs) as app:
    state = gr.State(init_state())

    with gr.Column(visible=True, elem_classes=["game-stage"]):
        # ── S1: Title ────────────────────────────────────────────────────────
        with gr.Column(visible=True, elem_classes=["game-panel", "panel-s1"]) as screen_s1:
            with gr.Column(elem_classes=["screen-body", "s1-body"]):
                gr.HTML("""
<nav class="ra-nav">
  <div class="ra-nav-logo">respondai</div>
  <div class="ra-nav-links">
    <span class="ra-nav-link" onclick="window.respondAIModal && window.respondAIModal.open('howto')">HOW TO PLAY</span>
    <a class="ra-nav-link" href="https://github.com/yolemonade/respondai" target="_blank" rel="noopener noreferrer">GITHUB</a>
    <span class="ra-nav-link" onclick="window.respondAIModal && window.respondAIModal.open('devs')">DEVELOPERS</span>
  </div>
</nav>
<div class="ra-title-wrap">
  <div class="ra-hero-head">A call &amp; response with AI</div>
  <div class="ra-hero-sub">The only piano session built for<br/>spontaneous improvisation between you and a model.</div>
  <div class="ra-hero-cosmos">
    <svg class="ra-orbit-outer" viewBox="0 0 600 600" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
      <defs>
        <linearGradient id="orbitGradA" x1="0%" y1="20%" x2="100%" y2="80%">
          <stop offset="0%" stop-color="#7E6FBE" stop-opacity="0.95"/>
          <stop offset="55%" stop-color="#5A6FA6" stop-opacity="0.85"/>
          <stop offset="100%" stop-color="#2C6E80" stop-opacity="0.85"/>
        </linearGradient>
        <linearGradient id="orbitGradB" x1="100%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stop-color="#6E5FA8" stop-opacity="0.65"/>
          <stop offset="100%" stop-color="#3A6E80" stop-opacity="0.55"/>
        </linearGradient>
        <linearGradient id="orbitGradC" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="#5A5288" stop-opacity="0.55"/>
          <stop offset="100%" stop-color="#2C5E72" stop-opacity="0.4"/>
        </linearGradient>

        <!-- Reusable note shapes (head centered at 0,0; stems up/right) -->
        <symbol id="note-quarter" overflow="visible">
          <ellipse cx="0" cy="0" rx="7" ry="5" transform="rotate(-22)"/>
          <rect x="6" y="-30" width="1.8" height="30"/>
        </symbol>
        <symbol id="note-eighth" overflow="visible">
          <ellipse cx="0" cy="0" rx="6.6" ry="4.8" transform="rotate(-22)"/>
          <rect x="5.7" y="-30" width="1.7" height="30"/>
          <path d="M 7.4 -30 C 18 -24 20 -15 13.5 -8 C 16 -16 11.5 -22 7.4 -24 Z"/>
        </symbol>
        <symbol id="note-beamed" overflow="visible">
          <ellipse cx="-9" cy="2"  rx="6" ry="4.5" transform="rotate(-18)"/>
          <ellipse cx="9"  cy="-1" rx="6" ry="4.5" transform="rotate(-18)"/>
          <rect x="-3.8" y="-26" width="1.7" height="28"/>
          <rect x="14.2" y="-29" width="1.7" height="28"/>
          <path d="M -3.8 -26 L 15.9 -29 L 15.9 -23 L -3.8 -20 Z"/>
        </symbol>
        <symbol id="note-sixteenth" overflow="visible">
          <ellipse cx="0" cy="0" rx="6.6" ry="4.8" transform="rotate(-22)"/>
          <rect x="5.7" y="-32" width="1.7" height="32"/>
          <path d="M 7.4 -32 C 18 -26 20 -17 13.5 -10 C 16 -18 11.5 -24 7.4 -26 Z"/>
          <path d="M 7.4 -22 C 18 -16 20 -7  13.5 0   C 16 -8  11.5 -14 7.4 -16 Z"/>
        </symbol>
      </defs>

      <!-- Outer curves -->
      <path d="M 6 318 C 92 232, 188 392, 296 320 S 470 234, 596 308"
            stroke="url(#orbitGradA)" stroke-width="2.6" fill="none" stroke-linecap="round"/>
      <path d="M 70 168 Q 200 96, 320 168 T 552 196"
            stroke="url(#orbitGradB)" stroke-width="1.8" fill="none" stroke-linecap="round" opacity="0.75"/>

      <!-- Outer notes — bigger, further from orb -->
      <g class="ra-orbit-note" data-x="38"  data-y="318" data-scale="1.15" fill="#2E3147" transform="translate(38,318) scale(1.15)">
        <use href="#note-eighth"/>
      </g>
      <g class="ra-orbit-note" data-x="478" data-y="200" data-scale="1.05" fill="#3F4F7E" transform="translate(478,200) scale(1.05)">
        <use href="#note-beamed"/>
      </g>
      <g class="ra-orbit-note" data-x="554" data-y="298" data-scale="1.00" fill="#2E5E72" transform="translate(554,298) scale(1)">
        <use href="#note-sixteenth"/>
      </g>
      <g class="ra-orbit-note" data-x="128" data-y="470" data-scale="0.90" fill="#4A4F7A" transform="translate(128,470) scale(0.9)">
        <use href="#note-quarter"/>
      </g>
      <g class="ra-orbit-note" data-x="116" data-y="150" data-scale="0.78" fill="#5E5288" transform="translate(116,150) scale(0.78)">
        <use href="#note-eighth"/>
      </g>
    </svg>

    <svg class="ra-orbit-inner" viewBox="0 0 600 600" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
      <!-- Inner curves — smaller accent loops + lower wave -->
      <path d="M 92 270 q 30 -34 60 -16 q 30 18 16 50 q -16 32 -52 14"
            stroke="url(#orbitGradA)" stroke-width="1.6" fill="none" stroke-linecap="round" opacity="0.7"/>
      <path d="M 504 248 q 32 -10 40 18 q 8 28 -22 38 q -32 8 -38 -22"
            stroke="url(#orbitGradB)" stroke-width="1.6" fill="none" stroke-linecap="round" opacity="0.7"/>
      <path d="M 60 462 Q 220 526, 384 460 T 588 432"
            stroke="url(#orbitGradC)" stroke-width="1.4" fill="none" stroke-linecap="round" opacity="0.55"/>

      <!-- Inner notes — smaller, closer to orb -->
      <g class="ra-orbit-note" data-x="296" data-y="118" data-scale="0.72" fill="#524678" transform="translate(296,118) scale(0.72)">
        <use href="#note-eighth"/>
      </g>
      <g class="ra-orbit-note" data-x="492" data-y="372" data-scale="0.70" fill="#3F4F7E" transform="translate(492,372) scale(0.7)">
        <use href="#note-quarter"/>
      </g>
      <g class="ra-orbit-note" data-x="182" data-y="430" data-scale="0.68" fill="#2E5E72" transform="translate(182,430) scale(0.68)">
        <use href="#note-beamed"/>
      </g>
      <g class="ra-orbit-note" data-x="420" data-y="498" data-scale="0.65" fill="#4A4F7A" transform="translate(420,498) scale(0.65)">
        <use href="#note-sixteenth"/>
      </g>
    </svg>

    <div class="ra-hero-orb"></div>
  </div>
</div>

<!-- Modals (S1 only) -->
<div class="ra-modal" id="ra-modal-howto" onclick="if(event.target===this) window.respondAIModal.close()">
  <div class="ra-modal-card ra-howto-card" role="dialog" aria-labelledby="ra-modal-howto-title">
    <div class="ra-modal-head">
      <span class="ra-modal-eyebrow">● HOW TO PLAY</span>
      <button type="button" class="ra-modal-close" onclick="window.respondAIModal.close()" aria-label="Close">×</button>
    </div>
    <h2 id="ra-modal-howto-title" class="ra-modal-title">How to play</h2>

    <div class="ra-howto-tabs">
      <button type="button" class="ra-howto-tab active" data-tab="play" onclick="window.respondAIModal.howtoTab('play', this)">🎮 게임 방식</button>
      <button type="button" class="ra-howto-tab" data-tab="score" onclick="window.respondAIModal.howtoTab('score', this)">🎵 점수 기준</button>
      <button type="button" class="ra-howto-tab" data-tab="tips" onclick="window.respondAIModal.howtoTab('tips', this)">⭐ 고득점 팁</button>
    </div>

    <div class="ra-modal-body">
      <!-- PART 1 -->
      <div class="ra-howto-pane active" data-pane="play">
        <h3 class="ra-howto-h">어떻게 진행되나요?</h3>
        <ul class="ra-modal-list ra-modal-list-bullets">
          <li>총 5라운드, 라운드마다 AI와 멜로디를 주고받습니다.</li>
          <li>매 라운드는 3번의 교환으로 이루어집니다.<br/><span class="ra-howto-dim">→ 내가 먼저 연주 → AI가 응답 → 내가 다시 연주 (×3)</span></li>
          <li>가상 피아노 건반을 클릭하거나 키보드 단축키로 음을 입력하세요.</li>
          <li>입력을 마치면 <b>[확정]</b> 버튼 또는 <kbd>Enter</kbd>를 누르세요.</li>
          <li>최대 16개의 음을 입력할 수 있고, 최소 제한은 없습니다.</li>
          <li>마지막으로 입력한 음은 <kbd>Backspace</kbd>로 취소할 수 있습니다.</li>
        </ul>
        <div class="ra-key-table">
          <div class="ra-key-row"><span class="ra-key-label">흰 건반</span><span><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd><kbd>F</kbd><kbd>G</kbd><kbd>H</kbd><kbd>J</kbd><kbd>K</kbd> → C D E F G A B C</span></div>
          <div class="ra-key-row"><span class="ra-key-label">검은 건반</span><span><kbd>W</kbd><kbd>E</kbd><kbd>R</kbd><kbd>T</kbd><kbd>Y</kbd> → C# D# F# G# A#</span></div>
          <div class="ra-key-row"><span class="ra-key-label">옥타브 ↑</span><span><kbd>Shift</kbd> + <kbd>↑</kbd></span></div>
          <div class="ra-key-row"><span class="ra-key-label">옥타브 ↓</span><span><kbd>Shift</kbd> + <kbd>↓</kbd></span></div>
          <div class="ra-key-row"><span class="ra-key-label">마지막 음 취소</span><span><kbd>Backspace</kbd></span></div>
          <div class="ra-key-row"><span class="ra-key-label">확정</span><span><kbd>Enter</kbd></span></div>
        </div>
      </div>

      <!-- PART 2 -->
      <div class="ra-howto-pane" data-pane="score">
        <h3 class="ra-howto-h">무엇으로 점수를 받나요?</h3>
        <p class="ra-howto-sub">교환마다 최대 1000점. 아래 4가지 기준으로 평가합니다.</p>
        <div class="ra-score-item" style="--pt:#7B9FD4;">
          <div class="ra-score-item-head"><b>① 조성 일관성</b><span>최대 300점</span></div>
          <p>라운드마다 조성(Key)이 정해집니다. 그 조성의 스케일 안에 있는 음을 많이 쓸수록 높은 점수를 받습니다.<br/><span class="ra-howto-dim">예: D minor 라운드에서 D minor 스케일 음을 80% 쓰면 → 240점</span></p>
        </div>
        <div class="ra-score-item" style="--pt:#6FBF8F;">
          <div class="ra-score-item-head"><b>② 리듬 호응</b><span>최대 300점</span></div>
          <p>AI가 응답한 리듬 패턴을 얼마나 따라갔는지 평가합니다. 완전히 같을 필요는 없지만, 박자감이 비슷할수록 유리합니다.<br/><span class="ra-howto-dim">(첫 번째 교환은 AI 응답이 없으므로 자동 만점 처리)</span></p>
        </div>
        <div class="ra-score-item" style="--pt:#9B8FD4;">
          <div class="ra-score-item-head"><b>③ 모티프 활용</b><span>최대 300점</span></div>
          <p>1라운드 첫 번째 교환에서 내가 입력한 멜로디가 게임 전체의 ‘기준 모티프’가 됩니다. 이후 교환에서 그 멜로디의 앞부분(첫 3~4음)이 다시 등장할수록 점수를 받습니다.<br/><span class="ra-howto-dim">(1라운드 첫 교환은 자동 만점)</span></p>
        </div>
        <div class="ra-score-item" style="--pt:#E8C26A;">
          <div class="ra-score-item-head"><b>④ 창의성 보너스</b><span>최대 100점</span></div>
          <p>AI의 응답을 그대로 따라하지 않으면 +50점. 음정이나 리듬에 변형을 주면 추가 +50점.</p>
        </div>
        <div class="ra-score-item" style="--pt:#E89A6A;">
          <div class="ra-score-item-head"><b>⑤ R5 모티프 보너스</b><span>최대 150점 · 5라운드 한정</span></div>
          <p>마지막 라운드에서 기준 모티프를 50% 이상 활용하면 +150점 보너스!</p>
        </div>
      </div>

      <!-- PART 3 -->
      <div class="ra-howto-pane" data-pane="tips">
        <h3 class="ra-howto-h">잘 하려면 어떻게 해야 하나요?</h3>
        <ul class="ra-tip-list">
          <li><span class="ra-tip-ico">🎼</span><span><b>라운드 시작 전 Key를 확인하세요.</b><br/>"Key: D minor" 라면 레·미♭·파·솔·라·시♭·도 안에서 연주하면 유리합니다.</span></li>
          <li><span class="ra-tip-ico">🔁</span><span><b>1라운드 첫 연주가 가장 중요합니다.</b><br/>이 멜로디가 게임 전체의 기준이 됩니다. 기억하기 쉬운 짧은 패턴을 만들어보세요.</span></li>
          <li><span class="ra-tip-ico">👂</span><span><b>AI의 리듬을 잘 들어보세요.</b><br/>AI가 응답한 뒤, 그 박자감을 다음 입력에 반영하면 리듬 호응 점수가 올라갑니다.</span></li>
          <li><span class="ra-tip-ico">🎹</span><span><b>마지막 라운드에서 첫 멜로디로 돌아오세요.</b><br/>R5에서 처음 만든 모티프를 다시 쓰면 +150점 보너스를 받습니다.</span></li>
        </ul>
        <div class="ra-grade-tables">
          <div class="ra-grade-block">
            <div class="ra-grade-cap">라운드 등급</div>
            <div class="ra-grade-row"><span>⭐⭐⭐ PERFECT</span><span>950+</span></div>
            <div class="ra-grade-row"><span>⭐⭐ GREAT</span><span>800+</span></div>
            <div class="ra-grade-row"><span>⭐ CLEAR</span><span>600+</span></div>
            <div class="ra-grade-row"><span>TRY AGAIN</span><span>&lt; 600</span></div>
          </div>
          <div class="ra-grade-block">
            <div class="ra-grade-cap">최종 등급 (5000점 만점)</div>
            <div class="ra-grade-row"><span>S</span><span>4500+</span></div>
            <div class="ra-grade-row"><span>A</span><span>3500+</span></div>
            <div class="ra-grade-row"><span>B</span><span>2500+</span></div>
            <div class="ra-grade-row"><span>C</span><span>&lt; 2500</span></div>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>

<div class="ra-modal" id="ra-modal-devs" onclick="if(event.target===this) window.respondAIModal.close()">
  <div class="ra-modal-card" role="dialog" aria-labelledby="ra-modal-devs-title">
    <div class="ra-modal-head">
      <span class="ra-modal-eyebrow">● DEVELOPERS</span>
      <button type="button" class="ra-modal-close" onclick="window.respondAIModal.close()" aria-label="Close">×</button>
    </div>
    <h2 id="ra-modal-devs-title" class="ra-modal-title">Developers</h2>
    <div class="ra-modal-body">
      <p>RespondAI는 음악 AI와 사람의 즉흥 인터랙션을 탐구하는 <b>Deep Learning for Music and Audio</b> final project입니다.</p>
      <ul class="ra-modal-list ra-modal-list-bullets">
        <li><b>박시현 (Team A) — 모델 &amp; 분석</b> · 데이터 전처리, Transformer 기반 Call &amp; Response 생성 모델, 응답 점수화 로직.</li>
        <li><b>강유영 (Team B) — 프론트엔드 &amp; UX</b> · Gradio 기반 인터랙션, 가상 피아노 입력, 라운드 흐름과 결과 화면 설계.</li>
      </ul>
      <p class="ra-modal-foot">코드와 자세한 문서는 상단의 GitHub 링크에서 확인하세요.</p>
    </div>
  </div>
</div>
""")
            with gr.Row(equal_height=True, elem_classes=["screen-actions", "s1-actions"]):
                btn_piano_start  = gr.Button("Start session", variant="primary", scale=0,
                                             elem_classes=["mode-card", "pill-cta", "pill-cta-primary"],
                                             elem_id="btn-piano-start")
                btn_humming_start = gr.Button("🎤 Humming mode (Beta)", variant="secondary", scale=0,
                                              elem_classes=["mode-card", "pill-cta", "pill-cta-ghost"],
                                              elem_id="btn-humming-start")

        # ── S2: Round start ──────────────────────────────────────────────────
        with gr.Column(visible=True, elem_classes=["game-panel", "panel-s2", "hide"]) as screen_s2:
            with gr.Column(elem_classes=["screen-body"]):
                s2_info = gr.HTML()
            with gr.Row(elem_classes=["screen-actions", "s2-actions"]):
                btn_round_start = gr.Button("Begin round", variant="primary",
                                            elem_classes=["pill-cta", "pill-cta-primary"])

        # ── S3: Main game ────────────────────────────────────────────────────
        with gr.Column(visible=True, elem_classes=["game-panel", "panel-s3", "hide"]) as screen_s3:
            with gr.Column(elem_classes=["screen-body", "s3-main"]):
                s3_hud_html  = gr.HTML()
                with gr.Row(elem_classes=["s3-roll-row"]):
                    s3_viz_player = gr.HTML(render_energy_svg([], "player"), elem_classes=["viz-side"])
                    with gr.Column(elem_classes=["piano-roll-host"]):
                        s3_roll = gr.Plot(show_label=False)
                    s3_viz_ai = gr.HTML(render_energy_svg([], "ai"), elem_classes=["viz-side"])
                with gr.Column(elem_id="s3-piano-host") as s3_piano_host:
                    s3_piano = gr.HTML(render_piano_html(4))

                    with gr.Row(elem_classes=["rhythm-input-controls"]):
                        note_duration_select = gr.Radio(
                            choices=[
                                ("1/16", 1),
                                ("1/8", 2),
                                ("Dotted 1/8", 3),
                                ("1/4", 4),
                                ("Dotted 1/4", 6),
                                ("1/2", 8),
                            ],
                            value=2,
                            label="Note length",
                            scale=4,
                        )

                        btn_rest = gr.Button(
                            "Rest",
                            scale=1,
                            elem_classes=["game-pill", "game-pill-sm"],
                        )
                with gr.Column(visible=False, elem_id="s3-mic-host",
                               elem_classes=["s3-mic-host"]) as s3_mic_host:
                    s3_mic = gr.Audio(
                        sources=["microphone"], type="numpy", format="wav",
                        label="🎤 Press record, hum, then press Enter (or Stop)",
                        show_label=True, elem_id="s3-mic-input",
                    )
                s3_note_list = gr.HTML(note_list_html([]), elem_classes=["ra-note-list-host"])
                s3_audio     = gr.HTML("", elem_id="s3-exchange-audio")
            with gr.Row(elem_classes=["screen-actions", "s3-actions-bar"]):
                btn_cancel      = gr.Button("↺  Undo", scale=1,
                                            elem_classes=["game-pill", "game-pill-sm", "s3-side-btn"])
                btn_confirm     = gr.Button("★\ufe0e  Confirm  (Enter)", scale=3,
                                            elem_id="btn-confirm",
                                            elem_classes=["game-pill", "game-pill-primary"])
                btn_preview     = gr.Button("♪\ufe0e  Preview", scale=1,
                                            elem_classes=["game-pill", "game-pill-sm", "s3-side-btn"])
                btn_next_inline    = gr.Button("☁\ufe0e  See result", scale=2,
                                               interactive=False, elem_id="btn-next-inline",
                                               elem_classes=["game-pill"])
                btn_restart_inline = gr.Button("↻  Play again", scale=2,
                                               interactive=False, elem_id="btn-restart-inline",
                                               elem_classes=["game-pill"])

        # ── S4: Round result ─────────────────────────────────────────────────
        with gr.Column(visible=True, elem_classes=["game-panel", "panel-s4", "hide"]) as screen_s4:
            with gr.Column(elem_classes=["screen-body"]):
                s4_result_html = gr.HTML()
            with gr.Row(elem_classes=["screen-actions"]):
                btn_next_round = gr.Button("♥\ufe0e  Next round", scale=2,
                                           elem_classes=["game-pill", "game-pill-primary"])

        # ── S5: Final result ─────────────────────────────────────────────────
        with gr.Column(visible=True, elem_classes=["game-panel", "panel-s5", "hide"]) as screen_s5:
            with gr.Column(elem_classes=["screen-body", "s5-body"]):
                s5_result_html = gr.HTML()
                with gr.Column(elem_classes=["piano-roll-host", "s5-roll-host"]):
                    s5_roll = gr.Plot(show_label=False)
            with gr.Row(elem_classes=["screen-actions", "s5-actions"]):
                btn_restart = gr.Button("↻  Play again", scale=0,
                                        elem_classes=["game-pill", "game-pill-primary"])

        screen_nav = gr.Textbox(value="S1", visible=False, elem_id="ra-screen-nav")
        phase_nav  = gr.Textbox(value="user_input", visible=False, elem_id="ra-phase-nav")
        exchange_nav = gr.Textbox(value="0", visible=False, elem_id="ra-exchange-nav")
        # 브라우저 <audio>의 onended 가 클릭하는 숨김 버튼 — '재생 끝' 이벤트 브리지.
        # visible=False 는 DOM에서 빠질 수 있어 CSS(.ra-hidden-btn)로 숨긴다.
        btn_audio_done = gr.Button("", elem_id="btn-audio-done",
                                   elem_classes=["ra-hidden-btn"])

    # 건반 브리지 (Blocks 맨 끝 + 화면 밖 배치)
    _note_names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    with gr.Row(elem_classes=["note-input-layer"]):
        btn_undo = gr.Button("⌫", elem_id="btn-undo")
        _note_btn_map = []
        for midi in KB_ALLOWED_MIDIS:
            label = _note_names[midi % 12]
            _btn = gr.Button(label, elem_id=f"nk-{midi}")
            _note_btn_map.append((midi, _btn))

    # ─────────────────────────────────────────────────────────────────────────
    # Callbacks
    # ─────────────────────────────────────────────────────────────────────────

    def _s2_info_html(st: dict) -> str:
        rnd = st["round"]
        cond = ROUND_CONDITIONS[rnd]
        return f"""
<div class="s2-wrap">
  <div class="ra-round-label">● ROUND {rnd} OF {TOTAL_ROUNDS}</div>
  <div class="s2-headline">{st['key']}</div>
  <div class="s2-meta">{st['bpm']} BPM</div>
  <div class="s2-cond">{cond}</div>
  <div class="ra-hero-orb s2-orb"></div>
</div>
"""

    # S1 → S2
    def on_piano_start(st):
        st = init_state()
        st["mode"] = "piano"
        st["key"]  = random.choice(KEYS)
        st["bpm"]  = random.choice(BPM_CHOICES)
        return st, _s2_info_html(st), _nav(st, "S2")

    btn_piano_start.click(
        on_piano_start, inputs=[state],
        outputs=[state, s2_info, screen_nav],
    ).then(fn=None, js=SHOW_SCREEN_JS, inputs=[screen_nav])

    def on_humming_start(st):
        st = init_state()
        st["mode"] = "humming"
        st["key"]  = random.choice(KEYS)
        st["bpm"]  = random.choice(BPM_CHOICES)
        return st, _s2_info_html(st), _nav(st, "S2")

    btn_humming_start.click(
        on_humming_start, inputs=[state],
        outputs=[state, s2_info, screen_nav],
    ).then(fn=None, js=SHOW_SCREEN_JS, inputs=[screen_nav])

    # S2 → S3 — 상태·S3 UI·화면 전환을 한 번에 (분리 시 Gradio 6 하얀 화면)
    def on_round_start(st):
        st["exchange"]      = 1
        st["phase"]         = "user_input"
        st["screen"]        = "S3"
        reset_note_input(st)
        st["ai_notes"]      = []
        st["exchange_log"]  = []
        is_humming = st["mode"] == "humming"
        return (
            st,
            hud_html(st),
            render_piano_roll(st),
            render_energy_svg([], "player"),
            render_energy_svg([], "ai"),
            note_list_html([]),
            render_piano_html(4),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=False),
            gr.update(interactive=False),
            _stop_audio(),
            gr.update(visible=not is_humming),   # s3_piano_host
            gr.update(visible=is_humming),        # s3_mic_host
            gr.update(value=None),                # s3_mic — 이전 녹음 비우기
            _nav(st, "S3"),
            _phase(st),
        )

    round_start_chain = btn_round_start.click(
        on_round_start, inputs=[state],
        outputs=[
            state, s3_hud_html, s3_roll, s3_viz_player, s3_viz_ai, s3_note_list,
            s3_piano, btn_confirm, btn_cancel, btn_preview, btn_next_inline,
            btn_restart_inline, s3_audio,
            s3_piano_host, s3_mic_host, s3_mic,
            screen_nav, phase_nav,
        ],
    )
    round_start_chain.then(fn=None, js=SHOW_SCREEN_JS, inputs=[screen_nav])
    round_start_chain.then(fn=None, js=FOCUS_GAME_JS)

    def on_note_event(midi: int, st: dict):
        """cursor 기반 건반 입력.

        note와 rest를 같은 input_events에 저장하므로 Undo가 둘 다 정확히
        취소되고, 실제 start/end 리듬이 모델 입력에 전달된다.
        """
        if st["phase"] != "user_input":
            return st, note_list_html(st["current_notes"]), render_piano_roll(st)

        events = list(st.get("input_events", []))
        duration = max(1, int(st.get("note_duration", DEFAULT_DURATION)))
        cursor = max(0, int(st.get("cursor_step", 0)))

        if midi == -1:
            if events:
                events.pop()
            st["input_events"] = events
            st["cursor_step"] = _event_cursor(events)
            st["current_notes"] = _notes_from_input_events(events)
            return st, note_list_html(st["current_notes"]), render_piano_roll(st)

        if not 21 <= midi <= 108:
            return st, note_list_html(st["current_notes"]), render_piano_roll(st)

        if len(st["current_notes"]) >= MAX_NOTES or cursor >= INPUT_MAX_STEPS:
            return st, note_list_html(st["current_notes"]), render_piano_roll(st)

        end = min(cursor + duration, INPUT_MAX_STEPS)
        if end <= cursor:
            return st, note_list_html(st["current_notes"]), render_piano_roll(st)

        events.append({
            "kind": "note",
            "pitch": int(midi),
            "start": cursor,
            "end": end,
        })
        st["input_events"] = events
        st["cursor_step"] = end
        st["current_notes"] = _notes_from_input_events(events)

        return st, note_list_html(st["current_notes"]), render_piano_roll(st)


    def on_duration_change(value, st):
        """다음에 입력할 음/쉼표의 길이를 변경한다."""
        try:
            st["note_duration"] = max(1, int(value))
        except (TypeError, ValueError):
            st["note_duration"] = DEFAULT_DURATION
        return st


    def on_rest(st):
        """현재 duration만큼 쉼표를 입력하고 cursor를 전진시킨다."""
        if st["phase"] != "user_input":
            return st, note_list_html(st["current_notes"]), render_piano_roll(st)

        events = list(st.get("input_events", []))
        duration = max(1, int(st.get("note_duration", DEFAULT_DURATION)))
        cursor = max(0, int(st.get("cursor_step", 0)))

        if cursor < INPUT_MAX_STEPS:
            end = min(cursor + duration, INPUT_MAX_STEPS)
            if end > cursor:
                events.append({
                    "kind": "rest",
                    "start": cursor,
                    "end": end,
                })
                st["input_events"] = events
                st["cursor_step"] = end
                st["current_notes"] = _notes_from_input_events(events)

        return st, note_list_html(st["current_notes"]), render_piano_roll(st)


    _note_outputs = [state, s3_note_list, s3_roll]
    _note_click_kw = dict(show_progress="hidden")

    note_duration_select.change(
        on_duration_change,
        inputs=[note_duration_select, state],
        outputs=[state],
        show_progress="hidden",
    )

    btn_rest.click(
        on_rest,
        inputs=[state],
        outputs=_note_outputs,
        **_note_click_kw,
    )

    btn_undo.click(
        lambda st: on_note_event(-1, st),
        inputs=[state],
        outputs=_note_outputs,
        **_note_click_kw,
    )

    for midi, btn in _note_btn_map:
        btn.click(
            lambda st, m=midi: on_note_event(m, st),
            inputs=[state],
            outputs=_note_outputs,
            **_note_click_kw,
        )

    btn_cancel.click(
        lambda st: on_note_event(-1, st),
        inputs=[state],
        outputs=_note_outputs,
        **_note_click_kw,
    )

    # 허밍 녹음 → PESTO 음 인식 (피아노 클릭과 동일하게 current_notes 채움)
    def on_humming_record(audio, st):
        if st["phase"] != "user_input":
            return st, note_list_html(st["current_notes"]), render_piano_roll(st), gr.update()
        try:
            notes = humming_to_notes(audio, bpm=st["bpm"])
        except Exception as exc:           # PESTO/오디오 오류 시 게임 흐름 유지
            print(f"[humming] recognition failed: {exc}")
            notes = []
        st["current_notes"] = notes
        st["input_events"] = [
            {
                "kind": "note",
                "pitch": int(note.pitch),
                "start": int(note.start),
                "end": int(note.end),
            }
            for note in notes
        ]
        st["cursor_step"] = max((note.end for note in notes), default=0)
        return (
            st,
            note_list_html(st["current_notes"]),
            render_piano_roll(st),
            render_energy_svg(st["current_notes"], "player"),
        )

    # 녹음 시작 → JS 플래그 (Enter 자동정지 판단용)
    s3_mic.start_recording(fn=None, js=MIC_START_JS)
    # 정지 → PESTO 인식(프로세싱 표시 숨김) → Enter 로 정지했다면 자동 전송
    s3_mic.stop_recording(
        on_humming_record, inputs=[s3_mic, state],
        outputs=[state, s3_note_list, s3_roll, s3_viz_player],
        show_progress="hidden",
    ).then(fn=None, js=MIC_AFTER_STOP_JS)

    def on_preview(st):
        yield gr.update(value="")

        audio = build_round_audio(
            st["current_notes"],
            [],
            st["bpm"],
            swing_amount=st.get(
                "swing_amount",
                0.60,
            ),
        )

        yield gr.update(
            value=audio_to_html(audio)
        )

    btn_preview.click(on_preview, inputs=[state], outputs=[s3_audio])

    _confirm_outputs = [
        state, s3_roll, s3_note_list, s3_hud_html, s3_viz_player, s3_viz_ai, s3_piano,
        btn_confirm, btn_cancel, btn_preview, btn_next_inline, btn_restart_inline, s3_audio,
        s4_result_html, s5_result_html, s5_roll, phase_nav, exchange_nav,
        s3_mic,   # 교환마다 마이크 비워 새 녹음 시작 (허밍 모드)
    ]
    _CONFIRM_NOOP = (gr.update(),) * 15
    _LOCKED_BTNS = (
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
    )
    _IDLE_BTNS = (
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
    )
    _confirm_tail = (gr.update(), gr.update(), gr.update())

    def _finalize_round(st: dict) -> None:
        exs = st["exchange_log"]
        exchange_scores = [e["score"] for e in exs]
        round_total = int(sum(s["total"] for s in exchange_scores) / len(exchange_scores))
        r5_bonus = False
        if st["round"] == TOTAL_ROUNDS and st["r1_motif"]:
            last_motif_raw = exchange_scores[-1]["raw"]["motif_overlap"]
            r5_bonus = last_motif_raw >= 0.5
        if r5_bonus:
            round_total += 150
        st["round_results"] = st["round_results"] + [{
            "round_num": st["round"],
            "key": st["key"],
            "bpm": st["bpm"],
            "total": round_total,
            "exchange_scores": exchange_scores,
            "exchange_log": list(exs),
            "r5_motif_bonus": r5_bonus,
        }]
        st["total_score"] += round_total
        st["phase"] = "round_result"
        st["screen"] = "S4"

    def on_confirm(st):
        """AI 생성 → 재생 yield → 대기 → YOUR TURN (한 제너레이터에서 state 동기화)."""
        if st["phase"] != "user_input":
            yield (st,) + _CONFIRM_NOOP + (_phase(st), str(len(st["exchange_log"])), gr.update())
            return

        _T = {}
        _T["start"] = time.time()
        
        user_notes = list(st["current_notes"])
        key_token = KEY_TO_TOKEN[st["key"]]
        exchange_num = len(st["exchange_log"]) + 1
        st["exchange"] = exchange_num
        st["phase"] = "ai_response"

        thinking_msg = (
            '<div class="ra-round-done">'
            '<span class="ra-help-dot">●</span> AI is composing a response…'
            '</div>'
        )
        _T["before_yield1"] = time.time()
        yield (
            st,
            render_piano_roll(st), thinking_msg, hud_html(st),
            render_energy_svg([], "player"),
            render_energy_svg([], "ai"),
            gr.update(),
            *_LOCKED_BTNS,
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(),
            *_confirm_tail,
            "ai_response",
            str(len(st["exchange_log"])),
            gr.update(),                          # s3_mic (no-op)
        )
        _T["after_yield1"] = time.time()
        print(f"[TIMING] yield1(thinking) render: {_T['after_yield1'] - _T['before_yield1']:.3f}s")


        _T["before_generate"] = time.time()
        ai_notes, result = generate_ai_notes(
            user_notes,
            key_token,
            st["bpm"],
            previous_ai_notes=st.get("ai_notes") or [],
        )
        _T["after_generate"] = time.time()
        print(f"[TIMING] generate(): {_T['after_generate'] - _T['before_generate']:.3f}s")
        attn_scores = align_attn_to_notes(result.attn_scores if result else [], ai_notes)

        _T["before_score"] = time.time()
        score = compute_exchange_score(
            user_notes, st["ai_notes"], st["r1_motif"],
            st["round"], exchange_num, key_token,
        )
        _T["after_score"] = time.time()

        if st["round"] == 1 and exchange_num == 1 and user_notes:
            st["r1_motif"] = user_notes

        st["exchange_log"] = st["exchange_log"] + [{
            "user_notes": user_notes,
            "ai_notes": ai_notes,
            "score": score,
            "attn_scores": attn_scores,
        }]
        st["ai_notes"] = ai_notes
        reset_note_input(st)

        _T["before_audio"] = time.time()
        audio = build_round_audio(
            user_notes,
            ai_notes,
            st["bpm"],
            swing_amount=st.get("swing_amount", 0.60),
            key_token=key_token,
            round_num=st["round"],
            exchange_num=exchange_num,
        )
        _T["after_audio"] = time.time()
        print(f"[TIMING] build_round_audio(): {_T['after_audio'] - _T['before_audio']:.3f}s")
        playback_ms = max(
            estimate_playback_ms(user_notes, ai_notes, st["bpm"]),
            audio_duration_ms(audio),
        )
        playing_msg = (
            '<div class="ra-round-done">'
            '<span class="ra-help-dot">●</span> Playing AI response…'
            '</div>'
        )
        # 재생 단계로 전환 — 이후 진행은 브라우저 onended → on_audio_done 이 담당.
        st["phase"] = "ai_playing"
        st["last_attn_scores"] = attn_scores

        _T["before_yield2"] = time.time()
        yield (
            st,
            render_piano_roll(st), playing_msg, hud_html(st),
            render_energy_svg([], "player"),
            render_energy_svg(st["ai_notes"], "ai", attn_scores),
            gr.update(),
            *_LOCKED_BTNS,
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(value=audio_to_html(audio, notify_done=True,
                                          playback_ms=playback_ms)),
            *_confirm_tail,
            "ai_playing",
            str(len(st["exchange_log"])),
            gr.update(),                          # s3_mic (no-op)
        )
        _T["after_yield2"] = time.time()
        print(f"[TIMING] yield2(playing) render: {_T['after_yield2'] - _T['before_yield2']:.3f}s")


        # 마지막 교환(3번째 AI 응답)은 AI 소리를 끝까지 들은 뒤 결과 화면(S4)으로,
        # 마지막 교환(3번째 AI 응답)은 AI 소리를 끝까지 들은 뒤 결과 화면(S4)으로,
        # 그 외 교환도 재생이 끝난 뒤 다음 입력(YOUR TURN)으로.
        # → 진행 타이밍은 서버 sleep 추측이 아니라 브라우저 <audio> 의 onended
        #   이벤트(btn_audio_done 클릭)가 결정한다. on_confirm 은 여기서 끝.
        _T["end"] = time.time()
        print(f"[TIMING] ─────────────────────────────")
        print(f"[TIMING] 전체 on_confirm 서버 처리: {_T['end'] - _T['start']:.3f}s")
        print(f"[TIMING]   yield1 render  : {_T['after_yield1'] - _T['before_yield1']:.3f}s")
        print(f"[TIMING]   generate()     : {_T['after_generate'] - _T['before_generate']:.3f}s")
        print(f"[TIMING]   score()        : {_T['after_score'] - _T['before_score']:.3f}s")
        print(f"[TIMING]   audio()        : {_T['after_audio'] - _T['before_audio']:.3f}s")
        print(f"[TIMING]   yield2 render  : {_T['after_yield2'] - _T['before_yield2']:.3f}s")
        print(f"[TIMING] ─────────────────────────────")

    def on_audio_done(st):
        """브라우저 오디오 재생 종료(onended/워치독) 시 다음 단계로 진행.

        on_confirm 의 옛 후반부(sleep 이후)를 그대로 옮긴 것. phase 가드로
        onended + 워치독 중복 클릭을 멱등 처리한다.
        """
        if st.get("phase") != "ai_playing":
            return (st,) + _CONFIRM_NOOP + (
                _phase(st), str(len(st["exchange_log"])), gr.update())

        attn_scores = st.get("last_attn_scores", [])
        completed = len(st["exchange_log"])
        if completed < MAX_EXCHANGES:
            st["exchange"] = completed + 1
            st["phase"] = "user_input"
            st["screen"] = "S3"
            return (
                st,
                render_piano_roll(st), note_list_html([]), hud_html(st),
                render_energy_svg([], "player"),
                render_energy_svg(st["ai_notes"], "ai", attn_scores),
                gr.update(value=render_piano_html(4)),
                *_IDLE_BTNS,
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(value=""),              # 오디오 엘리먼트 제거 (재생 정지)
                *_confirm_tail,
                "user_input",
                str(completed),
                gr.update(value=None),            # s3_mic 비우기 → 다음 교환 새 녹음
            )
        _finalize_round(st)
        s4, s5, roll = _result_panel_updates(st)
        return (
            st,
            render_piano_roll(st), note_list_html([]), hud_html(st),
            render_energy_svg([], "player"),
            render_energy_svg(st["ai_notes"], "ai", attn_scores),
            gr.update(value=""),
            *_LOCKED_BTNS,
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(value=""),                  # 오디오 엘리먼트 제거
            s4, s5, roll,
            _phase(st),
            str(completed),
            gr.update(value=None),                # s3_mic 비우기 (라운드 종료)
        )
    COMBINED_JS = """(screenId, phase) => {
      const id = String(screenId || 'S1').trim().toUpperCase();
      const target = 'panel-' + id.toLowerCase();
      const roots = [document];
      document.querySelectorAll('gradio-app, .gradio-container').forEach((h) => {
        if (h.shadowRoot) roots.push(h.shadowRoot);
      });
      for (const root of roots) {
        root.querySelectorAll('.game-stage .game-panel, .game-panel.panel-s1, .game-panel.panel-s2, .game-panel.panel-s3, .game-panel.panel-s4, .game-panel.panel-s5').forEach((el) => {
          if (el.classList.contains(target)) el.classList.remove('hide');
          else el.classList.add('hide');
        });
      }
      if (phase) window._raPhase = String(phase).trim();
      const ae = document.activeElement;
      if (ae && (ae.tagName === 'INPUT' || ae.tagName === 'TEXTAREA')) ae.blur();
      window.raPlayResultCue && window.raPlayResultCue();
    }"""

    # 재생 시작 후 워치독 — autoplay 차단·onended 누락 시에도 진행이 멈추지 않게
    # data-ms(예상 재생 길이) + 3초 뒤 숨김 버튼을 강제 클릭. onended 가 정상
    # 동작하면 그쪽에서 clearTimeout 하므로 중복 실행되지 않고, 만에 하나
    # 겹쳐도 on_audio_done 의 phase 가드가 멱등 처리한다.
    WATCHDOG_JS = """() => {
      clearTimeout(window._raWd);
      if (window._raPhase !== 'ai_playing') return;
      const el = document.getElementById('s3-audio-el');
      const ms = el ? (parseInt(el.dataset.ms || '0', 10) || 15000) : 15000;
      window._raWd = setTimeout(() => {
        if (window._raPhase !== 'ai_playing') return;
        const b = document.getElementById('btn-audio-done');
        const t = b ? (b.querySelector('button') || b) : null;
        if (t) t.click();
      }, ms + 3000);
    }"""

    confirm_chain = btn_confirm.click(
        lambda: gr.update(value=""),
        inputs=None, outputs=s3_audio,
    ).then(
        on_confirm, inputs=[state], outputs=_confirm_outputs, show_progress="hidden",
    ).then(
        lambda st: _nav(st, st.get("screen", "S3")), inputs=[state], outputs=[screen_nav],
    ).then(
        fn=None, js=COMBINED_JS, inputs=[screen_nav, phase_nav],
    ).then(
        fn=None, js=WATCHDOG_JS,
    )

    # 재생 종료 이벤트 → 다음 교환 또는 라운드 결과로 진행
    btn_audio_done.click(
        on_audio_done, inputs=[state], outputs=_confirm_outputs, show_progress="hidden",
    ).then(
        lambda st: _nav(st, st.get("screen", "S3")), inputs=[state], outputs=[screen_nav],
    ).then(
        fn=None, js=COMBINED_JS, inputs=[screen_nav, phase_nav],
    )

    # S4 → S2 (다음 라운드) 또는 S5 (세션 종료)
    def on_next_from_s4(st):
        # 화면 전환과 내용 채움을 한 핸들러에서 원자적으로 (분리 시 갱신 누락 발생)
        if st["round"] >= TOTAL_ROUNDS:
            st["phase"] = "game_over"
            nav = _nav(st, "S5")
            return (
                st, nav, gr.update(),
                s5_html(st),
                render_full_history_roll(st["round_results"]),
            )
        st["round"] += 1
        st["exchange"] = 1
        st["exchange_log"] = []
        st["ai_notes"] = []
        reset_note_input(st)
        st["phase"] = "user_input"
        st["key"] = random.choice(KEYS)
        st["bpm"] = random.choice(BPM_CHOICES)
        nav = _nav(st, "S2")
        return st, nav, _s2_info_html(st), gr.update(), gr.update()

    btn_next_round.click(
        on_next_from_s4, inputs=[state],
        outputs=[state, screen_nav, s2_info, s5_result_html, s5_roll],
    ).then(
        fn=None, js=SHOW_SCREEN_JS, inputs=[screen_nav],
    )

    # S3 "See result" (라운드 종료 후 수동 이동용 백업)
    def on_show_result_nav(st):
        sid = "S5" if st["phase"] == "game_over" else "S4"
        return st, _nav(st, sid)

    btn_next_inline.click(
        on_show_result_nav, inputs=[state],
        outputs=[state, screen_nav],
    ).then(
        fn=None, js=SHOW_SCREEN_JS, inputs=[screen_nav],
    ).then(
        _result_panel_updates, inputs=[state],
        outputs=[s4_result_html, s5_result_html, s5_roll],
    )

    # Restart → S1
    def on_restart(_st):
        st = init_state()
        return st, _nav(st, "S1")

    btn_restart.click(
        on_restart, inputs=[state],
        outputs=[state, screen_nav],
    ).then(fn=None, js=SHOW_SCREEN_JS, inputs=[screen_nav])

    btn_restart_inline.click(
        on_restart, inputs=[state],
        outputs=[state, screen_nav],
    ).then(fn=None, js=SHOW_SCREEN_JS, inputs=[screen_nav])

    app.load(fn=None, js=KEYBOARD_JS)
    app.load(fn=None, js=SHOW_SCREEN_JS, inputs=[screen_nav])


if __name__ == "__main__":
    server_name = os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
    server_port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    launch_kwargs: dict = {"share": False, "server_name": server_name, "server_port": server_port}
    if _GR_MAJOR >= 6:
        launch_kwargs["css"] = APP_CSS
        launch_kwargs["js"] = KEYBOARD_JS
    app.launch(**launch_kwargs)