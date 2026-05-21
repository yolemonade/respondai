"""RespondAI — Phase 2 Gradio app (real model)."""
from __future__ import annotations

import random
import time
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

from data.tokenizer import Note, STEPS_PER_BAR
from analysis.scoring import score_response, session_summary, key_consistency, grade_from_total
from input.piano import synth_notes
from inference.generate import load_model_for_inference, generate
from inference.decode import notes_to_wav

# ─── Constants ───────────────────────────────────────────────────────────────

MAX_EXCHANGES  = 3
MAX_NOTES      = 16
DEFAULT_DURATION = 2   # sixteenth-note steps per note slot
TOTAL_ROUNDS   = 2  # TODO: 테스트용 임시값, 발표 전 5로 복원
AI_MODE        = "model"

CHECKPOINT_PATH = "checkpoints/best_inference.pt"

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
    1: "자유롭게 연주하세요",
    2: "AI의 멜로디에 호응해보세요",
    3: "R1의 멜로디를 기억하나요?",
    4: "리듬에 집중해보세요",
    5: "처음 멜로디로 마무리해보세요",
}

EXCHANGE_STEPS = (MAX_NOTES * DEFAULT_DURATION) + 8   # x-width per exchange slot

# AI 응답 길이: 유저 입력에 맞추되 재생은 AI_RESPONSE_MAX_SEC 로 상한
AI_MAX_BARS_CAP = 2
AI_RESPONSE_MAX_SEC = 4.0
AI_MAX_NOTES_EMPTY = 4

# 키보드 입력 확장: 흰 건반(a s d f g h j k l), 검은 건반(w e r t y u i o)
KB_WHITE_SEMIS = [0, 2, 4, 5, 7, 9, 11, 12, 14]          # C4..D5
KB_BLACK_SEMIS = [1, 3, 6, 8, 10, 13, 15, 18]            # C#4..F#5
KB_ALLOWED_MIDIS = sorted({60 + s for s in (KB_WHITE_SEMIS + KB_BLACK_SEMIS)})


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


def compact_note_timeline(notes: List[Note], max_span: int) -> List[Note]:
    """마디 패딩·긴 공백 제거 — 노트를 이어 붙여 재생."""
    if not notes:
        return []
    ordered = sorted(notes, key=lambda n: (n.start, n.end, n.pitch))
    out: List[Note] = []
    t = 0
    for n in ordered:
        dur = max(1, min(n.end - n.start, 8))
        if t + dur > max_span:
            break
        out.append(Note(n.pitch, t, t + dur))
        t += dur
    return out


def fit_ai_response_to_user(
    ai_notes: List[Note], user_notes: List[Note], bpm: int,
) -> List[Note]:
    """유저와 비슷한 길이·밀도로 AI 노트 정리 (무음 구간 제거, 재생 ≤4초)."""
    if not ai_notes:
        return []
    cap_steps = ai_timeline_cap_steps(bpm)
    user_span = max(n.end for n in user_notes) if user_notes else cap_steps // 2
    if user_notes:
        max_notes = min(MAX_NOTES, len(user_notes) + 2)
        max_span = min(user_span + 4, cap_steps)
    else:
        max_notes = AI_MAX_NOTES_EMPTY
        max_span = cap_steps

    trimmed = sorted(ai_notes, key=lambda n: (n.start, n.end))[:max_notes]
    return compact_note_timeline(trimmed, max_span=max_span)


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
        "pending_s4": False,
    }


def round_grade(score: int) -> str:
    if score >= 950: return "⭐⭐⭐ PERFECT"
    if score >= 800: return "⭐⭐ GREAT"
    if score >= 600: return "⭐ CLEAR"
    return "TRY AGAIN"


# ─── Scoring ─────────────────────────────────────────────────────────────────

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
            "total": 0, "feedback": "빈 입력입니다.",
            "raw": {"key_consistency": 0.0, "rhythm_pearson": 0.0, "motif_overlap": 0.0},
        }

    # R1/E1: auto 300 for rhythm + motif; only key computed
    if round_num == 1 and exchange_num == 1:
        kc = int(key_consistency(user_notes, key_token) * 300)
        cb = 50
        return {
            "key_consistency": kc, "rhythm_similarity": 300,
            "motif_usage": 300, "creativity_bonus": cb,
            "total": kc + 300 + 300 + cb,
            "feedback": "기준 모티프가 저장됐습니다!",
            "raw": {"key_consistency": kc / 300, "rhythm_pearson": 1.0, "motif_overlap": 1.0},
        }

    # Rhythm: compare previous AI notes (or 300 auto if E1)
    if exchange_num == 1 or not ai_prev_notes:
        rhythm_score = 300
        rhythm_raw = 1.0
    else:
        tmp = score_response(ai_prev_notes, user_notes, key_token)
        rhythm_score = tmp["rhythm_similarity"]
        rhythm_raw = tmp["raw"]["rhythm_pearson"]

    # Key consistency and creativity: use ai_prev (or empty placeholder)
    base = score_response(ai_prev_notes or user_notes, user_notes, key_token)
    key_score = base["key_consistency"]
    creativity = base["creativity_bonus"]

    # Motif: compare against r1_motif (or ai_prev if no motif yet)
    if r1_motif:
        motif_res = score_response(r1_motif, user_notes, key_token)
        motif_score = motif_res["motif_usage"]
        motif_raw = motif_res["raw"]["motif_overlap"]
    else:
        motif_score = base["motif_usage"]
        motif_raw = base["raw"]["motif_overlap"]

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


def build_round_audio(
    user_notes: List[Note],
    ai_notes: List[Note],
    bpm: int,
) -> tuple:
    """Return (sample_rate, int16_mono) for Gradio gr.Audio."""
    gap = np.zeros(int(0.15 * SAMPLE_RATE), dtype=np.float32)
    user_wav = synth_notes(user_notes, bpm=bpm, sample_rate=SAMPLE_RATE, role="user")
    ai_wav = synth_notes(
        ai_notes, bpm=bpm, sample_rate=SAMPLE_RATE, role="ai", tail_sec=0.05,
    )
    ai_max_samples = int(AI_RESPONSE_MAX_SEC * SAMPLE_RATE)
    if len(ai_wav) > ai_max_samples:
        ai_wav = ai_wav[:ai_max_samples]
    combined = np.concatenate([user_wav, gap, ai_wav])
    peak = float(np.abs(combined).max())
    if peak < 1e-8:
        combined = gap
        peak = float(np.abs(gap).max()) or 1.0
    combined = (combined / peak * 0.92).astype(np.float32)
    return (SAMPLE_RATE, (combined * 32767).astype(np.int16))


# ─── Piano roll rendering ────────────────────────────────────────────────────

def render_piano_roll(state: dict) -> plt.Figure:
    plt.close("all")
    fig, ax = plt.subplots(figsize=(8.5, 1.65))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    pitch_lo, pitch_hi = 40, 90

    # Draw exchange separator lines
    for ex in range(MAX_EXCHANGES + 1):
        x = ex * EXCHANGE_STEPS
        ax.axvline(x, color="#444466", linewidth=0.8, linestyle="--")

    # Exchange label backgrounds
    for ex in range(MAX_EXCHANGES):
        x = ex * EXCHANGE_STEPS
        ax.text(x + EXCHANGE_STEPS / 2, pitch_hi - 1.5,
                f"교환 {ex+1}", ha="center", va="top",
                color="#888899", fontsize=7)

    def draw_notes(notes, exchange_idx, color, alpha=0.85):
        offset = exchange_idx * EXCHANGE_STEPS
        for n in notes:
            width = max(n.end - n.start, 1)
            rect = mpatches.FancyBboxPatch(
                (offset + n.start, n.pitch - 0.4), width, 0.8,
                boxstyle="round,pad=0.1",
                facecolor=color, edgecolor="white",
                linewidth=0.5, alpha=alpha,
                transform=ax.transData,
            )
            ax.add_patch(rect)

    for i, entry in enumerate(state["exchange_log"]):
        draw_notes(entry["user_notes"], i, "#4A9EF5")
        draw_notes(entry["ai_notes"],   i, "#F55A4A")

    # Current (unconfirmed) notes in current exchange slot
    cur_ex_idx = state["exchange"] - 1
    draw_notes(state["current_notes"], cur_ex_idx, "#4A9EF5", alpha=0.45)

    ax.set_xlim(0, MAX_EXCHANGES * EXCHANGE_STEPS)
    ax.set_ylim(pitch_lo, pitch_hi)
    ax.set_xlabel("Step", color="#aaaacc", fontsize=8)
    ax.set_ylabel("Pitch", color="#aaaacc", fontsize=8)
    ax.tick_params(colors="#aaaacc", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")

    # Legend
    user_patch = mpatches.Patch(color="#4A9EF5", label="플레이어")
    ai_patch   = mpatches.Patch(color="#F55A4A", label="AI")
    ax.legend(handles=[user_patch, ai_patch], loc="lower right",
              facecolor="#1a1a2e", edgecolor="#555577",
              labelcolor="white", fontsize=8)

    fig.tight_layout(pad=0.5)
    return fig


def render_full_history_roll(round_results: List[dict]) -> plt.Figure:
    """Piano roll showing all rounds (for S4/S5)."""
    plt.close("all")
    fig, ax = plt.subplots(figsize=(8.5, 1.75))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    pitch_lo, pitch_hi = 40, 90
    round_width = MAX_EXCHANGES * EXCHANGE_STEPS + 16

    for r_idx, rr in enumerate(round_results):
        r_offset = r_idx * round_width
        ax.axvline(r_offset, color="#666688", linewidth=1.2)
        ax.text(r_offset + round_width / 2, pitch_hi - 1.5,
                f"R{rr['round_num']}", ha="center", color="#aaaacc", fontsize=8)
        for ex_idx, entry in enumerate(rr["exchange_log"]):
            ex_offset = r_offset + ex_idx * EXCHANGE_STEPS
            for n in entry["user_notes"]:
                rect = mpatches.FancyBboxPatch(
                    (ex_offset + n.start, n.pitch - 0.4), max(n.end - n.start, 1), 0.8,
                    boxstyle="round,pad=0.1", facecolor="#4A9EF5",
                    edgecolor="white", linewidth=0.4, alpha=0.85,
                )
                ax.add_patch(rect)
            for n in entry["ai_notes"]:
                rect = mpatches.FancyBboxPatch(
                    (ex_offset + n.start, n.pitch - 0.4), max(n.end - n.start, 1), 0.8,
                    boxstyle="round,pad=0.1", facecolor="#F55A4A",
                    edgecolor="white", linewidth=0.4, alpha=0.85,
                )
                ax.add_patch(rect)

    total_width = len(round_results) * round_width
    ax.set_xlim(0, max(total_width, 1))
    ax.set_ylim(pitch_lo, pitch_hi)
    ax.tick_params(colors="#aaaacc", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")
    fig.tight_layout(pad=0.5)
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
            f'<svg width="160" height="160" style="background:#0d0d1a;border-radius:50%;">'
            f'<circle cx="{cx}" cy="{cy}" r="45" fill="none" '
            f'stroke="{"#334" if role == "player" else "#433"}" stroke-width="2"/>'
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
        f'<svg width="160" height="160" style="background:#0d0d1a;border-radius:50%;">'
        + "".join(slices)
        + f'<circle cx="{cx}" cy="{cy}" r="{inner_r}" fill="#0d0d1a" stroke="none"/>'
        + f'<text x="{cx}" y="{cy+5}" text-anchor="middle" fill="#ccc" font-size="14">{label}</text>'
        + f'</svg>'
    )


# ─── JS: elem_id nk-{midi} / btn-undo Gradio 버튼 클릭 ─────────────────────

KEYBOARD_JS = """() => {
  if (window.__respondAIKeyboardV === 5) return;
  window.__respondAIKeyboardV = 5;

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

  function isInputLocked() {
    const btn = gradioBtn('btn-confirm');
    return !!(btn && btn.disabled);
  }

  function isVisible(el) {
    if (!el) return false;
    const cs = window.getComputedStyle ? getComputedStyle(el) : null;
    if (cs && (cs.display === 'none' || cs.visibility === 'hidden')) return false;
    const r = el.getBoundingClientRect ? el.getBoundingClientRect() : null;
    return !!(r && r.width > 0 && r.height > 0);
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
  function ensureAudioCtx() {
    if (!audioCtx) {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (audioCtx.state === 'suspended') audioCtx.resume();
    return audioCtx;
  }

  function playPreviewTone(midi) {
    if (midi < 21 || midi > 108) return;
    try {
      const ctx = ensureAudioCtx();
      const freq = 440 * Math.pow(2, (midi - 69) / 12);
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      const t0 = ctx.currentTime;
      const dur = 0.22;
      osc.type = 'triangle';
      osc.frequency.value = freq;
      gain.gain.setValueAtTime(0.0001, t0);
      gain.gain.exponentialRampToValueAtTime(0.32, t0 + 0.015);
      gain.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start(t0);
      osc.stop(t0 + dur + 0.04);
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

  window.respondAI = { sendNote, flashKey };

  document.addEventListener('mousedown', function(e) {
    if (isInputLocked()) return;
    const key = e.target.closest && e.target.closest('[data-midi]');
    if (!key || !key.dataset.midi) return;
    e.preventDefault();
    const midi = parseInt(key.dataset.midi, 10);
    if (!isNaN(midi)) { sendNote(midi); flashKey(midi); }
  }, true);

  function isGameKey(e) {
    if (e.key === 'Enter' || e.key === 'Backspace') return true;
    if (e.shiftKey && (e.key === 'ArrowUp' || e.key === 'ArrowDown')) return true;
    return midiFromEvent(e) !== null;
  }

  window.addEventListener('keydown', function(e) {
    if (!isGameKey(e)) return;
    if (isInputLocked()) return;
    if (e.repeat) return;
    e.preventDefault();
    e.stopImmediatePropagation();
    if (e.shiftKey && e.key === 'ArrowUp') {
      baseOctave = Math.min(baseOctave + 1, 7);
      const d = document.getElementById('oct-display');
      if (d) d.textContent = baseOctave;
      return;
    }
    if (e.shiftKey && e.key === 'ArrowDown') {
      baseOctave = Math.max(baseOctave - 1, 0);
      const d = document.getElementById('oct-display');
      if (d) d.textContent = baseOctave;
      return;
    }
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

  console.log('[RespondAI] keyboard ready');
}"""

STOP_AUDIO_JS = """() => {
  document.querySelectorAll('audio').forEach(a => {
    try { a.pause(); a.currentTime = 0; } catch (_) {}
  });
}"""

# Gradio Audio blob URL이 비어 duration=NaN 인 경우가 있어 JS ended 대기는 멈춤 → 서버 sleep 사용
PLAY_EXCHANGE_JS = """async () => {
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  function findAudio() {
    const roots = [document];
    document.querySelectorAll('gradio-app, .gradio-container').forEach((h) => {
      if (h.shadowRoot) roots.push(h.shadowRoot);
    });
    for (const root of roots) {
      const host = root.getElementById('s3-exchange-audio')
        || root.querySelector('#s3-exchange-audio');
      if (host) {
        const a = host.tagName === 'AUDIO' ? host : host.querySelector('audio');
        if (a) return a;
      }
    }
    return null;
  }
  for (let i = 0; i < 40; i++) {
    const a = findAudio();
    if (a && (a.currentSrc || a.src)) {
      try { a.pause(); a.currentTime = 0; await a.play(); } catch (_) {}
      return;
    }
    await sleep(50);
  }
}"""

FOCUS_GAME_JS = """() => {
  const ae = document.activeElement;
  if (ae && (ae.tagName === 'INPUT' || ae.tagName === 'TEXTAREA')) ae.blur();
}"""


# ─── Piano keyboard HTML (pure HTML/CSS, NO <script> — onclick calls window.respondAI) ──

def render_piano_html(base_octave: int = 4) -> str:
    NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    white_midis = [60 + s for s in KB_WHITE_SEMIS]
    black_midis = [60 + s for s in KB_BLACK_SEMIS]

    WW, WH = 32, 76
    BW, BH = 18, 46
    total_w = len(white_midis) * WW

    whites, blacks = [], []
    for wi, midi in enumerate(white_midis):
        left = wi * WW
        name = NOTE_NAMES[midi % 12] + str(midi // 12 - 1)
        whites.append(
            f'<div data-midi="{midi}"'
            f' class="ra-key ra-key-white"'
            f' style="position:absolute;left:{left}px;width:{WW-2}px;height:{WH}px;'
            f'cursor:pointer;display:flex;align-items:flex-end;justify-content:center;'
            f'padding-bottom:3px;box-sizing:border-box;">'
            f'<span class="ra-key-label">{name}</span>'
            f'</div>'
        )

    # 검은 건반은 확장 키 배열 순서대로 배치 (8개)
    for bi, midi in enumerate(black_midis):
        left = (bi + 1) * WW - BW // 2 - 1
        blacks.append(
            f'<div data-midi="{midi}"'
            f' class="ra-key ra-key-black"'
            f' style="position:absolute;left:{left}px;top:0;width:{BW}px;height:{BH}px;'
            f'border-radius:0 0 3px 3px;'
            f'cursor:pointer;z-index:2;">'
            f'</div>'
        )

    return (
        f'<div class="ra-piano-wrap">'
        f'<div class="ra-piano-help">'
        f'옥타브 <span id="oct-display">{base_octave}</span>'
        f' &nbsp;|&nbsp;'
        f'<kbd>Shift+↑↓</kbd> 옥타브'
        f' &nbsp;'
        f'<kbd>a s d f g h j k l</kbd> '
        f'<kbd>w e r t y u i o</kbd> 또는 '
        f'<kbd>ㅁ ㄴ ㅇ ㄹ ㅎ ㅗ ㅓ ㅏ ㅣ / ㅈ ㄷ ㄱ ㅅ ㅛ ㅕ ㅑ ㅐ</kbd>'
        f'</div>'
        f'<div class="ra-piano-keys" style="width:{total_w}px;height:{WH}px;">'
        + ''.join(whites)
        + ''.join(blacks)
        + '</div></div>'
    )


# ─── HUD / info HTML helpers ─────────────────────────────────────────────────

def hud_html(state: dict) -> str:
    phase = state["phase"]
    if phase == "user_input":
        phase_label = "🎵 당신의 차례"
        phase_color = "#4af"
    elif phase == "ai_response":
        phase_label = "🤖 AI 응답 중..."
        phase_color = "#f84"
    elif phase == "round_result":
        phase_label = "📊 라운드 결과"
        phase_color = "#f4c430"
    else:
        phase_label = "🏁 게임 종료"
        phase_color = "#9ef"
    return f"""
<div class="ra-hud">
  <div class="ra-hud-group">
    <span class="ra-hud-round">R{state['round']}/{TOTAL_ROUNDS}</span>
    <span class="ra-hud-exchange">교환 {state['exchange']}/{MAX_EXCHANGES}</span>
  </div>
  <div class="ra-hud-group">
    <span class="ra-chip">
      🎼 {state['key']}
    </span>
    <span class="ra-chip">
      ♩ {state['bpm']} BPM
    </span>
  </div>
  <div class="ra-hud-group">
    <span class="ra-hud-score">
      {state['total_score']} pts
    </span>
  </div>
  <div class="ra-hud-phase" style="color:{phase_color};">{phase_label}</div>
</div>
"""


def note_list_html(notes: List[Note]) -> str:
    if not notes:
        return '<div class="ra-note-empty">노트 없음 - 건반을 눌러 입력하세요</div>'
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
    <div style="background:linear-gradient(90deg,{color},#00f3ff);width:{pct}%;height:10px;border-radius:999px;transition:width .4s;"></div>
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
        score_bar_html("조성 일관성", exs[-1]["key_consistency"],  300, "#4A9EF5")
        + score_bar_html("리듬 유사도",  exs[-1]["rhythm_similarity"],300, "#9b59b6")
        + score_bar_html("모티프 활용",  exs[-1]["motif_usage"],     300, "#2ecc71")
        + score_bar_html("창의성 보너스",exs[-1]["creativity_bonus"],100, "#f39c12")
    )
    r5_bonus = ""
    if rr["round_num"] == 5 and rr.get("r5_motif_bonus"):
        r5_bonus = '<div class="ra-r5-bonus">🌟 R5 모티프 보너스 +150점!</div>'

    feedback = exs[-1].get("feedback", "")
    return f"""
<div class="ra-result-card">
  <h2 class="ra-result-title">Round {rr['round_num']} 결과</h2>
  <div class="ra-result-grade">{grade}</div>
  <div class="ra-result-total">{round_total} / 1000</div>
  <div class="ra-result-feedback">"{feedback}"</div>
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

    grade_colors = {"S": "#ffd700", "A": "#c0c0c0", "B": "#cd7f32", "C": "#aaaaaa"}
    gc = grade_colors.get(grade, "#fff")

    round_grades_html = " | ".join(
        f"R{rr['round_num']} {round_grade(int(sum(s['total'] for s in rr['exchange_scores'])/len(rr['exchange_scores'])))}"
        for rr in round_results
    )

    bonus_detail = ""
    if summary["bonus_score"] or r5_motif_bonus:
        bonus_detail = (
            f'<div class="ra-final-bonus">'
            f'콤보/조성 보너스: +{summary["bonus_score"]} &nbsp; '
            f'R5 모티프 보너스: +{r5_motif_bonus}'
            f'</div>'
        )

    return f"""
<div class="ra-final-card">
  <div class="ra-final-grade" style="color:{gc};">{grade}</div>
  <div class="ra-final-total">{final_total} / 5000</div>
  {bonus_detail}
  <div class="ra-final-rounds">{round_grades_html}</div>
</div>
"""


# ─── CSS ─────────────────────────────────────────────────────────────────────

APP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Press+Start+2P&display=swap');

:root {
  --bg-0: #0d0e15;
  --bg-1: #12131c;
  --panel: rgba(18, 20, 34, 0.86);
  --line: rgba(188, 19, 254, 0.44);
  --txt-main: #e9efff;
  --txt-sub: #9ba9c9;
  --neon-blue: #00f3ff;
  --neon-pink: #ff007f;
  --neon-purple: #bc13fe;
  --cyber-yellow: #ffe600;
}

html, body {
  height: 100% !important;
  margin: 0 !important;
  overflow: hidden !important;
  color: var(--txt-main) !important;
  background:
    radial-gradient(1100px 420px at 10% -20%, rgba(0,243,255,0.15), transparent 70%),
    radial-gradient(900px 360px at 90% 0%, rgba(255,0,127,0.16), transparent 68%),
    linear-gradient(180deg, var(--bg-1), var(--bg-0)) !important;
  font-family: 'Orbitron', 'Apple SD Gothic Neo', 'Nanum Gothic', sans-serif !important;
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
}
footer, .footer { display: none !important; }
#main, .app {
  height: 100% !important;
  overflow: hidden !important;
}
.gradio-container, .gradio-container * { color: var(--txt-main); }

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
  padding: 10px !important;
  background: var(--panel) !important;
  border: 1px solid var(--line) !important;
  border-radius: 14px !important;
  box-shadow: 0 0 18px rgba(188, 19, 254, 0.22), inset 0 0 14px rgba(0, 243, 255, 0.08) !important;
}
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
.screen-actions button:disabled {
  display: none !important;
}
.quick-notes button { min-width: 1.8rem !important; padding: 3px 4px !important; font-size: 11px !important; }
.gradio-button, button {
  white-space: nowrap !important;
  background: linear-gradient(180deg, rgba(40,45,70,.94), rgba(18,20,32,.98)) !important;
  border: 1px solid rgba(0,243,255,.5) !important;
  color: #f6fbff !important;
  box-shadow: 0 0 10px rgba(0,243,255,.26) !important;
}
.gradio-button:hover, button:hover {
  border-color: rgba(0,243,255,.9) !important;
  box-shadow: 0 0 16px rgba(0,243,255,.55) !important;
}

/* S3: 피아노롤만 넓게, 사이드 viz 숨김 */
.s3-main .viz-side { display: none !important; }
.s3-main .piano-roll-host { flex: 1 1 auto !important; min-width: 0 !important; }
.s3-main .piano-roll-host .plot-container,
.s3-main .piano-roll-host canvas {
  max-height: 150px !important;
  height: 150px !important;
}
.game-panel .plot-container,
.game-panel .piano-roll-host canvas {
  max-height: 140px !important;
}

.ra-title-wrap {
  display: flex; flex-direction: column; justify-content: center; align-items: center;
  text-align: center; min-height: 220px;
}
.ra-logo {
  font-family: 'Press Start 2P', 'Orbitron', monospace;
  font-size: clamp(34px, 7vw, 58px);
  color: var(--neon-pink);
  text-shadow: 0 0 10px rgba(255,0,127,.8), 0 0 30px rgba(188,19,254,.45);
  letter-spacing: 2px;
}
.ra-subtitle { margin-top: 14px; font-size: 13px; color: #d6e2ff; opacity: .95; }
.mode-card { min-height: 52px !important; border-width: 2px !important; }

.ra-round-card, .ra-result-card, .ra-final-card {
  margin: 0 auto; width: min(96%, 640px);
  background: rgba(10, 14, 28, 0.86);
  border: 1px solid rgba(188,19,254,.5);
  border-radius: 14px;
  padding: 16px 18px;
  box-shadow: 0 0 20px rgba(188,19,254,.22), inset 0 0 12px rgba(0,243,255,.08);
}
.ra-round-title, .ra-result-title {
  margin: 0;
  color: #f5f9ff;
  font-size: 24px;
  font-weight: 900;
}
.ra-round-meta { margin: 10px 0 6px; color: #dbe6ff; font-size: 17px; }
.ra-round-cond, .ra-result-feedback { color: #b8c8ed; font-size: 14px; font-style: italic; margin-top: 8px; }

.ra-hud {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 8px;
  align-items: center;
  padding: 10px 14px;
  border-radius: 10px;
  border: 1px solid rgba(0,243,255,.36);
  background: rgba(10, 16, 32, .92);
  box-shadow: 0 0 14px rgba(0,243,255,.2), inset 0 0 10px rgba(188,19,254,.1);
}
.ra-hud-group { text-align: center; }
.ra-hud-round { font-family: 'Press Start 2P', 'Orbitron', monospace; font-size: 12px; color: var(--cyber-yellow); }
.ra-hud-exchange { margin-left: 10px; color: #e6eeff; font-size: 13px; }
.ra-chip {
  display: inline-block; padding: 4px 8px; margin: 0 4px;
  border-radius: 8px; background: rgba(188,19,254,.12); border: 1px solid rgba(188,19,254,.35);
}
.ra-hud-score { font-size: 20px; font-weight: 900; color: var(--cyber-yellow); text-shadow: 0 0 10px rgba(255,230,0,.28); }
.ra-hud-phase { font-weight: 700; text-align: center; }

.ra-piano-wrap {
  user-select: none; padding: 10px 8px; border-radius: 10px; text-align: center;
  background: rgba(10, 14, 26, 0.9); border: 1px solid rgba(0,243,255,.26);
}
.ra-piano-help { color: #d8e3ff; font-size: 11px; margin-bottom: 8px; line-height: 1.65; }
.ra-piano-help kbd {
  background: rgba(30, 36, 56, 0.95); color: #f6fbff; border: 1px solid rgba(0,243,255,.3);
  padding: 1px 5px; border-radius: 4px; font-size: 10px;
}
.ra-piano-keys { display: inline-block; position: relative; }
.ra-key { transition: transform .08s ease, box-shadow .12s ease, filter .12s ease; }
.ra-key-white {
  background: linear-gradient(180deg, rgba(255,255,255,.96), rgba(224,232,247,.9));
  border: 1px solid #5f6c8d;
  border-radius: 0 0 6px 6px;
}
.ra-key-black {
  background: linear-gradient(180deg, #262b39, #090b12);
  border: 1px solid #000;
  box-shadow: inset 0 -8px 10px rgba(0,0,0,.35);
}
.ra-key.active {
  transform: translateY(2px) scale(0.99);
  box-shadow: 0 0 16px rgba(0,243,255,.92), inset 0 0 10px rgba(0,243,255,.6);
  filter: brightness(1.16);
}
.ra-key-label { font-size: 8px; color: #5a6685; pointer-events: none; }

.ra-note-empty { color: #d3def6; font-size: 12px; padding: 6px; }
.ra-note-list { display: flex; flex-wrap: wrap; padding: 4px; align-items: center; }
.ra-note-chip {
  background: rgba(0,243,255,.14); border: 1px solid rgba(0,243,255,.52);
  border-radius: 999px; padding: 2px 8px; margin: 2px; font-size: 12px; color: #ebf8ff;
}
.ra-note-count { color: #b7c5e9; font-size: 11px; margin: 4px; }

.ra-score-row { margin: 6px 0; }
.ra-score-head { display: flex; justify-content: space-between; font-size: 12px; color: #dce8ff; margin-bottom: 3px; }
.ra-score-track { background: rgba(24, 32, 58, 0.98); border-radius: 999px; height: 10px; border: 1px solid rgba(188,19,254,.32); }
.ra-result-grade { font-size: 24px; margin: 8px 0 2px; color: #f7fbff; text-shadow: 0 0 12px rgba(188,19,254,.45); }
.ra-result-total { font-size: 19px; color: var(--cyber-yellow); margin-bottom: 9px; }
.ra-r5-bonus { color: #ffe873; font-size: 14px; margin-top: 10px; text-shadow: 0 0 10px rgba(255,230,0,.35); }

.ra-final-card { text-align: center; }
.ra-final-grade {
  font-size: 56px; font-weight: 900; letter-spacing: 4px;
  text-shadow: 0 0 14px currentColor, 0 0 32px rgba(188,19,254,.38);
  animation: ra-pulse 1.8s ease-in-out infinite;
}
.ra-final-total { font-size: 28px; color: var(--cyber-yellow); margin: 8px 0; }
.ra-final-bonus { color: #baff9a; font-size: 13px; margin-top: 6px; }
.ra-final-rounds { font-size: 13px; color: #d3dfff; margin-top: 12px; }
@keyframes ra-pulse {
  0%, 100% { transform: scale(1); opacity: 1; }
  50% { transform: scale(1.04); opacity: 0.92; }
}
"""


# ─── Gradio app ──────────────────────────────────────────────────────────────

def _vis(v: bool):
    return gr.update(visible=v)


def _stop_audio():
    """AI 재생 종료 — 내 턴으로 돌아올 때 이전 오디오가 반복되지 않게."""
    return gr.update(value=None, visible=False, autoplay=False)


def _screens(*visible_ids):
    """Return visibility updates for [s1,s2,s3,s4,s5] in order."""
    ids = ["S1", "S2", "S3", "S4", "S5"]
    return tuple(gr.update(visible=(sid in visible_ids)) for sid in ids)


with gr.Blocks(title="RespondAI") as app:
    state = gr.State(init_state())

    with gr.Column(visible=True, elem_classes=["game-stage"]):
        # ── S1: Title ────────────────────────────────────────────────────────
        with gr.Column(visible=True, elem_classes=["game-panel"]) as screen_s1:
            with gr.Column(elem_classes=["screen-body"]):
                gr.HTML("""
<div class="ra-title-wrap">
  <div class="ra-logo">RespondAI</div>
  <div class="ra-subtitle">AI와 함께하는 즉흥 연주 세션</div>
</div>
""")
            with gr.Row(equal_height=True, elem_classes=["screen-actions"]):
                btn_piano_start  = gr.Button("🎹 피아노 모드 시작", variant="primary", scale=2, elem_classes=["mode-card"])
                btn_humming_start = gr.Button("🎤 허밍 모드 (Beta)", variant="secondary",
                                              scale=1, interactive=False, elem_classes=["mode-card"])

        # ── S2: Round start ──────────────────────────────────────────────────
        with gr.Column(visible=False, elem_classes=["game-panel"]) as screen_s2:
            with gr.Column(elem_classes=["screen-body"]):
                s2_info = gr.HTML()
            with gr.Row(elem_classes=["screen-actions"]):
                btn_round_start = gr.Button("▶ 시작", variant="primary")

        # ── S3: Main game ────────────────────────────────────────────────────
        with gr.Column(visible=False, elem_classes=["game-panel"]) as screen_s3:
            with gr.Column(elem_classes=["screen-body", "s3-main"]):
                s3_hud_html  = gr.HTML()
                with gr.Row():
                    s3_viz_player = gr.HTML(render_energy_svg([], "player"), elem_classes=["viz-side"])
                    with gr.Column(elem_classes=["piano-roll-host"]):
                        s3_roll = gr.Plot(show_label=False)
                    s3_viz_ai = gr.HTML(render_energy_svg([], "ai"), elem_classes=["viz-side"])
                s3_piano     = gr.HTML(render_piano_html(4))
                s3_note_list = gr.HTML(note_list_html([]))
                s3_audio     = gr.Audio(label="", autoplay=True, visible=False, elem_id="s3-exchange-audio")
            with gr.Row(elem_classes=["screen-actions"]):
                btn_cancel   = gr.Button("← 취소", scale=1)
                btn_preview  = gr.Button("▶ 미리듣기", scale=1)
                btn_confirm  = gr.Button("✅ 확정 (Enter)", scale=2, variant="primary", elem_id="btn-confirm")
                btn_next_inline = gr.Button("다음 라운드 →", variant="primary", interactive=False, elem_id="btn-next-inline")
                btn_restart_inline = gr.Button("🔄 다시하기", variant="secondary", interactive=False, elem_id="btn-restart-inline")

        # ── S4: Round result ─────────────────────────────────────────────────
        with gr.Column(visible=False, elem_classes=["game-panel"]) as screen_s4:
            with gr.Column(elem_classes=["screen-body"]):
                s4_result_html = gr.HTML()
                with gr.Column(elem_classes=["piano-roll-host"]):
                    s4_roll = gr.Plot(show_label=False)
            with gr.Row(elem_classes=["screen-actions"]):
                btn_next_round = gr.Button("다음 라운드 →", variant="primary")

        # ── S5: Final result ─────────────────────────────────────────────────
        with gr.Column(visible=False, elem_classes=["game-panel"]) as screen_s5:
            with gr.Column(elem_classes=["screen-body"]):
                s5_result_html = gr.HTML()
                with gr.Column(elem_classes=["piano-roll-host"]):
                    s5_roll = gr.Plot(show_label=False)
            with gr.Row(elem_classes=["screen-actions"]):
                btn_restart = gr.Button("🔄 다시하기", variant="secondary")

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
<div class="ra-round-card">
  <div class="ra-round-title">
    Round {rnd} / {TOTAL_ROUNDS}
  </div>
  <div class="ra-round-meta">
    🎼 {st['key']} &nbsp;&nbsp; ♩ {st['bpm']} BPM
  </div>
  <div class="ra-round-cond">
    "{cond}"
  </div>
</div>
"""

    # S1 → S2
    def on_piano_start(st):
        st = init_state()
        st["mode"] = "piano"
        st["key"]  = random.choice(KEYS)
        st["bpm"]  = random.choice(BPM_CHOICES)
        return (
            st,
            _s2_info_html(st),
            *_screens("S2"),
        )

    btn_piano_start.click(
        on_piano_start, inputs=[state],
        outputs=[state, s2_info, screen_s1, screen_s2, screen_s3, screen_s4, screen_s5],
    )

    # S2 → S3 (Gradio 6: 자식 갱신이 부모 visible을 덮어씀 → 2단계 체인)
    def on_round_start_state(st):
        st["exchange"]      = 1
        st["phase"]         = "user_input"
        st["current_notes"] = []
        st["ai_notes"]      = []
        st["exchange_log"]  = []
        return st, *_screens("S3")

    def on_round_start_ui(st):
        return (
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
            *_screens("S3"),
        )

    round_start_chain = btn_round_start.click(
        on_round_start_state, inputs=[state],
        outputs=[state, screen_s1, screen_s2, screen_s3, screen_s4, screen_s5],
    ).then(
        on_round_start_ui, inputs=[state],
        outputs=[s3_hud_html, s3_roll, s3_viz_player, s3_viz_ai, s3_note_list,
                 s3_piano,
                 btn_confirm, btn_cancel, btn_preview, btn_next_inline, btn_restart_inline, s3_audio,
                 screen_s1, screen_s2, screen_s3, screen_s4, screen_s5],
    )
    round_start_chain.then(fn=None, js=FOCUS_GAME_JS)

    def on_note_event(midi: int, st: dict):
        if st["phase"] != "user_input":
            return st, render_piano_roll(st), note_list_html(st["current_notes"])

        if midi == -1:
            if st["current_notes"]:
                st["current_notes"] = st["current_notes"][:-1]
        elif 21 <= midi <= 108:
            if len(st["current_notes"]) < MAX_NOTES:
                i = len(st["current_notes"])
                st["current_notes"] = st["current_notes"] + [
                    Note(midi, i * DEFAULT_DURATION, (i + 1) * DEFAULT_DURATION)
                ]

        return st, render_piano_roll(st), note_list_html(st["current_notes"])

    btn_undo.click(
        lambda st: on_note_event(-1, st), inputs=[state],
        outputs=[state, s3_roll, s3_note_list],
    )
    for midi, btn in _note_btn_map:
        btn.click(
            lambda st, m=midi: on_note_event(m, st),
            inputs=[state],
            outputs=[state, s3_roll, s3_note_list],
        )

    # Cancel last note
    def on_cancel(st):
        if st["phase"] == "user_input" and st["current_notes"]:
            st["current_notes"] = st["current_notes"][:-1]
        return st, render_piano_roll(st), note_list_html(st["current_notes"])

    btn_cancel.click(on_cancel, inputs=[state],
                     outputs=[state, s3_roll, s3_note_list])

    # Preview (play current notes)
    def on_preview(st):
        audio = build_round_audio(st["current_notes"], [], st["bpm"])
        return gr.update(value=audio, visible=True)

    btn_preview.click(on_preview, inputs=[state], outputs=[s3_audio])

    _confirm_outputs = [
        state, s3_roll, s3_note_list, s3_hud_html, s3_viz_player, s3_viz_ai, s3_piano,
        btn_confirm, btn_cancel, btn_preview, btn_next_inline, btn_restart_inline, s3_audio,
    ]

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

    def on_confirm(st):
        """확정: AI 생성/재생 대기 + 교환/라운드/게임오버 UI 갱신."""
        if st["phase"] != "user_input":
            last = st["exchange_log"][-1] if st["exchange_log"] else None
            last_attn = last.get("attn_scores") if last else None
            show_next = (st.get("phase") == "round_result" and st.get("round", 1) < TOTAL_ROUNDS)
            show_restart = (st.get("phase") == "game_over")
            return (
                st,
                render_piano_roll(st), note_list_html([]), hud_html(st),
                render_energy_svg([], "player"),
                render_energy_svg(st["ai_notes"], "ai", last_attn),
                gr.update(),
                gr.update(interactive=(st["phase"] == "user_input")),
                gr.update(interactive=(st["phase"] == "user_input")),
                gr.update(interactive=(st["phase"] == "user_input")),
                gr.update(interactive=show_next),
                gr.update(interactive=show_restart),
                gr.update(),
            )

        user_notes = list(st["current_notes"])
        key_token = KEY_TO_TOKEN[st["key"]]
        exchange_num = len(st["exchange_log"]) + 1
        st["exchange"] = exchange_num
        st["phase"] = "ai_response"

        max_bars, max_new_tokens = generation_limits(user_notes)
        result = generate(
            _MODEL, _TOKENIZER, user_notes,
            key=key_token, tempo=st["bpm"],
            max_bars=max_bars,
            max_new_tokens=max_new_tokens,
            temperature=0.95,
        )
        ai_notes = fit_ai_response_to_user(result.response_notes, user_notes, st["bpm"])
        attn_scores = align_attn_to_notes(result.attn_scores, ai_notes)
        score = compute_exchange_score(
            user_notes, st["ai_notes"], st["r1_motif"],
            st["round"], exchange_num, key_token,
        )

        if st["round"] == 1 and exchange_num == 1 and user_notes:
            st["r1_motif"] = user_notes

        st["exchange_log"] = st["exchange_log"] + [{
            "user_notes": user_notes,
            "ai_notes": ai_notes,
            "score": score,
            "attn_scores": attn_scores,
        }]
        st["ai_notes"] = ai_notes
        st["current_notes"] = []

        audio = build_round_audio(user_notes, ai_notes, st["bpm"])
        ms = audio_duration_ms(audio)
        time.sleep(max(0.8, min(ms / 1000.0, AI_RESPONSE_MAX_SEC + 2.0)))
        audio_up = gr.update(value=audio, visible=True, autoplay=True)

        completed = len(st["exchange_log"])
        if completed < MAX_EXCHANGES:
            st["exchange"] = completed + 1
            st["phase"] = "user_input"
            return (
                st,
                render_piano_roll(st), note_list_html([]), hud_html(st),
                render_energy_svg([], "player"),
                render_energy_svg(st["ai_notes"], "ai", attn_scores),
                gr.update(value=render_piano_html(4)),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=False),
                gr.update(interactive=False),
                audio_up,
            )

        _finalize_round(st)
        is_final = st["round"] >= TOTAL_ROUNDS
        if is_final:
            st["phase"] = "game_over"
            result_html = s5_html(st)
        else:
            result_html = s4_html(st)
        return (
            st,
            render_full_history_roll(st["round_results"]),
            result_html,
            hud_html(st),
            render_energy_svg([], "player"),
            render_energy_svg([], "ai"),
            gr.update(value=""),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=not is_final),
            gr.update(interactive=is_final),
            audio_up,
        )

    confirm_chain = btn_confirm.click(
        on_confirm, inputs=[state], outputs=_confirm_outputs, show_progress="full",
    )
    confirm_chain.then(fn=None, js=FOCUS_GAME_JS)

    def on_next_round_state(st):
        if st["round"] >= TOTAL_ROUNDS:
            st["phase"] = "game_over"
            return st, *_screens("S5")
        st["round"] += 1
        st["exchange"] = 1
        st["exchange_log"] = []
        st["ai_notes"] = []
        st["current_notes"] = []
        st["phase"] = "user_input"
        st["key"] = random.choice(KEYS)
        st["bpm"] = random.choice(BPM_CHOICES)
        return st, *_screens("S2")

    def on_next_round_ui(st):
        if st["phase"] == "game_over":
            return (
                gr.update(),
                s5_html(st),
                render_full_history_roll(st["round_results"]),
                *_screens("S5"),
            )
        return (
            _s2_info_html(st),
            gr.update(),
            gr.update(),
            *_screens("S2"),
        )

    btn_next_round.click(
        on_next_round_state, inputs=[state],
        outputs=[state, screen_s1, screen_s2, screen_s3, screen_s4, screen_s5],
    ).then(
        on_next_round_ui, inputs=[state],
        outputs=[s2_info, s5_result_html, s5_roll,
                 screen_s1, screen_s2, screen_s3, screen_s4, screen_s5],
    )

    btn_next_inline.click(
        on_next_round_state, inputs=[state],
        outputs=[state, screen_s1, screen_s2, screen_s3, screen_s4, screen_s5],
    ).then(
        on_next_round_ui, inputs=[state],
        outputs=[s2_info, s5_result_html, s5_roll,
                 screen_s1, screen_s2, screen_s3, screen_s4, screen_s5],
    )

    # Restart → S1
    def on_restart(_st):
        return (
            init_state(),
            *_screens("S1"),
        )

    btn_restart.click(
        on_restart, inputs=[state],
        outputs=[state, screen_s1, screen_s2, screen_s3, screen_s4, screen_s5],
    )

    btn_restart_inline.click(
        on_restart, inputs=[state],
        outputs=[state, screen_s1, screen_s2, screen_s3, screen_s4, screen_s5],
    )

    app.load(fn=None, js=KEYBOARD_JS)


if __name__ == "__main__":
    app.launch(share=False, css=APP_CSS, js=KEYBOARD_JS)
