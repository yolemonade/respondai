# RespondAI — Public Interface for Team B

This document is the contract between the **model side** (A) and the **UI side**
(B). If anything here changes, both sides need to update.

---

## TL;DR — The three imports you need

```python
from data.tokenizer import Note, Tokenizer
from inference.generate import generate, load_model_for_inference
from inference.decode import decode_tokens_to_notes, notes_to_pretty_midi, notes_to_wav
from analysis.scoring import score_response, session_summary, grade_from_total
```

Everything below is detail. These are the only functions B should need.

---

## The `Note` dataclass

The universal currency between modules. Times are in **sixteenth-note steps**,
NOT seconds. One bar = 16 steps; one quarter note = 4 steps.

```python
@dataclass(frozen=True)
class Note:
    pitch: int   # MIDI pitch, 21..108
    start: int   # step count, ≥ 0
    end: int     # step count, > start  (i.e. duration = end - start)
```

When the UI needs seconds for audio playback or MIDI rendering, convert
using the round's BPM:

```python
seconds_per_step = (60.0 / bpm) / 4.0
```

`inference.decode.notes_to_pretty_midi(notes, tempo=bpm)` does this for you.

---

## 1. Loading the model — once at app startup

```python
from inference.generate import load_model_for_inference

model, tokenizer, device = load_model_for_inference(
    "checkpoints/full/best.pt",
    device="auto",  # picks cuda → mps → cpu
)
```

앱 시작 시 1회 로드하고 모듈 수준 변수로 캐시한다 (Gradio는 앱이 한 번 기동되면 상태를 유지하므로 별도 캐시 데코레이터 불필요). The model is read-only during inference; safe to share across sessions.

---

## 2. Generating an AI response

```python
from inference.generate import generate

result = generate(
    model, tokenizer,
    call_notes=user_notes,        # list[Note]
    key="Dm",                     # one of data.tokenizer.KEY_NAMES
    tempo=92,                     # BPM, integer
    max_new_tokens=256,           # safety cap
    max_bars=4,                   # stops after this many [BAR] tokens
    temperature=1.0,
    top_p=0.9,
    return_attention=True,        # set False to speed up if you don't need VIZ-02
)

# result is a GenerationResult dataclass:
result.response_tokens    # list[int]   — raw token ids
result.response_notes     # list[Note]  — already decoded for you
result.attn_scores        # list[float] — per generated token, attention back to prompt
result.stop_reason        # "eos" | "max_tokens" | "max_bars"
```

**Inference time** on Mac MPS for ~10M model + 4-bar output: ~1 second.
On CUDA A100: ~0.3 seconds.

### Notes on `attn_scores`
- Length equals `len(result.response_tokens)` — one score per generated token.
- Each value is the mean attention from that generation step back to the
  prompt tokens (CALL + structural). Higher = "AI is leaning more on the CALL
  for this token".
- For **VIZ-02**, you can interpolate these scores onto note positions or
  just use them as a per-token timeline.

---

## 3. Scoring the response

```python
from analysis.scoring import score_response

score = score_response(
    call_notes=user_notes,
    response_notes=result.response_notes,
    key="Dm",
    num_bars=4,    # match the round's bar count
)

# score is a dict:
score["key_consistency"]    # int, 0..300
score["rhythm_similarity"]  # int, 0..300
score["motif_usage"]        # int, 0..300
score["creativity_bonus"]   # int, 0..100
score["total"]              # int, 0..1000
score["feedback"]           # str — short UI message, e.g. "Motif carried over! Strong key center."
score["raw"]                # dict of raw 0..1 floats, for plotting trends
```

### After all 5 rounds:

```python
from analysis.scoring import session_summary

summary = session_summary([round1, round2, round3, round4, round5])
summary["base_score"]    # sum of round totals
summary["bonus_score"]   # combo / all-in-key bonuses
summary["total"]         # final score
summary["grade"]         # "S" | "A" | "B" | "C"
```

The R5 "include R1 motif" bonus (+150) is **not** applied automatically — the
game engine should add it before calling `session_summary` if applicable.

---

## 4. Rendering audio (FluidSynth)

```python
from inference.decode import notes_to_wav

audio = notes_to_wav(
    result.response_notes,
    tempo=92,
    sample_rate=22050,
    sound_font="/path/to/GeneralUser.sf2",  # optional; can be None on Linux with default sf2
)
# audio is float32 mono. Gradio에서는 (sample_rate, int16_array) 튜플로 변환해서 gr.Audio에 전달:
# audio_int16 = (audio * 32767).astype(np.int16)
# gr.Audio(value=(22050, audio_int16))
```

On Mac, install FluidSynth with `brew install fluid-synth`. The default
sound font on macOS is at `/usr/share/sounds/sf2/FluidR3_GM.sf2` (or similar
— check `brew --prefix fluid-synth`).

---

## 5. Building a `list[Note]` from the virtual piano

The UI captures user clicks as `(pitch, start_seconds, end_seconds)`. Convert
to step coordinates:

```python
def piano_input_to_notes(clicks, bpm: float) -> list[Note]:
    seconds_per_step = (60.0 / bpm) / 4.0
    return [
        Note(
            pitch=c.pitch,
            start=int(round(c.start_seconds / seconds_per_step)),
            end=max(
                int(round(c.start_seconds / seconds_per_step)) + 1,
                int(round(c.end_seconds / seconds_per_step)),
            ),
        )
        for c in clicks
    ]
```

For humming mode (after PESTO + quantization), do the same conversion from
your detected pitch contour.

---

## Available keys and tempos

```python
from data.tokenizer import KEY_NAMES, TEMPO_BINS

KEY_NAMES        # 24 strings: "C", "C#", ..., "B", "Cm", "C#m", ..., "Bm"
TEMPO_BINS       # [60, 70, 80, ..., 180]
```

When picking a random round, sample `KEY_NAMES` and `TEMPO_BINS` directly.
The model has been trained on every combination so all 24 × 13 are valid.

---

## Quick smoke test (run this once after pulling the repo)

```bash
# After preprocessing and a brief sanity training:
python -c "
from inference.generate import load_model_for_inference, generate
from data.tokenizer import Note
from analysis.scoring import score_response

model, tok, dev = load_model_for_inference('checkpoints/sanity/last.pt')
call = [Note(60, 0, 4), Note(62, 4, 8), Note(64, 8, 12), Note(65, 12, 16)]
r = generate(model, tok, call, key='C', tempo=120, max_bars=4)
print('response notes:', r.response_notes)
print('score:', score_response(call, r.response_notes, key='C'))
"
```

If that prints a non-empty list of notes and a sensible score dict, the
pipeline is live.
