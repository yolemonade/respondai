# RespondAI — Call-and-Response Improv Game

A small Decoder-only Transformer that answers your 4-bar melody with its own,
trained from scratch on the Lakh MIDI Dataset.

This repo is the **model-side codebase** (team A's deliverable). For the
Streamlit game UI, see team B.

---

## Repo layout

```
respondai/
├── data/
│   ├── tokenizer.py         REMI-style tokenizer (vocab ≈ 180)
│   ├── midi_utils.py        MIDI → monophonic Note sequence
│   ├── call_response.py     Sliding-window pair generation + key estimation
│   ├── preprocess.py        Build the binary token cache
│   └── dataset.py           PyTorch Dataset + RESPONSE-only loss mask
├── model/
│   ├── attention.py         Causal multi-head self-attention (from scratch)
│   ├── positional.py        Sinusoidal positional encoding
│   ├── transformer.py       Decoder-only Transformer (Pre-LN, KV cache)
│   └── config.py            sanity_config (~3M) / full_config (~10M)
├── training/
│   ├── train.py             AdamW + warmup-cosine, AMP, checkpointing
│   └── evaluate.py          Perplexity + sample-level game-score eval
├── inference/
│   ├── generate.py          generate(): top-p sampling with KV cache  ← team B uses this
│   └── decode.py            Notes → MIDI / WAV                        ← team B uses this
├── analysis/
│   ├── motif.py             n-gram interval overlap
│   ├── rhythm.py            Onset-vector Pearson correlation
│   └── scoring.py           score_response()                          ← team B uses this
├── scripts/
│   ├── download_data.sh
│   ├── run_preprocess.py
│   ├── run_sanity.py
│   └── run_train.py
├── INTERFACE.md             Public API for team B (read this first if you're B)
├── requirements.txt
└── README.md
```

---

## Quick start

```bash
# 1. Environment
pip install -r requirements.txt

# 2. Data
bash scripts/download_data.sh nottingham      # small + instant
bash scripts/download_data.sh lakh            # large, for the real run

# 3. Tokenize
python -m scripts.run_preprocess \
    --midi-root datasets/nottingham \
    --out data_cache/nottingham.npz

# 4. Sanity-check the pipeline (a few minutes on any GPU)
python -m scripts.run_sanity \
    --cache data_cache/nottingham.npz \
    --steps 2000

# 5. Full training (a few hours on a single A100 / Colab T4)
python -m scripts.run_preprocess \
    --midi-root datasets/lakh \
    --out data_cache/lakh.npz
python -m scripts.run_train \
    --cache data_cache/lakh.npz \
    --steps 20000
```

---

## What "good" looks like

| Stage                          | Expected outcome                                                |
|--------------------------------|-----------------------------------------------------------------|
| Sanity, step 0                 | loss ≈ ln(180) ≈ 5.2 (uniform over vocab)                       |
| Sanity, step 2 000             | loss < 1.5 on Nottingham                                        |
| Full run, end                  | val PPL < 10 on Lakh held-out (target from spec)                |
| Inference on Mac MPS           | < 1 s for a 4-bar response (~80 generated tokens)               |
| Inference on CUDA A100         | < 0.3 s                                                          |

If sanity loss plateaus near 5 instead of falling: the loss mask is wrong.
Check that `[RESPONSE]` is being found in every sequence and that the mask
covers positions *after* it.

---

## Design decisions (the short version)

| Decision              | Choice                                  | Why                                                |
|-----------------------|------------------------------------------|----------------------------------------------------|
| Tokenization          | REMI-style: `[BAR] [POS] [PITCH] [DUR]`  | Bar-aware; compact; transposition-friendly         |
| Pitch range           | MIDI 21–108 (88 piano keys)              | Drops unused extremes; smaller vocabulary          |
| Rest representation   | Implicit (POS jump or BAR token)         | No separate rest vocab; shorter sequences          |
| Key/Tempo prefix      | Yes (`[KEY_*] [TEMPO_*]` before [CALL])  | Direct conditioning matches the game's UX          |
| Velocity              | Not modelled                             | Monophonic melody; velocity adds noise, not signal |
| Positional encoding   | Sinusoidal (Vaswani 2017)                | Matches the lecture; works fine at this scale      |
| Block layout          | Pre-LN                                   | Trains far more stably than Post-LN                |
| Sampling              | top-p (0.9) + temperature (1.0)          | Most musical default; greedy collapses             |
| Loss masking          | Score RESPONSE only                      | CALL is context, not target                        |
| Key estimation        | music21 Krumhansl-Schmuckler             | Pure Python; robust for 4-bar fragments            |

---

## Coordination with team B

The contract is **INTERFACE.md**. Three functions are the only public surface:

* `inference.generate.generate(model, tokenizer, call_notes, key, tempo, ...)`
* `inference.decode.decode_tokens_to_notes(tokens, tokenizer)`
* `analysis.scoring.score_response(call_notes, response_notes, key)`

These are pure and synchronous. Cache the model in
`@st.cache_resource` and you'll never wait for it again.

---

## Known limits (be honest in the demo)

* The model is trained on Western tonal monophony; non-Western scales will be
  scored harshly by `key_consistency` because the scale definitions
  only cover major and natural minor.
* `decode_tokens_to_notes` is lenient — malformed token streams produce empty
  outputs rather than raising. If you see an empty response unexpectedly, it's
  the model, not the decoder.
* `notes_to_wav` requires FluidSynth to be installed system-wide. On Mac:
  `brew install fluid-synth`. On Linux: `apt-get install fluidsynth`.
