"""
Pitch-range diagnostics
========================

"왜 62~69만 나오나"의 원인을 데이터 vs 모델로 분리해 확인한다.

1) --cache : 학습 캐시(.npz)의 RESPONSE pitch 분포 — 데이터가 원래 좁은가?
2) --checkpoint : 다양한 음역의 CALL 로 생성해 모델 출력 음역 측정

사용 예:
    python -m scripts.pitch_diagnostics --cache data_cache/lakh_train.npz
    python -m scripts.pitch_diagnostics --checkpoint checkpoints/best_inference.pt
"""
from __future__ import annotations

import argparse
import pickle
from collections import Counter
from pathlib import Path

import numpy as np

from data.tokenizer import Tokenizer, PITCH_MIN


NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _name(midi: int) -> str:
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def _histogram(midis, title):
    if not midis:
        print(f"\n[{title}] (데이터 없음)")
        return
    lo, hi = min(midis), max(midis)
    print(f"\n[{title}]")
    print(f"  음역: {_name(lo)}({lo}) ~ {_name(hi)}({hi})  "
          f"= {hi - lo} semitone ({(hi - lo) / 12:.1f} 옥타브)")
    print(f"  중앙값: {_name(int(np.median(midis)))}  "
          f"표준편차: {np.std(midis):.1f} semitone")
    counter = Counter(midis)
    top = counter.most_common(8)
    total = len(midis)
    print("  최빈 음 (상위 8):")
    for m, c in top:
        bar = "█" * int(c / total * 60)
        print(f"    {_name(m):4s}({m:3d}): {c / total * 100:5.1f}% {bar}")


def _from_cache(path: Path):
    tok = Tokenizer()
    p_lo, p_hi = tok.VOCAB_LAYOUT["pitches"]
    with np.load(path, allow_pickle=True) as z:
        tokens = z["tokens"]
        offsets = z["offsets"]
    n = len(offsets) - 1
    resp_midis = []
    for i in range(n):
        ids = tokens[offsets[i]:offsets[i + 1]].tolist()
        try:
            r = ids.index(tok.response_id)
        except ValueError:
            continue
        for t in ids[r + 1:]:
            if p_lo <= t < p_hi:
                resp_midis.append(t - tok.pitch_offset + PITCH_MIN)
    _histogram(resp_midis, f"학습 캐시 RESPONSE 음역 — {path.name} ({n:,} pairs)")
    print("\n→ 데이터가 이미 좁으면(예: 1~1.5 옥타브, 표준편차<5) 모델 잘못이"
          " 아니라 데이터 분포 문제입니다.")


def _from_model(checkpoint: str):
    from data.tokenizer import Note
    from inference.generate import (
        load_model_for_inference, generate_candidates,
    )
    model, tok, dev = load_model_for_inference(checkpoint)

    # 음역이 서로 다른 4개의 CALL 로 모델 반응 폭을 본다.
    calls = {
        "좁은 D4-A4": [Note(62, 0, 4), Note(65, 4, 6), Note(69, 8, 12),
                       Note(67, 12, 14), Note(64, 16, 20)],
        "넓은 G3-C5": [Note(55, 0, 4), Note(60, 4, 8), Note(67, 8, 12),
                       Note(72, 12, 16), Note(59, 16, 20)],
        "저음 C3-C4": [Note(48, 0, 4), Note(52, 4, 8), Note(55, 8, 12),
                       Note(60, 12, 16), Note(50, 16, 20)],
        "고음 C5-C6": [Note(72, 0, 4), Note(76, 4, 8), Note(79, 8, 12),
                       Note(84, 12, 16), Note(74, 16, 20)],
    }
    for label, call in calls.items():
        out_midis = []
        for cand in generate_candidates(model, tok, call, key="Dm", tempo=92):
            out_midis.extend(n.pitch for n in cand)
        call_lo = min(n.pitch for n in call)
        call_hi = max(n.pitch for n in call)
        _histogram(
            out_midis,
            f"모델 출력 — CALL '{label}' "
            f"({_name(call_lo)}~{_name(call_hi)})",
        )
    print("\n→ 넓은/저음/고음 CALL 에도 출력이 62~69로 수렴하면 모델이 음역을"
          " 외운 것(데이터 편향). CALL 따라 음역이 움직이면 정상입니다.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache")
    parser.add_argument("--checkpoint")
    args = parser.parse_args()

    if args.cache:
        _from_cache(Path(args.cache))
    if args.checkpoint:
        _from_model(args.checkpoint)
    if not args.cache and not args.checkpoint:
        print("--cache 또는 --checkpoint 중 하나를 지정하세요.")


if __name__ == "__main__":
    main()