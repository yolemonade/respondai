"""
Inference latency benchmark — UI 없이 생성 시간만 측정.

RunPod/맥 터미널에서 바로 실행해 단일 생성 vs 배치 4후보 시간을 비교한다:

    python -m scripts.bench_inference --checkpoint checkpoints/best_inference.pt

옵션:
    --runs 5            반복 횟수 (기본 5; 첫 회는 워밍업으로 별도 표기)
    --device auto       auto | cuda | mps | cpu
    --candidates 4      배치 후보 수
"""
from __future__ import annotations

import argparse
import time

from data.tokenizer import Note
from inference.generate import (
    generate,
    generate_candidates,
    load_model_for_inference,
)

# 4마디짜리 대표 CALL (D minor, 16분음표 그리드)
SAMPLE_CALL = [
    Note(62, 0, 4), Note(65, 4, 6), Note(69, 8, 12), Note(67, 12, 14),
    Note(65, 16, 20), Note(64, 20, 22), Note(62, 24, 28), Note(65, 28, 30),
    Note(67, 32, 36), Note(69, 36, 38), Note(70, 40, 44), Note(69, 44, 46),
    Note(67, 48, 52), Note(65, 52, 54), Note(62, 56, 62),
]


def _bench(label: str, fn, runs: int) -> None:
    times = []
    for i in range(runs):
        t0 = time.perf_counter()
        out = fn()
        dt = time.perf_counter() - t0
        times.append(dt)
        extra = ""
        if isinstance(out, list) and out and isinstance(out[0], list):
            extra = f"  notes/cand={[len(c) for c in out]}"
        elif hasattr(out, "response_notes"):
            extra = f"  notes={len(out.response_notes)}  stop={out.stop_reason}"
        tag = " (warmup)" if i == 0 else ""
        print(f"  [{label}] run {i + 1}: {dt * 1000:7.1f} ms{tag}{extra}")
    steady = times[1:] or times
    print(f"  [{label}] steady avg: {sum(steady) / len(steady) * 1000:.1f} ms"
          f"  (min {min(steady) * 1000:.1f} / max {max(steady) * 1000:.1f})\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/best_inference.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--candidates", type=int, default=4)
    args = parser.parse_args()

    print(f"loading {args.checkpoint} ...")
    model, tokenizer, device = load_model_for_inference(
        args.checkpoint, device=args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device}  params={n_params:,}\n")

    _bench(
        "single generate",
        lambda: generate(
            model, tokenizer, SAMPLE_CALL,
            key="Dm", tempo=92, max_bars=4,
            temperature=0.95, return_attention=False,
        ),
        args.runs,
    )
    _bench(
        f"batch x{args.candidates}",
        lambda: generate_candidates(
            model, tokenizer, SAMPLE_CALL,
            key="Dm", tempo=92, max_bars=4,
            num_candidates=args.candidates,
        ),
        args.runs,
    )


if __name__ == "__main__":
    main()
