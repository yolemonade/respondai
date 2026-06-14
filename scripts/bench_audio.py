"""
FluidSynth vs synth_notes 벤치 (Mac 등 실제 환경에서 실행)
=========================================================

이 환경엔 fluidsynth 바이너리가 없어 직접 못 잰다. Mac 에서 실행해
FluidSynth 렌더가 실시간 목표(약 0.3초 이내) 안에 드는지 확인한다.

준비 (Mac):
    brew install fluid-synth
    pip install pretty_midi
    # SoundFont 하나 — 아래 중 하나를 받아 경로 지정:
    #   GeneralUser GS, FluidR3_GM.sf2, 또는 Salamander(피아노 전용)

실행:
    python -m scripts.bench_audio --sf2 /path/to/FluidR3_GM.sf2
    python -m scripts.bench_audio          # sf2 생략 시 pretty_midi 기본

해석:
    FluidSynth avg 가 ~150ms 이하면 채택 권장(품질↑, 실시간 OK).
    ~300ms 이상이면 UI 가 답답해질 수 있음 → 반주(synth_notes) 유지.
"""
from __future__ import annotations

import argparse
import time
from typing import List

from data.tokenizer import Note

# 4마디, 15음 대표 프레이즈 (D minor)
SAMPLE: List[Note] = [
    Note(62, 0, 4), Note(65, 4, 6), Note(69, 8, 12), Note(67, 12, 14),
    Note(65, 16, 20), Note(64, 20, 22), Note(62, 24, 28), Note(65, 28, 30),
    Note(67, 32, 36), Note(69, 36, 38), Note(70, 40, 44), Note(69, 44, 46),
    Note(67, 48, 52), Note(65, 52, 54), Note(62, 56, 62),
]


def _bench(label: str, fn, runs: int) -> float:
    times = []
    for i in range(runs):
        t0 = time.perf_counter()
        audio = fn()
        dt = time.perf_counter() - t0
        times.append(dt)
        tag = " (warmup)" if i == 0 else ""
        n = len(audio) if hasattr(audio, "__len__") else 0
        print(f"  [{label}] run {i + 1}: {dt * 1000:7.1f} ms{tag}  samples={n}")
    steady = times[1:] or times
    avg = sum(steady) / len(steady)
    print(f"  [{label}] steady avg: {avg * 1000:.1f} ms\n")
    return avg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sf2", default=None, help=".sf2 SoundFont 경로")
    parser.add_argument("--runs", type=int, default=6)
    parser.add_argument("--bpm", type=int, default=92)
    parser.add_argument("--sr", type=int, default=22050)
    args = parser.parse_args()

    print(f"notes={len(SAMPLE)}  bpm={args.bpm}  sr={args.sr}\n")

    # 1) 기존 NumPy 합성 (현재 사용 중)
    from input.piano import synth_notes
    synth_avg = _bench(
        "synth_notes (현재)",
        lambda: synth_notes(SAMPLE, bpm=args.bpm, sample_rate=args.sr,
                            role="ai", swing_amount=0.6, humanize_ms=8.0),
        args.runs,
    )

    # 2) FluidSynth 렌더
    fluid_avg = None
    try:
        from inference.decode import notes_to_wav
        fluid_avg = _bench(
            "FluidSynth",
            lambda: notes_to_wav(SAMPLE, tempo=args.bpm,
                                 sample_rate=args.sr, sound_font=args.sf2),
            args.runs,
        )
    except Exception as exc:
        print(f"  [FluidSynth] 실행 불가: {exc}")
        print("  → fluid-synth 설치 + pretty_midi + .sf2 경로를 확인하세요.\n")

    # 판정
    print("─" * 50)
    print(f"synth_notes : {synth_avg * 1000:.0f} ms")
    if fluid_avg is not None:
        print(f"FluidSynth  : {fluid_avg * 1000:.0f} ms")
        # build_round_audio 는 user+ai 2회 합성하므로 약 2배로 추정
        est_round = fluid_avg * 2 * 1000
        print(f"→ 라운드 1회(user+ai 2회 합성) 추정: ~{est_round:.0f} ms")
        if fluid_avg < 0.15:
            print("✅ 빠름 — FluidSynth 채택 권장 (품질↑, 실시간 OK)")
        elif fluid_avg < 0.30:
            print("⚠️  애매 — 채택 가능하나 user 합성 병렬화 권장")
        else:
            print("❌ 느림 — 반주(synth_notes) 유지 권장")
    print("─" * 50)


if __name__ == "__main__":
    main()
