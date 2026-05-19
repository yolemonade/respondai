"""
Preprocess a MIDI directory into a tokenized cache file.

Example:
    python -m scripts.run_preprocess --midi-root datasets/nottingham --out data_cache/nottingham.npz
    python -m scripts.run_preprocess --midi-root datasets/lakh --out data_cache/lakh.npz
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from data.preprocess import build_cache
from data.tokenizer import Tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--midi-root", required=True, help="Directory of MIDI files (recursive).")
    parser.add_argument("--out", required=True, help="Output .npz cache path.")
    parser.add_argument("--max-len", type=int, default=512,
                        help="Drop pairs longer than this many tokens.")
    parser.add_argument("--window-bars", type=int, default=4)
    parser.add_argument("--stride-bars", type=int, default=2)
    parser.add_argument("--quiet", action="store_true", help="Suppress tqdm progress bar.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    tokenizer = Tokenizer()
    summary = build_cache(
        midi_root=args.midi_root,
        output_path=args.out,
        tokenizer=tokenizer,
        max_seq_len=args.max_len,
        window_bars=args.window_bars,
        stride_bars=args.stride_bars,
        progress=not args.quiet,
    )
    print(summary)


if __name__ == "__main__":
    main()
