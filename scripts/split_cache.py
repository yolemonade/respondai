"""
Split a tokenized cache (.npz) into train / val.

best.pt 는 val_loss 가 개선될 때만 저장되므로, val 캐시 없이 학습하면
best 체크포인트가 만들어지지 않는다. 학습 전에 한 번 실행:

    python -m scripts.split_cache \
        --cache data_cache/lakh.npz \
        --val-ratio 0.02

→ data_cache/lakh_train.npz, data_cache/lakh_val.npz 생성.
원본 lakh.npz 는 건드리지 않는다.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


def _load(path: Path) -> tuple[np.ndarray, np.ndarray, list, int]:
    with np.load(path, allow_pickle=True) as z:
        tokens = z["tokens"]
        offsets = z["offsets"]
        meta = pickle.loads(z["meta"].item())
        vocab_size = int(z["vocab_size"])
    return tokens, offsets, meta, vocab_size


def _save(path: Path, seqs: list[np.ndarray], metas: list[dict], vocab_size: int) -> None:
    offsets = np.zeros(len(seqs) + 1, dtype=np.int64)
    for i, s in enumerate(seqs):
        offsets[i + 1] = offsets[i] + s.size
    tokens = (
        np.concatenate(seqs).astype(np.int32)
        if seqs else np.zeros(0, dtype=np.int32)
    )
    np.savez(
        path,
        tokens=tokens,
        offsets=offsets,
        meta=np.array(pickle.dumps(metas), dtype=object),
        vocab_size=np.int64(vocab_size),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, help="원본 .npz (예: data_cache/lakh.npz)")
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    src = Path(args.cache)
    tokens, offsets, meta, vocab_size = _load(src)
    n = len(offsets) - 1
    assert len(meta) == n, f"meta({len(meta)}) != sequences({n})"

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(n)
    n_val = max(1, int(n * args.val_ratio))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    def take(indices):
        seqs = [tokens[offsets[i]:offsets[i + 1]] for i in indices]
        metas = [meta[i] for i in indices]
        return seqs, metas

    stem = src.stem  # "lakh"
    train_path = src.with_name(f"{stem}_train.npz")
    val_path = src.with_name(f"{stem}_val.npz")
    _save(train_path, *take(train_idx), vocab_size)
    _save(val_path, *take(val_idx), vocab_size)
    print(f"total {n:,} pairs  →  train {len(train_idx):,} ({train_path})")
    print(f"                      val   {len(val_idx):,} ({val_path})")


if __name__ == "__main__":
    main()
