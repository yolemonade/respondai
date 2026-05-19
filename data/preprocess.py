"""
Dataset preprocessing
=====================

Walk a directory of MIDI files, build all Call-and-Response pairs we can find,
and serialize the resulting token sequences to a single binary cache file.

Why a single file?
  * One ``np.memmap`` over the cache is the fastest possible DataLoader.
  * Avoids the file-system load of 100k small pickle files (Lakh MIDI).
  * Reproducible: re-running with the same args produces the same file.

Cache file format (``.npz``):
    tokens   : int32 array, flat concatenation of all sequences
    offsets  : int64 array, length N+1; sequence i is tokens[offsets[i]:offsets[i+1]]
    meta     : pickled list of dicts, one per sequence (key, tempo, source_path)
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
from tqdm import tqdm

from .call_response import CRPair, generate_pairs
from .midi_utils import MidiParseError, midi_to_monophonic_notes
from .tokenizer import Tokenizer

log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Iteration over a corpus
# -----------------------------------------------------------------------------

def iter_midi_files(root: str | Path, suffixes: Sequence[str] = (".mid", ".midi")) -> Iterable[Path]:
    """Yield every MIDI file under ``root`` (recursive)."""
    root = Path(root)
    for suf in suffixes:
        yield from root.rglob(f"*{suf}")


def pairs_for_file(
    path: Path,
    *,
    window_bars: int = 4,
    stride_bars: int = 2,
) -> List[CRPair]:
    """All ``CRPair``s for a single MIDI file. Returns ``[]`` on parse errors."""
    try:
        notes, bpm = midi_to_monophonic_notes(path, return_bpm=True)
    except MidiParseError:
        return []
    return list(
        generate_pairs(
            notes,
            tempo=int(round(bpm)),
            window_bars=window_bars,
            stride_bars=stride_bars,
        )
    )


# -----------------------------------------------------------------------------
# Building & saving the cache
# -----------------------------------------------------------------------------

def build_cache(
    midi_root: str | Path,
    output_path: str | Path,
    tokenizer: Tokenizer,
    *,
    max_seq_len: int = 512,
    window_bars: int = 4,
    stride_bars: int = 2,
    progress: bool = True,
) -> dict:
    """Build a tokenized cache from a directory of MIDI files.

    Returns a small summary dict (counts, lengths).
    """
    midi_root = Path(midi_root)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    files = list(iter_midi_files(midi_root))
    if not files:
        raise FileNotFoundError(f"No MIDI files under {midi_root}")

    log.info("Scanning %d MIDI files under %s", len(files), midi_root)

    all_tokens: List[int] = []
    offsets: List[int] = [0]
    meta: List[dict] = []
    n_pairs = 0
    n_truncated = 0
    n_too_long_dropped = 0

    iterator = tqdm(files, desc="Preprocessing", disable=not progress)
    for path in iterator:
        for pair in pairs_for_file(
            path,
            window_bars=window_bars,
            stride_bars=stride_bars,
        ):
            tokens = tokenizer.build_pair(
                call_notes=pair.call,
                response_notes=pair.response,
                key=pair.key,
                tempo=pair.tempo,
            )

            if len(tokens) > max_seq_len:
                # The CALL alone is informative even if the RESPONSE is long;
                # however arbitrary mid-stream truncation produces malformed
                # pairs. Drop the whole pair rather than corrupt it.
                n_too_long_dropped += 1
                continue

            all_tokens.extend(tokens)
            offsets.append(len(all_tokens))
            meta.append(
                {
                    "key": pair.key,
                    "tempo": pair.tempo,
                    "source": str(path.relative_to(midi_root)),
                }
            )
            n_pairs += 1

    if n_pairs == 0:
        raise RuntimeError(f"Found no usable pairs under {midi_root}")

    tokens_np = np.asarray(all_tokens, dtype=np.int32)
    offsets_np = np.asarray(offsets, dtype=np.int64)

    np.savez(
        output_path,
        tokens=tokens_np,
        offsets=offsets_np,
        meta=np.array(pickle.dumps(meta), dtype=object),
        vocab_size=np.int64(tokenizer.vocab_size),
    )

    summary = {
        "num_pairs": n_pairs,
        "num_files_scanned": len(files),
        "num_tokens": int(tokens_np.size),
        "avg_len": float(tokens_np.size / max(n_pairs, 1)),
        "max_len": int(np.max(offsets_np[1:] - offsets_np[:-1])) if n_pairs else 0,
        "dropped_too_long": n_too_long_dropped,
        "truncated": n_truncated,
        "output": str(output_path),
    }
    log.info("Cache built: %s", summary)
    return summary


def load_cache(path: str | Path) -> dict:
    """Mirror of :func:`build_cache` output. Returns numpy arrays + meta list."""
    data = np.load(path, allow_pickle=True)
    return {
        "tokens": data["tokens"],
        "offsets": data["offsets"],
        "meta": pickle.loads(bytes(data["meta"].item())),
        "vocab_size": int(data["vocab_size"]),
    }
