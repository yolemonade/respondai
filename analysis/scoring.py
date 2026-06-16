"""
Game scoring
============

This is the single entrypoint team B calls after each round::

    result = score_response(call_notes, response_notes, key="Dm")
    # → {'key_consistency', 'rhythm_similarity', 'motif_usage',
    #    'creativity_bonus', 'total', 'feedback'}

Score breakdown (per the spec)
------------------------------

* Key consistency (max 300): fraction of RESPONSE notes that belong to the
  CALL's diatonic scale.
* Rhythm similarity (max 300): Pearson correlation of onset vectors, rescaled
  from [-1, 1] to [0, 300].
* Motif usage (max 300): n-gram overlap of pitch intervals.
* Creativity bonus (max 100): granted when the response is *neither* a
  near-copy *nor* totally unrelated. We measure this as a soft penalty on the
  extremes of motif_usage.
"""
from __future__ import annotations

from typing import List, Sequence

from data.tokenizer import KEY_NAMES, Note

from .motif import motif_overlap_weighted
from .rhythm import rhythm_similarity


# -----------------------------------------------------------------------------
# Key consistency
# -----------------------------------------------------------------------------

# Pitch-classes (0..11) for each major scale rooted at C.
_MAJOR_INTERVALS = (0, 2, 4, 5, 7, 9, 11)
# Natural minor scale.
_MINOR_INTERVALS = (0, 2, 3, 5, 7, 8, 10)


def _key_pitch_classes(key: str) -> set[int]:
    """Set of pitch classes (mod 12) belonging to ``key``."""
    if key not in KEY_NAMES:
        raise ValueError(f"Unknown key: {key!r}")
    is_minor = key.endswith("m")
    root_name = key[:-1] if is_minor else key
    # Map root_name (sharps only) to pitch class.
    root_pc = {
        "C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5,
        "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11,
    }[root_name]
    intervals = _MINOR_INTERVALS if is_minor else _MAJOR_INTERVALS
    return {(root_pc + iv) % 12 for iv in intervals}


def key_consistency(notes: Sequence[Note], key: str) -> float:
    """Fraction of notes whose pitch-class is in ``key``."""
    if not notes:
        return 0.0
    in_key_pcs = _key_pitch_classes(key)
    in_key = sum(1 for n in notes if (n.pitch % 12) in in_key_pcs)
    return in_key / len(notes)


# -----------------------------------------------------------------------------
# Feedback messaging
# -----------------------------------------------------------------------------

def _feedback_message(
    *,
    key_score: int,
    rhythm_score: int,
    motif_score: int,
    creativity: int,
) -> str:
    """A single short sentence to display in the UI."""
    parts = []
    if motif_score >= 200:
        parts.append("Motif carried over!")
    elif motif_score < 60:
        parts.append("Could echo more of the call.")

    if key_score >= 240:
        parts.append("Strong key center.")
    elif key_score < 150:
        parts.append("Strayed from the key.")

    if rhythm_score >= 200:
        parts.append("Rhythm locked in.")
    elif rhythm_score < 80:
        parts.append("Rhythm went its own way.")

    if creativity >= 80:
        parts.append("Creative variation!")

    if not parts:
        parts.append("Solid response.")
    return " ".join(parts)


# -----------------------------------------------------------------------------
# Main scoring entrypoint
# -----------------------------------------------------------------------------

def score_response(
    call_notes: Sequence[Note],
    response_notes: Sequence[Note],
    key: str,
    *,
    num_bars: int = 4,
) -> dict:
    """Compute the round score.

    All sub-scores are returned as **integers in 0..max**, for direct display in
    the UI. ``total`` is the sum and is the headline number for that round.
    """
    # --- key consistency --------------------------------------------------
    kc = key_consistency(response_notes, key)  # 0..1
    key_score = int(round(kc * 300))

    # --- rhythm similarity ------------------------------------------------
    rs = rhythm_similarity(call_notes, response_notes, num_bars=num_bars)  # -1..1
    # Clamp negatives to 0; a negatively correlated rhythm is just "different",
    # not actively bad. Then scale.
    rs_clamped = max(0.0, rs)
    rhythm_score = int(round(rs_clamped * 300))

    # --- motif usage ------------------------------------------------------
    # n=2 (단일 인터벌) 로 느슨하게 매칭 + sqrt 커브로 낮은 점수 구간 부스트
    mo = motif_overlap_weighted(call_notes, response_notes, n=2)  # 0..1
    motif_score = int(round(mo ** 0.5 * 300))

    # --- creativity bonus -------------------------------------------------
    # 문턱을 0.03으로 낮추고 구간을 넓혀 0이 나오는 경우를 줄임
    if 0.03 <= mo <= 0.97:
        if mo <= 0.45:
            tri = (mo - 0.03) / (0.45 - 0.03)
        else:
            tri = (0.97 - mo) / (0.97 - 0.45)
        creativity = int(round(max(0.0, tri) * 100))
    else:
        creativity = 0

    total = key_score + rhythm_score + motif_score + creativity

    return {
        "key_consistency": key_score,
        "rhythm_similarity": rhythm_score,
        "motif_usage": motif_score,
        "creativity_bonus": creativity,
        "total": total,
        # Raw (un-rescaled) values, useful for plotting trends across rounds.
        "raw": {
            "key_consistency": kc,
            "rhythm_pearson": rs,
            "motif_overlap": mo,
        },
        "feedback": _feedback_message(
            key_score=key_score,
            rhythm_score=rhythm_score,
            motif_score=motif_score,
            creativity=creativity,
        ),
    }


# -----------------------------------------------------------------------------
# Session-level aggregation (for the final results screen)
# -----------------------------------------------------------------------------

def grade_from_total(total: int) -> str:
    """Map a cumulative 2-round score (max 2000) to a letter grade."""
    if total >= 1800:
        return "S"
    if total >= 1400:
        return "A"
    if total >= 1000:
        return "B"
    return "C"


def session_summary(round_results: Sequence[dict]) -> dict:
    """Aggregate a list of round results into a final report.

    Applies the bonuses described in the spec:
      * +100 if both rounds are above 700.
      * +200 if every round's response stayed in the same key (we approximate
        this as ``raw.key_consistency >= 0.8`` in every round).

    R5's "include R1 motif" bonus is *not* applied here because we don't carry
    R1's motif through the function signature; the game engine should add it.
    """
    base = sum(r["total"] for r in round_results)
    bonus = 0

    # Combo: 두 라운드 모두 700 이상이면 보너스.
    if all(r["total"] >= 700 for r in round_results):
        bonus += 100

    # All-rounds in key.
    if all(r["raw"]["key_consistency"] >= 0.8 for r in round_results):
        bonus += 200

    total = base + bonus
    return {
        "rounds": list(round_results),
        "base_score": base,
        "bonus_score": bonus,
        "total": total,
        "grade": grade_from_total(total),
    }
