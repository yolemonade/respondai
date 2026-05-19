"""
RespondAI Tokenizer
===================

REMI-style tokenizer for Call-and-Response symbolic music generation.

Token vocabulary
----------------
Indices are allocated in contiguous ranges; see ``Tokenizer.VOCAB_LAYOUT`` for
the authoritative source. Roughly:

    [PAD] [BOS] [EOS] [CALL] [SEP] [RESPONSE] [BAR]
    [KEY_*]   24 keys (12 major + 12 minor)
    [TEMPO_*] 12 tempo bins (60..180 BPM, step 10)
    [POS_*]   16 sixteenth-note positions within a bar
    [PITCH_*] 88 pitches (MIDI 21..108, piano range)
    [DUR_*]   32 durations (1..32 sixteenth-notes)

A monophonic note event is represented as a (POS, PITCH, DUR) triple. Bars are
delimited by a BAR token. Rests are *not* tokenized explicitly: the next
POS / BAR token implies any silence in between, which keeps the vocabulary
compact and the sequences short.

Full sequence layout for one training pair::

    [BOS] [KEY_x] [TEMPO_y]
        [CALL]
            [BAR] [POS_0] [PITCH_p] [DUR_d] [POS_4] [PITCH_p] [DUR_d] ...
            [BAR] ...
        [SEP] [RESPONSE]
            [BAR] [POS_0] [PITCH_p] [DUR_d] ...
            [BAR] ...
        [EOS]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


# -----------------------------------------------------------------------------
# Domain constants
# -----------------------------------------------------------------------------

#: Lowest / highest MIDI pitch we model (piano range, 88 keys).
PITCH_MIN, PITCH_MAX = 21, 108
NUM_PITCHES = PITCH_MAX - PITCH_MIN + 1  # 88

#: Sixteenth-note grid resolution per bar (4/4 only for now).
STEPS_PER_BAR = 16

#: Duration range in sixteenth-note steps (1 = sixteenth, 32 = two bars).
DUR_MIN, DUR_MAX = 1, 32
NUM_DURS = DUR_MAX - DUR_MIN + 1  # 32

#: Tempo bins (BPM), inclusive endpoints.
TEMPO_MIN, TEMPO_MAX, TEMPO_STEP = 60, 180, 10
TEMPO_BINS = list(range(TEMPO_MIN, TEMPO_MAX + 1, TEMPO_STEP))  # 13 bins
NUM_TEMPOS = len(TEMPO_BINS)

#: All 24 keys in a fixed canonical order.
KEY_NAMES_MAJOR = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
KEY_NAMES_MINOR = [f"{n}m" for n in KEY_NAMES_MAJOR]
KEY_NAMES = KEY_NAMES_MAJOR + KEY_NAMES_MINOR  # 24 entries
NUM_KEYS = len(KEY_NAMES)


# -----------------------------------------------------------------------------
# Lightweight Note dataclass shared across the codebase
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Note:
    """A single monophonic note on the sixteenth-note grid.

    Times are measured in sixteenth-note steps from the start of the segment.
    ``end`` is exclusive (i.e. ``end - start == duration_in_steps``).
    """

    pitch: int       # MIDI pitch, PITCH_MIN..PITCH_MAX
    start: int       # sixteenth-note step, >= 0
    end: int         # sixteenth-note step, > start

    @property
    def duration(self) -> int:
        return self.end - self.start


# -----------------------------------------------------------------------------
# Tokenizer
# -----------------------------------------------------------------------------

class Tokenizer:
    """REMI-style tokenizer for monophonic Call-and-Response pairs.

    The class is *stateless* with respect to song data; all vocabulary is
    determined at construction time and never grows. This makes the tokenizer
    safe to share across processes (e.g. DataLoader workers).
    """

    # Special tokens, in fixed order. Do not reorder once the model is trained.
    SPECIALS: Tuple[str, ...] = (
        "[PAD]",
        "[BOS]",
        "[EOS]",
        "[CALL]",
        "[SEP]",
        "[RESPONSE]",
        "[BAR]",
    )

    def __init__(self) -> None:
        self._id_to_token: List[str] = []
        self._token_to_id: dict[str, int] = {}

        # Allocate ranges. Order matters: it defines the layout.
        for tok in self.SPECIALS:
            self._add(tok)

        self.key_offset = len(self._id_to_token)
        for k in KEY_NAMES:
            self._add(f"[KEY_{k}]")

        self.tempo_offset = len(self._id_to_token)
        for t in TEMPO_BINS:
            self._add(f"[TEMPO_{t}]")

        self.pos_offset = len(self._id_to_token)
        for p in range(STEPS_PER_BAR):
            self._add(f"[POS_{p}]")

        self.pitch_offset = len(self._id_to_token)
        for p in range(PITCH_MIN, PITCH_MAX + 1):
            self._add(f"[PITCH_{p}]")

        self.dur_offset = len(self._id_to_token)
        for d in range(DUR_MIN, DUR_MAX + 1):
            self._add(f"[DUR_{d}]")

        # Cache frequent ids for speed.
        self.pad_id = self._token_to_id["[PAD]"]
        self.bos_id = self._token_to_id["[BOS]"]
        self.eos_id = self._token_to_id["[EOS]"]
        self.call_id = self._token_to_id["[CALL]"]
        self.sep_id = self._token_to_id["[SEP]"]
        self.response_id = self._token_to_id["[RESPONSE]"]
        self.bar_id = self._token_to_id["[BAR]"]

    def _add(self, tok: str) -> int:
        idx = len(self._id_to_token)
        self._id_to_token.append(tok)
        self._token_to_id[tok] = idx
        return idx

    # -- introspection ---------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self._id_to_token)

    @property
    def VOCAB_LAYOUT(self) -> dict:
        """Authoritative description of the token index layout."""
        return {
            "specials": (0, len(self.SPECIALS)),
            "keys": (self.key_offset, self.key_offset + NUM_KEYS),
            "tempos": (self.tempo_offset, self.tempo_offset + NUM_TEMPOS),
            "positions": (self.pos_offset, self.pos_offset + STEPS_PER_BAR),
            "pitches": (self.pitch_offset, self.pitch_offset + NUM_PITCHES),
            "durations": (self.dur_offset, self.dur_offset + NUM_DURS),
            "total": self.vocab_size,
        }

    def id_to_token(self, idx: int) -> str:
        return self._id_to_token[idx]

    def token_to_id(self, tok: str) -> int:
        return self._token_to_id[tok]

    # -- typed accessors -------------------------------------------------------

    def key_token(self, key: str) -> int:
        if key not in KEY_NAMES:
            raise ValueError(f"Unknown key: {key!r}. Expected one of {KEY_NAMES}.")
        return self.key_offset + KEY_NAMES.index(key)

    def tempo_token(self, bpm: int) -> int:
        # Snap to nearest bin, clamp to range.
        bpm = max(TEMPO_MIN, min(TEMPO_MAX, int(bpm)))
        idx = round((bpm - TEMPO_MIN) / TEMPO_STEP)
        return self.tempo_offset + idx

    def position_token(self, pos: int) -> int:
        if not 0 <= pos < STEPS_PER_BAR:
            raise ValueError(f"Position {pos} out of range [0, {STEPS_PER_BAR}).")
        return self.pos_offset + pos

    def pitch_token(self, pitch: int) -> int:
        if not PITCH_MIN <= pitch <= PITCH_MAX:
            raise ValueError(
                f"Pitch {pitch} out of range [{PITCH_MIN}, {PITCH_MAX}]."
            )
        return self.pitch_offset + (pitch - PITCH_MIN)

    def duration_token(self, dur: int) -> int:
        if dur < DUR_MIN:
            dur = DUR_MIN
        elif dur > DUR_MAX:
            dur = DUR_MAX
        return self.dur_offset + (dur - DUR_MIN)

    # -- range queries (used by sampling masks) --------------------------------

    def is_pitch(self, idx: int) -> bool:
        return self.pitch_offset <= idx < self.pitch_offset + NUM_PITCHES

    def is_duration(self, idx: int) -> bool:
        return self.dur_offset <= idx < self.dur_offset + NUM_DURS

    def is_position(self, idx: int) -> bool:
        return self.pos_offset <= idx < self.pos_offset + STEPS_PER_BAR

    def is_special(self, idx: int) -> bool:
        return 0 <= idx < len(self.SPECIALS)

    def pitch_value(self, idx: int) -> int:
        """Inverse of ``pitch_token``."""
        return PITCH_MIN + (idx - self.pitch_offset)

    def duration_value(self, idx: int) -> int:
        return DUR_MIN + (idx - self.dur_offset)

    def position_value(self, idx: int) -> int:
        return idx - self.pos_offset

    # -- encoding --------------------------------------------------------------

    def encode_notes(self, notes: Sequence[Note]) -> List[int]:
        """Encode a sequence of notes into REMI tokens (no special wrappers).

        Notes must be:
          * monophonic and non-overlapping
          * sorted by ``start`` (not enforced; caller responsibility)
          * within pitch range (silently clipped)

        Bars are emitted on every crossing of a 16-step boundary. Notes that
        fall in the same bar share a [BAR] token.
        """
        tokens: List[int] = []
        if not notes:
            return tokens

        cur_bar = -1
        for n in notes:
            bar = n.start // STEPS_PER_BAR
            # Emit bar tokens for every bar from cur_bar+1 up to ``bar``.
            while cur_bar < bar:
                tokens.append(self.bar_id)
                cur_bar += 1

            pos_in_bar = n.start % STEPS_PER_BAR
            tokens.append(self.position_token(pos_in_bar))
            pitch = max(PITCH_MIN, min(PITCH_MAX, n.pitch))
            tokens.append(self.pitch_token(pitch))
            tokens.append(self.duration_token(n.duration))
        return tokens

    def build_pair(
        self,
        call_notes: Sequence[Note],
        response_notes: Sequence[Note],
        key: str,
        tempo: int,
    ) -> List[int]:
        """Assemble a full training sample: [BOS] [KEY] [TEMPO] [CALL] ... [SEP] [RESPONSE] ... [EOS]."""
        out: List[int] = [
            self.bos_id,
            self.key_token(key),
            self.tempo_token(tempo),
            self.call_id,
        ]
        out.extend(self.encode_notes(call_notes))
        out.append(self.sep_id)
        out.append(self.response_id)
        out.extend(self.encode_notes(response_notes))
        out.append(self.eos_id)
        return out

    def build_prompt(
        self,
        call_notes: Sequence[Note],
        key: str,
        tempo: int,
    ) -> List[int]:
        """Build the inference prompt (everything up to and including [RESPONSE])."""
        out: List[int] = [
            self.bos_id,
            self.key_token(key),
            self.tempo_token(tempo),
            self.call_id,
        ]
        out.extend(self.encode_notes(call_notes))
        out.append(self.sep_id)
        out.append(self.response_id)
        return out

    # -- decoding --------------------------------------------------------------

    def decode_notes(self, tokens: Iterable[int]) -> List[Note]:
        """Convert a REMI token stream back into ``Note`` objects.

        This is a *lenient* decoder: invalid token orderings are skipped rather
        than raising, because model output is not guaranteed to be well-formed.
        Special tokens (BAR / specials / KEY / TEMPO) are handled in-stream.

        State machine
        -------------
        We track:
          * ``cur_bar`` — number of [BAR] tokens seen so far (each adds 16 steps)
          * ``pending_pos`` — last [POS_*] seen but not yet attached to a note
          * ``pending_pitch`` — last [PITCH_*] seen but not yet attached
        A note is emitted only when (POS, PITCH, DUR) all arrive in order.
        """
        notes: List[Note] = []
        cur_bar = -1
        pending_pos: int | None = None
        pending_pitch: int | None = None

        for idx in tokens:
            if idx == self.bar_id:
                cur_bar += 1
                pending_pos = None
                pending_pitch = None
            elif self.is_position(idx):
                pending_pos = self.position_value(idx)
                pending_pitch = None
            elif self.is_pitch(idx):
                # Need a position before a pitch.
                if pending_pos is None:
                    continue
                pending_pitch = self.pitch_value(idx)
            elif self.is_duration(idx):
                if pending_pos is None or pending_pitch is None or cur_bar < 0:
                    continue
                dur = self.duration_value(idx)
                start = cur_bar * STEPS_PER_BAR + pending_pos
                notes.append(Note(pending_pitch, start, start + dur))
                # Keep position; only invalidate pitch so successive
                # POS/PITCH/DUR can chain naturally.
                pending_pitch = None
            else:
                # Specials / key / tempo inside the stream: ignore for note
                # decoding. Callers can inspect them separately if needed.
                pass

        return notes

    def split_call_response(self, tokens: Sequence[int]) -> Tuple[List[int], List[int]]:
        """Slice a full sample on [SEP] / [RESPONSE] / [EOS] boundaries.

        Returns ``(call_tokens, response_tokens)``, each containing only the
        REMI body tokens (no [CALL], [SEP], [RESPONSE], [EOS]).
        """
        try:
            sep = tokens.index(self.sep_id)
        except ValueError:
            return list(tokens), []

        # Call body: between [CALL] and [SEP].
        try:
            call_start = tokens.index(self.call_id) + 1
        except ValueError:
            call_start = 0
        call_body = list(tokens[call_start:sep])

        # Response body: after [RESPONSE] (which sits right after [SEP]) until [EOS].
        resp_start = sep + 1
        if resp_start < len(tokens) and tokens[resp_start] == self.response_id:
            resp_start += 1
        try:
            eos = tokens.index(self.eos_id, resp_start)
        except ValueError:
            eos = len(tokens)
        resp_body = list(tokens[resp_start:eos])

        return call_body, resp_body


# -----------------------------------------------------------------------------
# Module-level singleton (most code only needs one)
# -----------------------------------------------------------------------------

_DEFAULT: Tokenizer | None = None


def get_default_tokenizer() -> Tokenizer:
    """Return a process-wide default tokenizer.

    Useful for scripts that don't want to thread a tokenizer instance through
    every function. Tests and library code should still construct their own.
    """
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Tokenizer()
    return _DEFAULT
