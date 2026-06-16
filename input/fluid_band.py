from __future__ import annotations

import base64
import io
import os
import wave
from pathlib import Path
from threading import Lock
from typing import Iterable, Sequence

import numpy as np

from data.tokenizer import Note


class FluidBandRenderer:
    """Persistent in-memory FluidSynth renderer."""

    def __init__(self, sample_rate: int = 22050) -> None:
        try:
            import fluidsynth
        except ImportError as exc:
            raise RuntimeError("pyfluidsynth is not installed") from exc

        self.fluidsynth = fluidsynth
        self.sample_rate = int(sample_rate)
        self.lock = Lock()
        sf_path = Path(os.environ.get(
            "RESPONDAI_SOUNDFONT",
            "/usr/share/sounds/sf2/FluidR3_GM.sf2",
        ))
        if not sf_path.exists():
            raise FileNotFoundError(f"SoundFont not found: {sf_path}")

        self.synth = fluidsynth.Synth(samplerate=self.sample_rate, gain=0.42)
        self.sfid = self.synth.sfload(str(sf_path))
        if self.sfid < 0:
            raise RuntimeError(f"Failed to load SoundFont: {sf_path}")
        self._select_programs()
        self.synth.get_samples(256)

    def _select_programs(self) -> None:
        self.synth.program_select(0, self.sfid, 0, 0)   # Acoustic Grand Piano
        self.synth.program_select(1, self.sfid, 0, 32)  # Acoustic Bass
        self.synth.program_select(2, self.sfid, 0, 4)   # Electric Piano 1
        self.synth.program_select(3, self.sfid, 0, 0)   # AI Acoustic Grand Piano

        # 리드는 선명하게, 베이스·컴핑은 별도 stem에서 충분히 들리도록 설정.
        channel_settings = {
            0: (100, 16, 0),   # user piano
            1: (118, 5, 0),    # acoustic bass
            2: (104, 24, 5),   # electric piano comp
            3: (92, 14, 0),    # softer AI piano
        }
        for channel, (volume, reverb, chorus) in channel_settings.items():
            self.synth.cc(channel, 7, volume)
            self.synth.cc(channel, 91, reverb)
            self.synth.cc(channel, 93, chorus)

    def render(
        self,
        tracks: Sequence[tuple[int, Sequence[Note], int]],
        *,
        bpm: int,
        swing_amount: float = 0.58,
        tail_sec: float = 0.45,
    ) -> np.ndarray:
        seconds_per_step = (60.0 / float(bpm)) / 4.0
        events: list[tuple[int, int, int, int, int]] = []

        for channel, notes, velocity in tracks:
            for note in notes:
                # channel 0은 사용자가 친 pitch/start/end를 그대로 보존한다.
                if channel == 0:
                    track_swing = 0.0
                    gate = 0.98
                elif channel == 1:
                    track_swing = min(0.16, swing_amount)
                    gate = 0.92
                elif channel == 2:
                    track_swing = swing_amount
                    gate = 0.80
                else:
                    track_swing = max(0.62, swing_amount)
                    gate = 0.84

                swing = (
                    (2.0 / 3.0) * track_swing * seconds_per_step
                    if int(note.start) % 4 == 2
                    else 0.0
                )
                start = max(
                    0,
                    round((int(note.start) * seconds_per_step + swing) * self.sample_rate),
                )
                duration = max(
                    seconds_per_step * 0.25,
                    (int(note.end) - int(note.start)) * seconds_per_step,
                )
                end = max(
                    start + 1,
                    round((start / self.sample_rate + duration * gate) * self.sample_rate),
                )

                beat_position = int(note.start) % 16
                note_velocity = int(velocity)
                if beat_position == 0:
                    note_velocity += 4
                elif beat_position == 8:
                    note_velocity += 2
                elif beat_position in (2, 6, 10, 14) and channel != 0:
                    note_velocity -= 3
                note_velocity = max(1, min(127, note_velocity))

                events.append((start, 1, channel, int(note.pitch), note_velocity))
                events.append((end, 0, channel, int(note.pitch), 0))

        if not events:
            return np.zeros(int(0.4 * self.sample_rate), dtype=np.float32)
        events.sort(key=lambda event: (event[0], event[1]))

        with self.lock:
            try:
                self.synth.system_reset()
            except Exception:
                for channel in range(4):
                    try:
                        self.synth.cc(channel, 123, 0)
                    except Exception:
                        pass
            self._select_programs()

            chunks = []
            cursor = 0
            index = 0
            while index < len(events):
                position = events[index][0]
                if position > cursor:
                    chunks.append(np.asarray(self.synth.get_samples(position - cursor)))
                while index < len(events) and events[index][0] == position:
                    _, kind, channel, pitch, velocity = events[index]
                    if kind == 0:
                        self.synth.noteoff(channel, pitch)
                    else:
                        self.synth.noteon(channel, pitch, velocity)
                    index += 1
                cursor = position
            chunks.append(np.asarray(self.synth.get_samples(max(1, int(tail_sec * self.sample_rate)))))

        raw = np.concatenate(chunks)
        if raw.size % 2:
            raw = raw[:-1]
        stereo = raw.reshape(-1, 2)
        if np.issubdtype(stereo.dtype, np.integer):
            stereo = stereo.astype(np.float32) / float(np.iinfo(stereo.dtype).max)
        else:
            stereo = stereo.astype(np.float32)
        mono = stereo.mean(axis=1)
        peak = float(np.max(np.abs(mono))) if mono.size else 0.0
        if peak > 1e-8:
            mono = mono / peak * 0.76
        if mono.size:
            fade = min(
                len(mono) // 2,
                max(1, int(0.006 * self.sample_rate)),
            )
            if fade > 1:
                mono[:fade] *= np.linspace(
                    0.0,
                    1.0,
                    fade,
                    dtype=np.float32,
                )
                mono[-fade:] *= np.linspace(
                    1.0,
                    0.0,
                    fade,
                    dtype=np.float32,
                )

        return mono.astype(np.float32, copy=False)


_SHARED_RENDERERS: dict[int, FluidBandRenderer] = {}
_SHARED_RENDERERS_LOCK = Lock()


def get_shared_renderer(sample_rate: int = 22050) -> FluidBandRenderer:
    """프로세스 전체에서 SoundFont를 한 번만 로드한 renderer를 재사용한다."""
    rate = int(sample_rate)
    with _SHARED_RENDERERS_LOCK:
        renderer = _SHARED_RENDERERS.get(rate)
        if renderer is None:
            renderer = FluidBandRenderer(sample_rate=rate)
            _SHARED_RENDERERS[rate] = renderer
        return renderer


def _wav_data_uri(audio: np.ndarray, sample_rate: int) -> str:
    pcm = (
        np.clip(audio, -1.0, 1.0) * 32767
    ).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(pcm.tobytes())

    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:audio/wav;base64,{encoded}"


def build_browser_piano_sample_bank(
    midi_notes: Iterable[int],
    *,
    sample_rate: int = 22050,
) -> dict[str, str]:
    """브라우저 건반 즉시 재생용 SoundFont 피아노 샘플을 미리 만든다."""
    renderer = get_shared_renderer(sample_rate)
    bank: dict[str, str] = {}

    for midi in sorted({int(value) for value in midi_notes}):
        sample = renderer.render(
            [
                (
                    0,
                    [Note(midi, 0, 4)],
                    104,
                )
            ],
            bpm=180,
            swing_amount=0.0,
            tail_sec=0.38,
        )
        bank[str(midi)] = _wav_data_uri(sample, sample_rate)

    return bank
