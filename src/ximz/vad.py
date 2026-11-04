"""Voice activity detection: Silero VAD + a segmenter that cuts speech out of a stream."""

from __future__ import annotations

import logging
from collections import deque
from typing import Protocol

import numpy as np

from ximz.config import AudioConfig, VADConfig

log = logging.getLogger(__name__)


class VAD(Protocol):
    def speech_prob(self, chunk: np.ndarray) -> float: ...
    def reset(self) -> None: ...


class SileroVAD:
    """Silero VAD on CPU, single thread — cheap enough to run continuously."""

    def __init__(self, sample_rate: int) -> None:
        import torch
        from silero_vad import load_silero_vad

        torch.set_num_threads(1)
        self._torch = torch
        self._model = load_silero_vad()
        self._sr = sample_rate

    def speech_prob(self, chunk: np.ndarray) -> float:
        with self._torch.no_grad():
            return float(self._model(self._torch.from_numpy(chunk), self._sr).item())

    def reset(self) -> None:
        self._model.reset_states()


class Segmenter:
    """Feed fixed-size chunks; get back one complete speech segment at a time.

    A segment starts when speech probability crosses `threshold`, keeps a short
    pre-roll so the first syllable isn't clipped, and ends after `silence_ms`
    below `neg_threshold` (hysteresis) or when `max_s` is reached.
    """

    def __init__(self, vad: VAD, vad_cfg: VADConfig, audio_cfg: AudioConfig) -> None:
        self._vad = vad
        self._cfg = vad_cfg
        self._chunk_ms = 1000 * audio_cfg.chunk_samples / audio_cfg.sample_rate
        self._sr = audio_cfg.sample_rate
        pre_roll_chunks = max(1, round(vad_cfg.pre_roll_ms / self._chunk_ms))
        self._pre_roll: deque[np.ndarray] = deque(maxlen=pre_roll_chunks)
        self._max_s = vad_cfg.max_command_s
        self._reset_segment()

    def _reset_segment(self) -> None:
        self._buffer: list[np.ndarray] = []
        self._speech_ms = 0.0
        self._silence_ms = 0.0

    @property
    def in_speech(self) -> bool:
        return bool(self._buffer)

    def reset(self, max_s: float | None = None) -> None:
        """Forget all state; optionally change the max segment length."""
        self._vad.reset()
        self._pre_roll.clear()
        self._reset_segment()
        if max_s is not None:
            self._max_s = max_s

    def feed(self, chunk: np.ndarray) -> np.ndarray | None:
        prob = self._vad.speech_prob(chunk)

        if not self.in_speech:
            if prob >= self._cfg.threshold:
                self._buffer = [*self._pre_roll, chunk]
                self._speech_ms = self._chunk_ms
                self._silence_ms = 0.0
            else:
                self._pre_roll.append(chunk)
            return None

        self._buffer.append(chunk)
        if prob >= self._cfg.neg_threshold:
            self._speech_ms += self._chunk_ms
            self._silence_ms = 0.0
        else:
            self._silence_ms += self._chunk_ms

        too_long = len(self._buffer) * self._chunk_ms >= self._max_s * 1000
        if self._silence_ms < self._cfg.silence_ms and not too_long:
            return None

        audio, speech_ms = np.concatenate(self._buffer), self._speech_ms
        self._reset_segment()
        self._pre_roll.clear()
        self._vad.reset()
        if speech_ms < self._cfg.min_speech_ms:
            return None
        return audio
