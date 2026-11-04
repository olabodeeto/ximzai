"""Speech-to-text. mlx-whisper runs on the Apple GPU via MLX."""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

from ximz.config import STTConfig

log = logging.getLogger(__name__)


class STT(Protocol):
    def transcribe(self, audio: np.ndarray, prompt: str | None = None) -> str: ...


class MLXWhisperSTT:
    def __init__(self, cfg: STTConfig) -> None:
        self._cfg = cfg

    def transcribe(self, audio: np.ndarray, prompt: str | None = None) -> str:
        """`prompt` biases vocabulary/spelling (e.g. the unusual name "Ximz")."""
        # Imported lazily so idle startup stays cheap; mlx_whisper caches the
        # loaded model internally after the first call.
        import mlx_whisper

        result = mlx_whisper.transcribe(
            audio.astype(np.float32),
            path_or_hf_repo=self._cfg.model,
            language=self._cfg.language,
            temperature=0.0,  # greedy decode: fastest, and no sampling retries
            condition_on_previous_text=False,
            initial_prompt=prompt,
        )
        return str(result.get("text", "")).strip()


def create_stt(cfg: STTConfig) -> STT:
    return MLXWhisperSTT(cfg)
