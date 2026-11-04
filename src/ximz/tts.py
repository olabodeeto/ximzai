"""Text-to-speech.

Backends:
- kokoro: natural neural voice (Kokoro-82M via onnxruntime), fully offline.
  Sentences are synthesised one ahead of playback, so speech starts after the
  first sentence instead of after the whole reply.
- say: macOS built-in voice. Also the automatic fallback if Kokoro fails.
"""

from __future__ import annotations

import logging
import re
import subprocess
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Protocol

import numpy as np

from ximz.config import PROJECT_ROOT, TTSConfig

log = logging.getLogger(__name__)

_CODE_BLOCK = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`([^`]*)`")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_LIST_MARKER = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+", re.MULTILINE)
_EMPHASIS = re.compile(r"(\*\*|__|\*|_|~~)(.+?)\1")
_LEFTOVER = re.compile(r"[*_#>|~`]")
_WHITESPACE = re.compile(r"\s+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def clean_for_speech(text: str) -> str:
    """Strip markdown and formatting so TTS reads only natural words."""
    text = _CODE_BLOCK.sub(" ", text)
    text = _INLINE_CODE.sub(r"\1", text)
    text = _LINK.sub(r"\1", text)
    text = _HEADING.sub("", text)
    text = _LIST_MARKER.sub("", text)
    text = _EMPHASIS.sub(r"\2", text)
    text = _LEFTOVER.sub("", text)
    return _WHITESPACE.sub(" ", text).strip()


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]


class TTS(Protocol):
    def speak(self, text: str) -> None: ...
    def warm_up(self) -> None: ...


class SayTTS:
    def __init__(self, cfg: TTSConfig) -> None:
        self._cfg = cfg

    def warm_up(self) -> None:
        pass

    def speak(self, text: str) -> None:
        cleaned = clean_for_speech(text)
        if not cleaned:
            return
        cmd = ["say", "-r", str(self._cfg.say_rate)]
        if self._cfg.say_voice:
            cmd += ["-v", self._cfg.say_voice]
        # Text goes via stdin, never as an argv/shell string.
        subprocess.run(cmd, input=cleaned, text=True, check=True)


def _resolve(path: str) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else PROJECT_ROOT / p


class KokoroTTS:
    def __init__(self, cfg: TTSConfig) -> None:
        self._cfg = cfg
        self._kokoro = None  # loaded lazily (warm_up at startup)
        self._synth = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ximz-tts")

    def warm_up(self) -> None:
        if self._kokoro is None:
            from kokoro_onnx import Kokoro

            self._kokoro = Kokoro(str(_resolve(self._cfg.model_path)), str(_resolve(self._cfg.voices_path)))
            self._kokoro.create("Ready.", voice=self._cfg.voice, speed=self._cfg.speed, lang=self._cfg.lang)
            log.info("Kokoro voice ready (%s)", self._cfg.voice)

    def _create(self, sentence: str) -> tuple[np.ndarray, int]:
        assert self._kokoro is not None
        return self._kokoro.create(sentence, voice=self._cfg.voice, speed=self._cfg.speed, lang=self._cfg.lang)

    def speak(self, text: str) -> None:
        import sounddevice as sd

        sentences = split_sentences(clean_for_speech(text))
        if not sentences:
            return
        self.warm_up()
        # Synthesis runs one sentence ahead on a worker thread while we play.
        pending: list[Future[tuple[np.ndarray, int]]] = [self._synth.submit(self._create, s) for s in sentences]
        for future in pending:
            audio, sample_rate = future.result()
            sd.play(audio, sample_rate)
            sd.wait()


class FallbackTTS:
    """Use the primary backend; if it fails, say it with the fallback instead."""

    def __init__(self, primary: TTS, fallback: TTS) -> None:
        self._primary = primary
        self._fallback = fallback
        self._primary_ok = True

    def warm_up(self) -> None:
        try:
            self._primary.warm_up()
        except Exception:
            log.exception("primary TTS failed to load; using fallback voice")
            self._primary_ok = False

    def speak(self, text: str) -> None:
        if self._primary_ok:
            try:
                self._primary.speak(text)
                return
            except Exception:
                log.exception("primary TTS failed; using fallback voice for this reply")
        self._fallback.speak(text)


def create_tts(cfg: TTSConfig) -> TTS:
    if cfg.backend == "say":
        return SayTTS(cfg)
    if cfg.backend == "kokoro":
        return FallbackTTS(KokoroTTS(cfg), SayTTS(cfg))
    raise ValueError(f"unsupported TTS backend: {cfg.backend}")
