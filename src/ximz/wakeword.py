"""Wake phrase detection ("Hello Ximz").

v1 backend: transcribe short speech bursts with the STT model and fuzzy-match
"<greeting> <name>" near the start. Long speech is never transcribed, so idle
cost stays low. A trained openWakeWord model can replace this later behind the
same `WakeWord` interface.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Protocol

import numpy as np

from ximz.config import WakeConfig
from ximz.stt import STT

log = logging.getLogger(__name__)

_WORD = re.compile(r"[A-Za-z0-9']+")
_LEADING_PUNCT = " \t\n,.!?;:-–—"


@dataclass(frozen=True)
class WakeResult:
    transcript: str
    command: str  # whatever was said after the wake phrase ("" if nothing)


class WakeWord(Protocol):
    def check(self, segment: np.ndarray) -> WakeResult | None: ...


def _norm(word: str) -> str:
    return word.lower().replace("'", "")


_NAME_ALONE_FILLER = {"ok", "okay", "um", "uh", "erm", "so", "oh"}
_VOCATIVE_PUNCT = set(",.!?;:-–—")


def _similar(word: str, names: list[str], threshold: float) -> bool:
    return any(SequenceMatcher(None, word, name).ratio() >= threshold for name in names)


def match_wake_phrase(text: str, cfg: WakeConfig) -> str | None:
    """Return the text after the wake phrase, or None if it isn't there.

    Accepts "<greeting> <name>" near the start, and (if `name_alone`) the name
    on its own as the first word."""
    words = list(_WORD.finditer(text))
    normed = [_norm(w.group()) for w in words]

    def after(i: int) -> str:
        return text[words[i].end():].lstrip(_LEADING_PUNCT).strip()

    greetings = {g.lower() for g in cfg.greetings}
    for i in range(min(cfg.max_leading_words + 1, len(words) - 1)):
        if normed[i] in greetings and _similar(normed[i + 1], cfg.names, cfg.name_similarity):
            return after(i + 1)

    if cfg.name_alone and normed:
        first = 1 if normed[0] in _NAME_ALONE_FILLER and len(normed) > 1 else 0
        # Addressing someone by name is punctuated ("Ximz, what's…" / "Ximz.").
        # Requiring that rejects sentences the spelling hint bent into
        # "Ximz like it's going to rain" (really "Seems like…").
        next_char = text[words[first].end():].lstrip()[:1]
        addressed = next_char == "" or next_char in _VOCATIVE_PUNCT
        if addressed and _similar(normed[first], cfg.name_alone_names, cfg.name_alone_similarity):
            return after(first)
    return None


class WhisperWakeWord:
    def __init__(self, cfg: WakeConfig, stt: STT, sample_rate: int) -> None:
        self._cfg = cfg
        self._stt = stt
        self._sr = sample_rate

    def check(self, segment: np.ndarray) -> WakeResult | None:
        duration = len(segment) / self._sr
        if not self._cfg.min_segment_s <= duration <= self._cfg.max_segment_s:
            log.debug("ignoring %.1fs speech burst", duration)
            return None
        text = self._stt.transcribe(segment, prompt=self._cfg.stt_prompt or None)
        command = match_wake_phrase(text, self._cfg)
        if command is None and self._cfg.stt_prompt:
            # Whisper sometimes *drops* words that repeat its prompt, so retry
            # unprompted before giving up (~80 ms extra, only on non-matches).
            text = self._stt.transcribe(segment)
            command = match_wake_phrase(text, self._cfg)
        log.debug("wake check %.1fs: %r -> %s", duration, text, "MATCH" if command is not None else "no")
        return None if command is None else WakeResult(text, command)
