"""Conversation helpers: detecting when the user wants to end the follow-up window."""

from __future__ import annotations

import re

_NON_WORD = re.compile(r"[^a-z' ]+")
# Words allowed around an end phrase: "okay, that's all, thanks" still ends it,
# but "stop the music" or "nothing much, you?" does not.
_FILLER = {"ok", "okay", "alright", "right", "so", "then", "thanks", "thank", "you", "please",
           "ximz", "sims", "zims", "well", "and"}


def is_yes(text: str | None, yes_words: list[str], no_words: list[str]) -> bool:
    """A clear spoken yes, with no 'no' anywhere ("yes... actually no" is a no)."""
    if not text:
        return False
    norm = " " + " ".join(_NON_WORD.sub(" ", text.lower()).split()) + " "
    if any(f" {w.lower()} " in norm for w in no_words):
        return False
    return any(f" {w.lower()} " in norm for w in yes_words)


def is_end_phrase(text: str, phrases: list[str]) -> bool:
    words = _NON_WORD.sub(" ", text.lower()).split()
    if not words:
        return False
    norm = " ".join(words)
    for phrase in phrases:
        p = phrase.lower()
        idx = f" {norm} ".find(f" {p} ")
        if idx == -1:
            continue
        rest = (norm[:idx] + " " + norm[idx + len(p):]).split()
        if all(w in _FILLER for w in rest):
            return True
    return False
