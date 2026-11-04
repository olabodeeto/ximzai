import numpy as np
import pytest

from ximz.config import load_config
from ximz.wakeword import WhisperWakeWord, match_wake_phrase

WAKE = load_config().wake


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hello Ximz.", ""),
        ("Hello, Sims!", ""),
        ("hello zims", ""),
        ("Hello Zimm's.", ""),
        ("Hello, Ximes, what time is it?", "what time is it?"),
        ("Uh, hello Ximz turn up the volume", "turn up the volume"),
        ("Hey Ximz", ""),
        ("Hello scenes.", ""),  # real-world mishearing
        ("Ximz.", ""),  # name alone
        ("Ximz, what time is it?", "what time is it?"),
        ("Okay Ximz, turn it up.", "turn it up."),
        ("Zims, what's the weather", "what's the weather"),
    ],
)
def test_wake_phrase_matches(text: str, expected: str) -> None:
    assert match_wake_phrase(text, WAKE) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Hello everyone, welcome back.",
        "I said hello to Jim yesterday.",
        "So anyway I was telling them hello Ximz",  # too far into the sentence
        "",
        "Hello",
        "Seems like it's going to rain.",  # name alone must not fire on real words
        "Sims is a fun game.",
        "Scenes from the movie were great.",
        "I really love Ximz.",  # name alone only counts as the first word
        "Ximz like it's going to rain.",  # what the spelling hint makes of "Seems like…"
    ],
)
def test_wake_phrase_rejects(text: str) -> None:
    assert match_wake_phrase(text, WAKE) is None


class FakeSTT:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    def transcribe(self, audio: np.ndarray, prompt: str | None = None) -> str:
        self.calls += 1
        self.prompt = prompt
        return self.text


def test_long_or_short_segments_are_never_transcribed() -> None:
    stt = FakeSTT("Hello Ximz")
    wake = WhisperWakeWord(WAKE, stt, 16000)
    assert wake.check(np.zeros(int(16000 * (WAKE.max_segment_s + 1)), np.float32)) is None
    assert wake.check(np.zeros(int(16000 * WAKE.min_segment_s / 2), np.float32)) is None
    assert stt.calls == 0
    result = wake.check(np.zeros(16000, np.float32))
    assert result is not None and result.command == ""
