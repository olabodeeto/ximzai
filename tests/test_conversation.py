import pytest

from ximz.config import load_config
from ximz.conversation import is_end_phrase

PHRASES = load_config().conversation.end_phrases


@pytest.mark.parametrize(
    "text", ["Stop.", "That's all.", "Okay, that's all, thanks.", "Goodbye!", "Thank you, bye.",
             "Never mind.", "No thanks.", "Nothing."]
)
def test_ends(text: str) -> None:
    assert is_end_phrase(text, PHRASES)


@pytest.mark.parametrize(
    "text", ["Stop the music.", "Nothing much, how about you?", "What's the weather tomorrow?",
             "Is that all you know?", "Say goodbye to my friend Tunde.", ""]
)
def test_does_not_end(text: str) -> None:
    assert not is_end_phrase(text, PHRASES)
