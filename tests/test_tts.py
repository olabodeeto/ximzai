from ximz.tts import FallbackTTS, clean_for_speech, split_sentences


def test_split_sentences() -> None:
    assert split_sentences("Hi Bode. It's 6 a.m.! Ready?  Go") == ["Hi Bode.", "It's 6 a.m.!", "Ready?", "Go"]


class _Recorder:
    def __init__(self, fail: bool = False) -> None:
        self.fail, self.said = fail, []

    def warm_up(self) -> None:
        if self.fail:
            raise RuntimeError("model missing")

    def speak(self, text: str) -> None:
        if self.fail:
            raise RuntimeError("boom")
        self.said.append(text)


def test_fallback_voice_used_when_primary_fails() -> None:
    primary, fallback = _Recorder(fail=True), _Recorder()
    tts = FallbackTTS(primary, fallback)
    tts.warm_up()
    tts.speak("hello")
    assert fallback.said == ["hello"]


def test_primary_used_when_healthy() -> None:
    primary, fallback = _Recorder(), _Recorder()
    tts = FallbackTTS(primary, fallback)
    tts.warm_up()
    tts.speak("hello")
    assert primary.said == ["hello"] and fallback.said == []


def test_strips_markdown() -> None:
    text = "# Title\n- **Bold** item\n2. `code` and [a link](http://x.y)\n```py\nprint(1)\n```"
    assert clean_for_speech(text) == "Title Bold item code and a link"


def test_plain_text_untouched() -> None:
    assert clean_for_speech("It's 3 o'clock, and sunny.") == "It's 3 o'clock, and sunny."
