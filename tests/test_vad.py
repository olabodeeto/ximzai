import numpy as np

from ximz.config import load_config
from ximz.vad import Segmenter

CFG = load_config()
CHUNK = CFG.audio.chunk_samples
CHUNK_MS = 1000 * CHUNK / CFG.audio.sample_rate


class ScriptedVAD:
    """Returns speech probability from the chunk's first sample."""

    def __init__(self) -> None:
        self.resets = 0

    def speech_prob(self, chunk: np.ndarray) -> float:
        return float(chunk[0])

    def reset(self) -> None:
        self.resets += 1


def chunks(prob: float, ms: float) -> list[np.ndarray]:
    return [np.full(CHUNK, prob, np.float32) for _ in range(round(ms / CHUNK_MS))]


def run(seg: Segmenter, stream: list[np.ndarray]) -> list[np.ndarray]:
    return [s for c in stream if (s := seg.feed(c)) is not None]


def test_segment_ends_after_silence_and_keeps_pre_roll() -> None:
    seg = Segmenter(ScriptedVAD(), CFG.vad, CFG.audio)
    stream = chunks(0.0, 1000) + chunks(0.9, 800) + chunks(0.0, CFG.vad.silence_ms + 100)
    out = run(seg, stream)
    assert len(out) == 1
    seconds = len(out[0]) / CFG.audio.sample_rate
    assert seconds >= 0.8 + CFG.vad.pre_roll_ms / 1000 - 0.05
    assert not seg.in_speech


def test_short_blip_is_discarded() -> None:
    seg = Segmenter(ScriptedVAD(), CFG.vad, CFG.audio)
    stream = chunks(0.9, CHUNK_MS) + chunks(0.0, CFG.vad.silence_ms + 100)
    assert run(seg, stream) == []


def test_hysteresis_keeps_segment_open_through_dips() -> None:
    seg = Segmenter(ScriptedVAD(), CFG.vad, CFG.audio)
    mid = (CFG.vad.threshold + CFG.vad.neg_threshold) / 2  # between the thresholds
    stream = chunks(0.9, 400) + chunks(mid, CFG.vad.silence_ms * 2) + chunks(0.0, CFG.vad.silence_ms + 100)
    out = run(seg, stream)
    assert len(out) == 1


def test_max_length_cuts_segment() -> None:
    seg = Segmenter(ScriptedVAD(), CFG.vad, CFG.audio)
    seg.reset(max_s=1.0)
    out = run(seg, chunks(0.9, 2500))
    assert len(out) >= 2
    assert len(out[0]) / CFG.audio.sample_rate <= 1.0 + CHUNK_MS / 1000
