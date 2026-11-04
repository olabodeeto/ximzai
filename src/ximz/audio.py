"""Continuous microphone stream (and debug audio saving)."""

from __future__ import annotations

import logging
import queue
import time
from pathlib import Path
from types import TracebackType

import numpy as np
import sounddevice as sd
import soundfile as sf

from ximz.config import PROJECT_ROOT, AudioConfig, DebugConfig

log = logging.getLogger(__name__)


class MicStream:
    """Opens the default mic once and hands out fixed-size float32 mono chunks.

    The PortAudio callback runs on its own thread and only enqueues; all
    processing happens in the reader. If the reader falls behind (e.g. while
    Ximz is thinking), the oldest audio is dropped once the buffer is full.
    """

    def __init__(self, cfg: AudioConfig) -> None:
        self._cfg = cfg
        max_chunks = int(cfg.queue_seconds * cfg.sample_rate / cfg.chunk_samples)
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=max_chunks)
        self._stream: sd.InputStream | None = None

    def _callback(self, indata: np.ndarray, frames: int, time_info: object, status: sd.CallbackFlags) -> None:
        if status:
            log.debug("mic status: %s", status)
        chunk = indata.mean(axis=1) if indata.shape[1] > 1 else indata[:, 0]
        chunk = chunk.astype(np.float32, copy=True)
        try:
            self._queue.put_nowait(chunk)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(chunk)

    def start(self) -> None:
        self._stream = sd.InputStream(
            samplerate=self._cfg.sample_rate,
            channels=self._cfg.channels,
            blocksize=self._cfg.chunk_samples,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()
        log.info("microphone open: %s", sd.query_devices(kind="input")["name"])

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def read(self, timeout: float | None = None) -> np.ndarray | None:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def clear(self) -> None:
        """Drop buffered audio, e.g. Ximz's own voice captured while speaking."""
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def __enter__(self) -> MicStream:
        self.start()
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.stop()


def maybe_save_debug(audio: np.ndarray, sample_rate: int, debug: DebugConfig) -> Path | None:
    """Write audio to disk only when debug.save_audio is enabled (privacy rule)."""
    if not debug.save_audio:
        return None
    out_dir = Path(debug.audio_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{time.strftime('%Y%m%d-%H%M%S')}.wav"
    sf.write(path, audio, sample_rate)
    log.debug("saved debug audio to %s", path)
    return path
