"""Ximz entry point.

Wake mode (default):  always listening → "Hello Ximz" → "Yes?" → record until
silence → STT → LLM → speak → keep listening for follow-ups (no wake phrase
needed) until silence or "that's all" → back to listening.
Push-to-talk (dev):   press Enter instead of saying the wake phrase.
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ximz.agent import Agent, LLMUnavailableError, ToolLimitError
from ximz.audio import MicStream, maybe_save_debug
from ximz.config import Config, load_config
from ximz.conversation import is_end_phrase, is_yes
from ximz.events import EventBus, State, StateEvent
from ximz.memory import MemoryStore
from ximz.stt import create_stt
from ximz.tools.catalog import build_registry
from ximz.tts import create_tts
from ximz.vad import Segmenter, SileroVAD
from ximz.wakeword import WhisperWakeWord

log = logging.getLogger("ximz")


class Assistant:
    def __init__(self, cfg: Config, bus: EventBus) -> None:
        self.cfg = cfg
        self.bus = bus
        self.stt = create_stt(cfg.stt)
        self.tts = create_tts(cfg.tts)
        memory = None
        # One background worker for memory extraction, so it never delays a reply.
        self._learner = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ximz-learn")
        if cfg.memory.enabled:
            memory = MemoryStore(cfg.memory)
            log.info("memory: %d fact(s) from %s", len(memory.facts()), memory.path)
        self.memory = memory
        self.agent = Agent(cfg.llm, tools=build_registry(cfg, memory), memory=memory)
        self.mic = MicStream(cfg.audio)
        self.segmenter = Segmenter(SileroVAD(cfg.audio.sample_rate), cfg.vad, cfg.audio)
        self.wake = WhisperWakeWord(cfg.wake, self.stt, cfg.audio.sample_rate)
        self._silent_chunks = 0
        self._warned_silent_mic = False

    # -- startup -----------------------------------------------------------

    def startup(self) -> None:
        if "calendar" in self.cfg.tools.enabled:
            # Ask for Calendar/Reminders access now (macOS shows a one-time
            # prompt) rather than in the middle of a spoken request.
            threading.Thread(target=self._request_calendar_access, name="ximz-eventkit", daemon=True).start()
        self._wait_for_llm()
        self.tts.warm_up()
        log.info("loading speech model %s", self.cfg.stt.model)
        self.stt.transcribe(np.zeros(self.cfg.audio.sample_rate // 2, dtype=np.float32))
        try:
            self.agent.warm_up()
        except Exception as e:  # not fatal: we'll report per interaction
            log.warning("could not warm up LLM: %s", e)

    def _request_calendar_access(self) -> None:
        try:
            from EventKit import EKEntityTypeEvent, EKEntityTypeReminder

            from ximz.tools.calendar import EventKitStore

            store = EventKitStore(self.cfg.tools)
            for label, entity in (("calendar", EKEntityTypeEvent), ("reminders", EKEntityTypeReminder)):
                log.info("%s access: %s", label, "granted" if store.ensure_access(entity) else "NOT granted")
        except Exception:
            log.exception("could not request calendar access")

    def _wait_for_llm(self) -> None:
        """At login Ollama may still be starting: retry with exponential backoff."""
        delay, deadline = self.cfg.llm.retry_initial_s, time.monotonic() + self.cfg.llm.startup_wait_s
        while not self.agent.is_available():
            if time.monotonic() >= deadline:
                log.error("Ollama not available after %ss; continuing anyway", self.cfg.llm.startup_wait_s)
                return
            log.warning("waiting for Ollama at %s (retry in %.0fs)", self.cfg.llm.host, delay)
            time.sleep(delay)
            delay = min(delay * 2, self.cfg.llm.retry_max_s)
        log.info("Ollama ready (%s)", self.cfg.llm.model)

    # -- audio helpers -----------------------------------------------------

    def _chunks(self, timeout: float | None = None) -> Iterator[np.ndarray]:
        while True:
            chunk = self.mic.read(timeout=timeout)
            if chunk is None:
                return
            self._check_silent_mic(chunk)
            yield chunk

    def _check_silent_mic(self, chunk: np.ndarray) -> None:
        if np.any(chunk):
            self._silent_chunks = 0
            return
        self._silent_chunks += 1
        silent_s = self._silent_chunks * self.cfg.audio.chunk_samples / self.cfg.audio.sample_rate
        if silent_s >= self.cfg.audio.silent_mic_warn_s and not self._warned_silent_mic:
            self._warned_silent_mic = True
            log.error("microphone is delivering pure silence — Ximz probably has no mic "
                      "permission. Check System Settings → Privacy & Security → Microphone.")

    def _next_segment(self, max_s: float, start_timeout_s: float | None = None) -> np.ndarray | None:
        """Block until one speech segment completes. With `start_timeout_s`,
        give up if speech hasn't *started* within that time."""
        self.segmenter.reset(max_s)
        started = time.monotonic()
        for chunk in self._chunks():
            segment = self.segmenter.feed(chunk)
            if segment is not None:
                return segment
            if (start_timeout_s is not None and not self.segmenter.in_speech
                    and time.monotonic() - started > start_timeout_s):
                return None
        return None

    def _say(self, text: str) -> None:
        self.bus.emit(State.SPEAKING, text)
        self.tts.speak(text)
        # Drop anything the mic picked up while we were talking (our own voice).
        self.mic.clear()
        self.segmenter.reset()

    # -- interaction -------------------------------------------------------

    def _listen_for_command(self, start_timeout_s: float | None = None) -> str | None:
        self.bus.emit(State.LISTENING)
        timeout = self.cfg.vad.listen_timeout_s if start_timeout_s is None else start_timeout_s
        audio = self._next_segment(self.cfg.vad.max_command_s, timeout)
        if audio is None:
            log.info("no speech within %.0fs", timeout)
            return None
        maybe_save_debug(audio, self.cfg.audio.sample_rate, self.cfg.debug)
        self.bus.emit(State.TRANSCRIBING)
        vocab = self.cfg.stt.vocabulary
        text = self.stt.transcribe(audio, prompt=", ".join(vocab) + "." if vocab else None)
        log.info("heard: %r", text)
        return text or None

    def _answer(self, text: str) -> None:
        self.bus.emit(State.THINKING, text)
        try:
            reply = self.agent.respond(text, confirm=self._confirm)
            self._learner.submit(self._learn, text)
        except LLMUnavailableError as e:
            log.error("LLM unavailable: %s", e)
            reply = self.cfg.messages.llm_unavailable
        except ToolLimitError as e:
            log.error("%s", e)
            reply = self.cfg.messages.tool_limit
        log.info("reply: %r", reply)
        self._say(reply)

    def _confirm(self, question: str) -> bool:
        """Ask a yes/no question out loud; only a clear spoken yes counts."""
        self._say(question)
        answer = self._listen_for_command(self.cfg.tools.confirm_timeout_s)
        ok = is_yes(answer, self.cfg.tools.yes_words, self.cfg.tools.no_words)
        log.info("confirmation %r -> %s", answer, "yes" if ok else "no")
        self.bus.emit(State.THINKING)
        return ok

    def _learn(self, user_text: str) -> None:
        """Background: save lasting facts the model may have failed to `remember`."""
        if self.memory is None or not self.cfg.memory.auto_extract:
            return
        try:
            facts = self.agent.extract_facts(
                user_text, self.memory.facts(), self.cfg.memory.extract_prompt,
                self.cfg.memory.max_extract_facts,
            )
            for fact in facts:
                self.memory.add(fact)
        except Exception:
            log.exception("memory extraction failed")

    def _conversation(self, first_request: str) -> None:
        """Answer, then keep listening for follow-ups without the wake phrase
        until the user goes quiet or says an end phrase."""
        text: str | None = first_request
        while text:
            if is_end_phrase(text, self.cfg.conversation.end_phrases):
                log.info("conversation ended by user")
                self._say(self.cfg.conversation.end_reply)
                return
            self._answer(text)
            if self.cfg.conversation.follow_up_s <= 0:
                return
            text = self._listen_for_command(self.cfg.conversation.follow_up_s)
        log.info("conversation ended (quiet)")

    def _guarded(self, interaction: Callable[[], None]) -> None:
        """Never crash the loop: log, apologise, return to listening."""
        try:
            interaction()
        except Exception:
            log.exception("interaction failed")
            self.bus.emit(State.ERROR)
            try:
                self.tts.speak(self.cfg.messages.error)
            except Exception:
                log.exception("could not speak error message")
        finally:
            self.mic.clear()
            self.segmenter.reset()
            self.bus.emit(State.IDLE)

    def _on_wake(self, command: str) -> None:
        if len(command.split()) >= self.cfg.wake.min_command_words:
            log.info("wake phrase + request: %r", command)
            self.bus.emit(State.LISTENING)  # clear previous text in the overlay
            self._conversation(command)
            return
        self._say(self.cfg.wake.greeting)
        text = self._listen_for_command()
        if text:
            self._conversation(text)

    # -- main loops --------------------------------------------------------

    def run_wake(self) -> None:
        with self.mic:
            log.info("Ximz listening for “%s %s”", self.cfg.wake.greetings[0].title(),
                     self.cfg.wake.names[0].title())
            self.bus.emit(State.IDLE)
            while True:
                segment = self._next_segment(self.cfg.wake.max_segment_s)
                if segment is None:
                    continue
                try:
                    result = self.wake.check(segment)
                except Exception:
                    log.exception("wake check failed")
                    continue
                if result is not None:
                    log.info("wake phrase detected: %r", result.transcript)
                    self._guarded(lambda: self._on_wake(result.command))

    def run_push_to_talk(self) -> None:
        with self.mic:
            while True:
                self.bus.emit(State.IDLE)
                try:
                    input("Press Enter and speak (Ctrl+C to quit)... ")
                except EOFError:
                    return
                self.mic.clear()

                def interaction() -> None:
                    text = self._listen_for_command()
                    if text:
                        self._conversation(text)
                    else:
                        self._say(self.cfg.messages.no_speech)

                self._guarded(interaction)


def _log_state(event: StateEvent) -> None:
    log.debug("state=%s", event.state.value)


def setup_logging(level: str) -> None:
    """Interactive: everything to stderr. Under launchd (no TTY): everything to
    stdout (ximz.log) and warnings+ also to stderr (ximz.err.log)."""
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(level)
    for noisy in ("httpx", "httpx2", "httpcore", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if sys.stderr.isatty():
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(fmt)
        root.addHandler(handler)
        return
    out = logging.StreamHandler(sys.stdout)
    out.setFormatter(fmt)
    err = logging.StreamHandler(sys.stderr)
    err.setFormatter(fmt)
    err.setLevel(logging.WARNING)
    root.addHandler(out)
    root.addHandler(err)


def main() -> None:
    parser = argparse.ArgumentParser(prog="ximz")
    parser.add_argument("--push-to-talk", action="store_true", help="press Enter instead of wake phrase")
    parser.add_argument("--config", help="path to config.yaml")
    parser.add_argument("--no-gui", action="store_true", help="disable the overlay for this run")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.logging.level)
    bus = EventBus()
    bus.subscribe(_log_state)

    def voice_loop() -> None:
        assistant = Assistant(cfg, bus)
        assistant.startup()
        if args.push_to_talk:
            assistant.run_push_to_talk()
        else:
            assistant.run_wake()

    try:
        if cfg.gui.enabled and not args.no_gui:
            from ximz.gui import run_with_gui  # AppKit imported only when needed

            run_with_gui(cfg.gui, bus, voice_loop)
        else:
            voice_loop()
    except KeyboardInterrupt:
        pass
    log.info("bye")


if __name__ == "__main__":
    main()
