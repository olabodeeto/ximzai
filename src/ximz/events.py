"""Assistant state events.

The main loop emits a state change at each stage. Anything that wants to react
(logging today, a GUI overlay / menu-bar icon later) subscribes here, so the
loop never needs to know whether a UI exists.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

log = logging.getLogger(__name__)


class State(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    SPEAKING = "speaking"
    ERROR = "error"


@dataclass(frozen=True)
class StateEvent:
    state: State
    text: str | None = None  # transcript or reply text, when relevant


Listener = Callable[[StateEvent], None]


class EventBus:
    def __init__(self) -> None:
        self._listeners: list[Listener] = []

    def subscribe(self, listener: Listener) -> None:
        self._listeners.append(listener)

    def emit(self, state: State, text: str | None = None) -> None:
        event = StateEvent(state, text)
        for listener in self._listeners:
            try:
                listener(event)
            except Exception:
                # A broken UI must never take down the assistant.
                log.exception("state listener failed")
