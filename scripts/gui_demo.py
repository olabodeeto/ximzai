"""Cycle the overlay through every state without mic/STT/LLM, for visual checks.

    uv run python scripts/gui_demo.py
"""

from __future__ import annotations

import logging
import time

from ximz.config import load_config
from ximz.events import EventBus, State
from ximz.gui import run_with_gui

STEP_S = 1.6

SCRIPT: list[tuple[State, str | None]] = [
    (State.LISTENING, None),
    (State.TRANSCRIBING, None),
    (State.THINKING, "What's the weather like in Lagos tomorrow?"),
    (State.SPEAKING, "Tomorrow in Lagos expect around 31 degrees with scattered "
                     "afternoon showers, so take an umbrella."),
    (State.IDLE, None),
]


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    cfg = load_config()
    bus = EventBus()

    def worker() -> None:
        time.sleep(STEP_S)
        for state, text in SCRIPT:
            bus.emit(state, text)
            time.sleep(STEP_S * 2 if state == State.SPEAKING else STEP_S)
        time.sleep(cfg.gui.hide_delay_s + 1)

    run_with_gui(cfg.gui, bus, worker)


if __name__ == "__main__":
    main()
