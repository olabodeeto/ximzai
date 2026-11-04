"""Tools that let the model save and delete long-term facts about the user."""

from __future__ import annotations

from ximz.memory import MemoryStore
from ximz.tools import Tool


def memory_tools(store: MemoryStore, max_fact_chars: int) -> list[Tool]:
    return [
        Tool(
            name="remember",
            description=(
                "Save one lasting fact about the user to long-term memory, e.g. their name, "
                "family, work, location or preferences. Write it about 'the user'."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "fact": {
                        "type": "string",
                        "description": "One short fact, e.g. \"The user's name is Ada.\"",
                        "minLength": 3,
                        "maxLength": max_fact_chars,
                    }
                },
                "required": ["fact"],
            },
            risk="safe",
            func=lambda fact: store.add(fact),
            triggers=(r"\bremember (that|this)\b", r"\bdon't forget that\b"),
        ),
        Tool(
            name="forget",
            description="Delete remembered facts about the user that contain the given words.",
            parameters={
                "type": "object",
                "properties": {
                    "about": {
                        "type": "string",
                        "description": "A word or phrase identifying what to forget, e.g. 'birthday'.",
                        "minLength": 2,
                        "maxLength": max_fact_chars,
                    }
                },
                "required": ["about"],
            },
            risk="safe",
            func=lambda about: store.forget(about),
            triggers=(r"\bforget\b",),
        ),
    ]
