"""LLM agent backed by Ollama, using the standard tool-calling loop:
send messages + tool schemas → run any tool_calls → append results as
role "tool" → repeat until a plain answer (capped at llm.max_tool_steps)."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Protocol

import httpx
import ollama

from ximz.config import LLMConfig
from ximz.memory import MemoryStore
from ximz.tools import ToolRegistry

log = logging.getLogger(__name__)


_MIN_FACT_WORDS = 4
# Vague "facts" small models produce that carry no information.
_VAGUE_FACT = re.compile(
    r"^the user (has|is|wants|needs|likes) (a |an |some )?(name|question|request|assistant|thing|something)\.?$",
    re.IGNORECASE,
)


def is_useful_fact(fact: str) -> bool:
    words = re.findall(r"[A-Za-z0-9']+", fact)
    return len(words) >= _MIN_FACT_WORDS and not _VAGUE_FACT.match(fact.strip())


class LLMUnavailableError(RuntimeError):
    """Ollama could not be reached or rejected the request."""


class ToolLimitError(RuntimeError):
    """The model kept calling tools past the step cap."""


class ChatClient(Protocol):
    def chat(self, **kwargs: Any) -> Any: ...


class Agent:
    def __init__(
        self,
        cfg: LLMConfig,
        client: ChatClient | None = None,
        tools: ToolRegistry | None = None,
        memory: MemoryStore | None = None,
    ) -> None:
        self._cfg = cfg
        self._client: ChatClient = client or ollama.Client(host=cfg.host, timeout=cfg.request_timeout_s)
        self._tools = tools or ToolRegistry()
        self._memory = memory
        self._history: list[dict[str, str]] = []

    def _system_prompt(self) -> str:
        parts = [self._cfg.system_prompt, f"Today is {time.strftime('%A, %d %B %Y')}."]
        if self._memory is not None:
            parts.append(self._memory.prompt_block())
        return "\n\n".join(parts)

    def _chat(self, messages: list[Any]) -> Any:
        try:
            return self._client.chat(
                model=self._cfg.model,
                messages=messages,
                tools=self._tools.schemas() or None,
                keep_alive=self._cfg.keep_alive,
                options={"temperature": self._cfg.temperature},
            )
        except (ConnectionError, httpx.HTTPError, ollama.ResponseError) as e:
            raise LLMUnavailableError(str(e)) from e

    def respond(self, user_text: str) -> str:
        messages: list[Any] = [
            {"role": "system", "content": self._system_prompt()},
            *self._history,
            {"role": "user", "content": user_text},
        ]
        for _ in range(self._cfg.max_tool_steps):
            message = self._chat(messages).message
            tool_calls = getattr(message, "tool_calls", None)
            if not tool_calls:
                reply = (message.content or "").strip()
                self._remember(user_text, reply)
                return reply
            messages.append(message)
            for call in tool_calls:
                name, args = call.function.name, call.function.arguments
                log.info("tool call: %s", name)
                messages.append({"role": "tool", "tool_name": name, "content": self._tools.call(name, args)})
        raise ToolLimitError(f"no answer after {self._cfg.max_tool_steps} tool steps")

    def extract_facts(self, user_text: str, known: list[str], prompt: str, limit: int) -> list[str]:
        """Ask the model for new lasting facts in `user_text` (JSON mode, no tools)."""
        known_block = "\n".join(f"- {f}" for f in known) or "(nothing yet)"
        response = self._client.chat(
            model=self._cfg.model,
            messages=[
                {"role": "system", "content": f"{prompt}\n\nAlready known:\n{known_block}"},
                {"role": "user", "content": user_text},
            ],
            format="json",
            keep_alive=self._cfg.keep_alive,
            options={"temperature": 0},
        )
        try:
            data = json.loads(response.message.content or "{}")
        except json.JSONDecodeError:
            log.warning("fact extraction returned invalid JSON")
            return []
        facts = data.get("facts", []) if isinstance(data, dict) else []
        known_lower = {k.lower().rstrip(".") for k in known}
        clean = [
            f.strip() for f in facts
            if isinstance(f, str) and is_useful_fact(f) and f.strip().lower().rstrip(".") not in known_lower
        ]
        return clean[:limit]

    def is_available(self) -> bool:
        """True if Ollama answers and the configured model is installed."""
        try:
            models = self._client.list().models  # type: ignore[attr-defined]
        except (ConnectionError, httpx.HTTPError, ollama.ResponseError):
            return False
        names = {m.model for m in models}
        if self._cfg.model not in names and f"{self._cfg.model}:latest" not in names:
            log.error("Ollama is up but model %s is not installed (ollama pull %s)",
                      self._cfg.model, self._cfg.model)
            return False
        return True

    def warm_up(self) -> None:
        """Load the model into memory now so the first answer is fast."""
        self._client.generate(model=self._cfg.model, prompt="", keep_alive=self._cfg.keep_alive)  # type: ignore[attr-defined]

    def _remember(self, user_text: str, reply: str) -> None:
        self._history += [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": reply},
        ]
        limit = self._cfg.max_history_messages
        self._history = self._history[-limit:] if limit else []
