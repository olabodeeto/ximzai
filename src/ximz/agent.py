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
from ximz.tools import Confirmer, ToolRegistry

log = logging.getLogger(__name__)


_MIN_FACT_WORDS = 4
# Vague "facts" small models produce that carry no information.
_VAGUE_FACT = re.compile(
    r"^the user (has|is|wants|needs|likes) (a |an |some )?(name|question|request|assistant|thing|something)\.?$",
    re.IGNORECASE,
)


# Replies that narrate or claim an action ("Opening Spotify.", "I've added a
# reminder") — suspicious when no tool was actually called.
_ACTION_CLAIM = re.compile(
    r"^(?:ok(?:ay)?[,.]?\s+)?(?:opening|quitting|closing|creating|adding|setting|muting|unmuting|turning|"
    r"listing|getting|checking|launching|scheduling)\b"
    r"|\bI(?:'ve| have| just)\s+(?:added|set|created|opened|quit|closed|muted|unmuted|turned|changed|"
    r"scheduled|saved|launched)\b"
    r"|\b(?:has|have) been (?:added|set|created|opened|muted|scheduled|saved)\b"
    r"|\bis now (?:muted|unmuted|open|set)\b",
    re.IGNORECASE,
)
_NUDGE = (
    "You said you did something, but you did not call any tool, so nothing actually happened. "
    "If the request needs an action or live information, call the right tool now. If it does "
    "not, answer again without claiming any action."
)


def claims_action(reply: str) -> bool:
    return bool(_ACTION_CLAIM.search(reply))


# Trailing filler small models add despite instructions; dropped before speaking.
_FILLER_SENTENCE = re.compile(
    r"(is there anything else|anything else (i can|you('d| would) like)|how (else )?can i (help|assist)"
    r"|what would you like to do next|let me know if|feel free to (ask|let me know)|if you need anything"
    r"|happy to help|hope (this|that) helps)",
    re.IGNORECASE,
)
_SENTENCES = re.compile(r"(?<=[.!?])\s+")


def tidy_reply(reply: str) -> str:
    sentences = [s for s in _SENTENCES.split(reply.strip()) if s]
    while len(sentences) > 1 and _FILLER_SENTENCE.search(sentences[-1]):
        sentences.pop()
    return " ".join(sentences)


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
        # Past exchanges, each the full message list for one request: user,
        # any assistant tool calls + tool results, final answer. Keeping the
        # tool steps matters: if history shows only "Muting the sound." with no
        # tool call, small models learn to claim actions instead of doing them.
        self._exchanges: list[list[Any]] = []

    def _system_prompt(self) -> str:
        now = time.strftime("%A, %d %B %Y, %H:%M")
        parts = [self._cfg.system_prompt, f"The current local date and time is {now} ({time.strftime('%Z')})."]
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

    def respond(self, user_text: str, confirm: Confirmer | None = None) -> str:
        """`confirm` asks the user yes/no out loud before any risk="confirm" tool runs."""
        history = [m for exchange in self._exchanges for m in exchange]
        exchange: list[Any] = [{"role": "user", "content": user_text}]
        used_tool = nudged = False
        retry_context: list[Any] = []
        system = self._system_prompt()
        hinted = self._tools.suggest(user_text)
        if hinted:
            log.debug("tool hint: %s", hinted)
            system += (f"\n\nThe user's latest request most likely needs the {' or '.join(hinted)} tool. "
                       "Call it to actually do it; describing the action is not enough.")
        for _ in range(self._cfg.max_tool_steps):
            messages = [{"role": "system", "content": system}, *history, *exchange]
            message = self._chat(messages).message
            tool_calls = getattr(message, "tool_calls", None)
            if not tool_calls:
                reply = tidy_reply(message.content or "")
                if not nudged and not used_tool and claims_action(reply):
                    # Small models sometimes *say* they did something without
                    # calling the tool. Push back once and let it retry.
                    log.warning("model claimed an action without a tool call; nudging: %r", reply)
                    nudged = True
                    retry_context = [{"role": "assistant", "content": reply}, {"role": "system", "content": _NUDGE}]
                    exchange += retry_context
                    continue
                exchange.append({"role": "assistant", "content": reply})
                # History keeps neither the false claim nor the nudge.
                self._remember([m for m in exchange if not any(m is r for r in retry_context)])
                return reply
            used_tool = True
            exchange.append(message)
            for call in tool_calls:
                name, args = call.function.name, call.function.arguments
                log.info("tool call: %s", name)
                result = self._tools.call(name, args, confirm=confirm)
                exchange.append({"role": "tool", "tool_name": name, "content": result})
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

    def _remember(self, exchange: list[Any]) -> None:
        # max_history_messages counts user+assistant turns (2 per exchange).
        keep = self._cfg.max_history_messages // 2
        self._exchanges = (self._exchanges + [exchange])[-keep:] if keep else []
