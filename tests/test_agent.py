from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ximz.agent import Agent, LLMUnavailableError, ToolLimitError
from ximz.config import load_config
from ximz.memory import MemoryStore
from ximz.tools import ToolRegistry
from ximz.tools.memory import memory_tools


def msg(content: str = "", tool_calls: list | None = None) -> Any:
    return SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls))


def call(name: str, **args: Any) -> Any:
    return SimpleNamespace(function=SimpleNamespace(name=name, arguments=args))


class FakeClient:
    def __init__(self, replies: list[Any] | None = None, fail: bool = False) -> None:
        self.replies = list(replies or [])
        self.fail = fail
        self.calls: list[dict[str, Any]] = []

    def chat(self, **kwargs: Any) -> Any:
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        if self.fail:
            raise ConnectionError("refused")
        return self.replies.pop(0) if self.replies else msg("Hello there.")


def test_respond_sends_system_prompt_and_keeps_history() -> None:
    cfg = load_config().llm
    client = FakeClient()
    agent = Agent(cfg, client=client)

    assert agent.respond("hi") == "Hello there."
    agent.respond("again")

    msgs = client.calls[-1]["messages"]
    assert msgs[0]["role"] == "system" and cfg.system_prompt in msgs[0]["content"]
    assert [m["content"] for m in msgs[1:]] == ["hi", "Hello there.", "again"]
    assert client.calls[-1]["keep_alive"] == cfg.keep_alive


def test_connection_error_becomes_llm_unavailable() -> None:
    agent = Agent(load_config().llm, client=FakeClient(fail=True))
    with pytest.raises(LLMUnavailableError):
        agent.respond("hi")


def _memory_agent(tmp_path: Path, client: FakeClient) -> tuple[Agent, MemoryStore]:
    cfg = load_config()
    store = MemoryStore(replace(cfg.memory, path=str(tmp_path / "memory.json")))
    reg = ToolRegistry()
    for t in memory_tools(store, cfg.memory.max_fact_chars):
        reg.register(t)
    return Agent(cfg.llm, client=client, tools=reg, memory=store), store


def test_tool_loop_runs_tool_then_answers(tmp_path: Path) -> None:
    client = FakeClient([
        msg(tool_calls=[call("remember", fact="The user's name is Ada.")]),
        msg("Nice to meet you, Ada."),
    ])
    agent, store = _memory_agent(tmp_path, client)

    assert agent.respond("My name is Ada") == "Nice to meet you, Ada."
    assert store.facts() == ["The user's name is Ada."]
    tool_msg = client.calls[1]["messages"][-1]
    assert tool_msg["role"] == "tool" and tool_msg["content"].startswith("Remembered")
    assert client.calls[0]["tools"]  # schemas were sent

    # Next request: the remembered fact is in the system prompt.
    agent.respond("What's my name?")
    assert "Ada" in client.calls[-1]["messages"][0]["content"]


@pytest.mark.parametrize(
    ("fact", "useful"),
    [
        ("the user has a name", False),
        ("...", False),
        ("The user likes rice", True),
        ("The user's name is Ada.", True),
        ("Gym at six", False),  # too short to be a self-contained fact
    ],
)
def test_is_useful_fact(fact: str, useful: bool) -> None:
    from ximz.agent import is_useful_fact

    assert is_useful_fact(fact) is useful


@pytest.mark.parametrize(
    ("reply", "claim"),
    [("Opening Spotify for you.", True), ("I have added a reminder to call Funmi.", True),
     ("I've muted the sound.", True), ("The sound is now muted.", True),
     ("The capital of Nigeria is Abuja.", False), ("You have a meeting at ten.", False),
     ("Yes Olabode, I can hear you.", False)],
)
def test_claims_action(reply: str, claim: bool) -> None:
    from ximz.agent import claims_action

    assert claims_action(reply) is claim


@pytest.mark.parametrize(
    ("reply", "tidy"),
    [("Spotify is open. Is there anything else I can help with?", "Spotify is open."),
     ("It's 6 PM. How else can I assist you today?", "It's 6 PM."),
     ("Is there anything else?", "Is there anything else?"),  # never empty the reply
     ("Abuja is the capital.", "Abuja is the capital.")],
)
def test_tidy_reply(reply: str, tidy: str) -> None:
    from ximz.agent import tidy_reply

    assert tidy_reply(reply) == tidy


def test_false_action_claim_is_nudged_into_a_tool_call(tmp_path: Path) -> None:
    client = FakeClient([
        msg("I have added that to memory."),                       # claim, no tool call
        msg(tool_calls=[call("remember", fact="The user likes tea.")]),
        msg("Got it, you like tea."),
    ])
    agent, store = _memory_agent(tmp_path, client)
    assert agent.respond("I like tea") == "Got it, you like tea."
    assert store.facts() == ["The user likes tea."]
    # The false claim and the nudge are not kept in history.
    agent.respond("hi")
    history = [m.get("content") if isinstance(m, dict) else m.content for m in client.calls[-1]["messages"]]
    assert "I have added that to memory." not in history


def test_tool_loop_is_capped(tmp_path: Path) -> None:
    looping = [msg(tool_calls=[call("forget", about="x" * 3)]) for _ in range(20)]
    agent, _ = _memory_agent(tmp_path, FakeClient(looping))
    with pytest.raises(ToolLimitError):
        agent.respond("loop forever")
