import json
import stat
from dataclasses import replace
from pathlib import Path

from ximz.config import load_config
from ximz.memory import MemoryStore
from ximz.tools import ToolRegistry
from ximz.tools.memory import memory_tools


def store(tmp_path: Path, **overrides) -> MemoryStore:
    cfg = replace(load_config().memory, path=str(tmp_path / "mem" / "memory.json"), **overrides)
    return MemoryStore(cfg)


def test_add_persists_privately_and_dedupes(tmp_path: Path) -> None:
    s = store(tmp_path)
    s.add("The user's name is Ada.")
    assert "Already" in s.add("the user's name is ada.")
    assert MemoryStore(replace(load_config().memory, path=str(s.path))).facts() == ["The user's name is Ada."]
    assert stat.S_IMODE(s.path.stat().st_mode) == 0o600
    assert "Ada" in s.prompt_block()


def test_forget_and_cap(tmp_path: Path) -> None:
    s = store(tmp_path, max_facts=2)
    s.add("The user likes jollof rice.")
    s.add("The user's birthday is 3 May.")
    s.add("The user lives in Lagos.")
    assert s.facts() == ["The user's birthday is 3 May.", "The user lives in Lagos."]
    assert "Forgot 1" in s.forget("birthday")
    assert s.facts() == ["The user lives in Lagos."]
    assert "Nothing" in s.forget("cats")


def test_empty_and_corrupt_files(tmp_path: Path) -> None:
    s = store(tmp_path)
    assert "don't know anything" in s.prompt_block()
    s.path.parent.mkdir(parents=True, exist_ok=True)
    s.path.write_text("{not json")
    assert MemoryStore(replace(load_config().memory, path=str(s.path))).facts() == []


def test_memory_tools_validate_arguments(tmp_path: Path) -> None:
    s = store(tmp_path)
    reg = ToolRegistry()
    for t in memory_tools(s, 50):
        reg.register(t)
    assert reg.call("remember", {"fact": "The user's name is Ada."}).startswith("Remembered")
    assert reg.call("remember", {"fact": 42}).startswith("Error")
    assert reg.call("remember", {"fact": "x" * 51}).startswith("Error")
    assert reg.call("remember", {"fact": "ok", "extra": 1}).startswith("Error")
    assert reg.call("remember", {}).startswith("Error")
    assert reg.call("format_disk", {}).startswith("Error")
    assert json.loads(s.path.read_text())["facts"][0]["text"] == "The user's name is Ada."
