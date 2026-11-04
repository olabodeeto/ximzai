"""Persistent facts about the user, injected into the system prompt.

Stored as a small JSON file (chmod 600) outside the repo. Only facts the model
explicitly saves via the `remember` tool are stored — never raw transcripts.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from pathlib import Path

from ximz.config import MemoryConfig

log = logging.getLogger(__name__)


class MemoryStore:
    def __init__(self, cfg: MemoryConfig) -> None:
        self._cfg = cfg
        self._path = Path(os.path.expanduser(cfg.path))
        self._lock = threading.Lock()
        self._facts: list[dict[str, str]] = self._load()

    @property
    def path(self) -> Path:
        return self._path

    def _load(self) -> list[dict[str, str]]:
        try:
            data = json.loads(self._path.read_text())
        except FileNotFoundError:
            return []
        except (OSError, json.JSONDecodeError):
            log.exception("could not read memory file %s; starting empty", self._path)
            return []
        facts = data.get("facts", []) if isinstance(data, dict) else []
        return [f for f in facts if isinstance(f, dict) and isinstance(f.get("text"), str)]

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self._path.parent, prefix=".memory-")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump({"facts": self._facts}, f, indent=2, ensure_ascii=False)
            os.chmod(tmp, 0o600)
            os.replace(tmp, self._path)  # atomic: never leaves a half-written file
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def facts(self) -> list[str]:
        with self._lock:
            return [f["text"] for f in self._facts]

    def add(self, text: str) -> str:
        text = " ".join(text.split())[: self._cfg.max_fact_chars]
        if not text:
            raise ValueError("fact is empty")
        with self._lock:
            if any(f["text"].lower() == text.lower() for f in self._facts):
                return f"Already remembered: {text}"
            self._facts.append({"text": text, "added": time.strftime("%Y-%m-%d")})
            self._facts = self._facts[-self._cfg.max_facts:]
            self._save()
        log.info("remembered a new fact (%d total)", len(self._facts))
        return f"Remembered: {text}"

    def forget(self, about: str) -> str:
        needle = about.lower().strip()
        if not needle:
            raise ValueError("nothing to forget")
        with self._lock:
            kept = [f for f in self._facts if needle not in f["text"].lower()]
            removed = len(self._facts) - len(kept)
            if removed:
                self._facts = kept
                self._save()
        log.info("forgot %d fact(s)", removed)
        return f"Forgot {removed} fact(s)." if removed else "Nothing matching that was remembered."

    def prompt_block(self) -> str:
        facts = self.facts()
        if not facts:
            return "You don't know anything about the user yet."
        return "What you remember about the user:\n" + "\n".join(f"- {f}" for f in facts)
