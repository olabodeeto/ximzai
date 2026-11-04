"""Tool registry: each tool is one function + one JSON schema + a risk level.

The registry validates arguments against the schema before calling anything —
model output is never trusted as-is.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

log = logging.getLogger(__name__)

Risk = Literal["safe", "confirm"]

_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
}


class ToolArgumentError(ValueError):
    pass


Confirmer = Callable[[str], bool]  # asks the user a yes/no question out loud


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema, type "object"
    risk: Risk
    func: Callable[..., str]
    # For risk="confirm": builds the spoken question from the validated args.
    confirm_prompt: Callable[..., str] | None = None
    # Regexes over the user's words that strongly suggest this tool. Used to
    # hint small models that otherwise sometimes *claim* an action instead.
    triggers: tuple[str, ...] = ()

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }


def validate_args(schema: dict[str, Any], args: Any) -> dict[str, Any]:
    """Minimal JSON-schema check for flat objects of primitives."""
    if not isinstance(args, dict):
        raise ToolArgumentError("arguments must be an object")
    props: dict[str, Any] = schema.get("properties", {})
    unknown = set(args) - set(props)
    if unknown:
        raise ToolArgumentError(f"unknown arguments: {sorted(unknown)}")
    for key in schema.get("required", []):
        if key not in args:
            raise ToolArgumentError(f"missing argument: {key}")
    for key, value in args.items():
        spec = props[key]
        types = _JSON_TYPES.get(spec.get("type", ""), ())
        # bool is a subclass of int in Python; don't let True pass as a number.
        if types and (not isinstance(value, types) or (isinstance(value, bool) and bool not in types)):
            raise ToolArgumentError(f"{key} must be {spec['type']}")
        if isinstance(value, str):
            if len(value.strip()) < spec.get("minLength", 0):
                raise ToolArgumentError(f"{key} is too short")
            if len(value) > spec.get("maxLength", 10_000):
                raise ToolArgumentError(f"{key} is too long")
        if "enum" in spec and value not in spec["enum"]:
            raise ToolArgumentError(f"{key} must be one of {spec['enum']}")
    return args


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool: {tool.name}")
        self._tools[tool.name] = tool

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    def suggest(self, user_text: str) -> list[str]:
        """Names of tools whose trigger phrases appear in the user's words."""
        return [
            t.name for t in self._tools.values()
            if any(re.search(p, user_text, re.IGNORECASE) for p in t.triggers)
        ]

    def call(self, name: str, args: Any, confirm: Confirmer | None = None) -> str:
        """Run a tool and return its result text. Errors become text the model can read.

        `confirm`-risk tools only run after `confirm(question)` returns True."""
        tool = self._tools.get(name)
        if tool is None:
            return f"Error: there is no tool called {name}."
        try:
            clean = validate_args(tool.parameters, args)
            if tool.risk == "confirm":
                if confirm is None:
                    return f"Error: {name} needs the user's spoken confirmation, which isn't available here."
                question = tool.confirm_prompt(**clean) if tool.confirm_prompt else f"Should I run {name}?"
                if not confirm(question):
                    log.info("tool %s declined by user", name)
                    return "The user said no, so nothing was done. Acknowledge briefly."
            result = tool.func(**clean)
        except ToolArgumentError as e:
            log.warning("tool %s rejected arguments: %s", name, e)
            return f"Error: invalid arguments for {name}: {e}"
        except Exception as e:
            log.exception("tool %s failed", name)
            return f"Error: {name} failed: {e}"
        log.info("tool %s ok", name)
        return result
