"""Builds the tool registry from config: one place that knows every tool group."""

from __future__ import annotations

import logging

from ximz.config import Config
from ximz.memory import MemoryStore
from ximz.tools import ToolRegistry

log = logging.getLogger(__name__)


def build_registry(cfg: Config, memory: MemoryStore | None) -> ToolRegistry:
    registry = ToolRegistry()
    groups = []
    if memory is not None:
        from ximz.tools.memory import memory_tools

        groups.append(("memory", lambda: memory_tools(memory, cfg.memory.max_fact_chars)))
    if "system" in cfg.tools.enabled:
        from ximz.tools.system import system_tools

        groups.append(("system", lambda: system_tools(cfg.tools)))
    if "weather" in cfg.tools.enabled:
        from ximz.tools.weather import weather_tools

        groups.append(("weather", lambda: weather_tools(cfg.tools)))
    if "calendar" in cfg.tools.enabled:
        from ximz.tools.calendar import calendar_tools

        groups.append(("calendar", lambda: calendar_tools(cfg.tools)))

    for name, make in groups:
        try:
            for tool in make():
                registry.register(tool)
        except Exception:
            # One broken tool group must not take the assistant down.
            log.exception("could not load %s tools; continuing without them", name)
    log.info("tools: %s", ", ".join(t["function"]["name"] for t in registry.schemas()))
    return registry
