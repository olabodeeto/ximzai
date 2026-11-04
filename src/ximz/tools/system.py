"""System tools: battery, volume, opening and quitting apps.

No shell strings anywhere: commands run as argv lists, and app names are only
ever ones found on disk / currently running, never raw model text.
"""

from __future__ import annotations

import difflib
import logging
import re
import subprocess
from pathlib import Path

from ximz.config import ToolsConfig
from ximz.tools import Tool

log = logging.getLogger(__name__)

APP_DIRS = (
    Path("/Applications"),
    Path("/Applications/Utilities"),
    Path("/System/Applications"),
    Path("/System/Applications/Utilities"),
    Path.home() / "Applications",
)
_RUN_TIMEOUT_S = 10
_BATTERY = re.compile(r"(\d+)%;\s*([^;]+);")
# Spoken nicknames → real app names.
_APP_ALIASES = {"chrome": "Google Chrome", "vs code": "Visual Studio Code", "vscode": "Visual Studio Code",
                "code": "Visual Studio Code", "excel": "Microsoft Excel", "word": "Microsoft Word",
                "settings": "System Settings", "browser": "Safari", "music": "Music"}


def _run(argv: list[str]) -> str:
    return subprocess.run(argv, capture_output=True, text=True, check=True, timeout=_RUN_TIMEOUT_S).stdout.strip()


def _osascript(lines: list[str], *argv: str) -> str:
    cmd = ["osascript"]
    for line in lines:
        cmd += ["-e", line]
    return _run(cmd + list(argv))


# -- battery ------------------------------------------------------------------

def parse_battery(pmset_output: str) -> str:
    m = _BATTERY.search(pmset_output)
    if not m:
        return "This Mac doesn't report a battery."
    percent, status = m.group(1), m.group(2).strip()
    source = "on power adapter" if "AC Power" in pmset_output else "on battery"
    return f"Battery is at {percent} percent, {status}, {source}."


def battery() -> str:
    return parse_battery(_run(["pmset", "-g", "batt"]))


# -- volume -------------------------------------------------------------------

def volume(action: str, level: int | None = None, step: int = 10) -> str:
    current = int(_osascript(["output volume of (get volume settings)"]) or 0)
    if action == "get":
        muted = _osascript(["output muted of (get volume settings)"]) == "true"
        return f"Volume is {current} percent" + (", muted." if muted else ".")
    if action == "mute":
        _osascript(["set volume output muted true"])
        return "Muted."
    if action == "unmute":
        _osascript(["set volume output muted false"])
        return f"Unmuted, volume {current} percent."
    if action == "set":
        if level is None:
            return "Error: 'set' needs a level from 0 to 100."
        target = level
    else:  # up / down
        target = current + step if action == "up" else current - step
    target = max(0, min(100, target))
    _osascript(["on run argv", "set volume output volume (item 1 of argv as integer)",
                "set volume output muted false", "end run"], str(target))
    return f"Volume set to {target} percent."


# -- apps ---------------------------------------------------------------------

def installed_apps() -> dict[str, Path]:
    apps: dict[str, Path] = {}
    for d in APP_DIRS:
        if d.is_dir():
            for p in d.glob("*.app"):
                apps.setdefault(p.stem, p)
    return apps


def match_app(spoken: str, names: list[str]) -> str | None:
    """Map a spoken app name to a real one ("chrome" → "Google Chrome")."""
    key = spoken.lower().strip().removesuffix(".app")
    key = _APP_ALIASES.get(key, key)
    lowered = {n.lower(): n for n in names}
    if key.lower() in lowered:
        return lowered[key.lower()]
    contains = [n for n in names if key.lower() in n.lower()]
    if len(contains) == 1:
        return contains[0]
    # Running apps can have short names: VS Code runs as "Code".
    within = [n for n in names if len(n) > 2 and f" {n.lower()} " in f" {key.lower()} "]
    if len(within) == 1:
        return within[0]
    close = difflib.get_close_matches(key.lower(), list(lowered), n=1, cutoff=0.6)
    return lowered[close[0]] if close else None


def open_app(name: str) -> str:
    apps = installed_apps()
    match = match_app(name, list(apps))
    if match is None:
        return f"I couldn't find an app called {name} on this Mac."
    _run(["open", str(apps[match])])
    return f"Opened {match}."


def running_apps() -> list[str]:
    from AppKit import NSApplicationActivationPolicyRegular, NSWorkspace

    return [
        str(a.localizedName()) for a in NSWorkspace.sharedWorkspace().runningApplications()
        if a.activationPolicy() == NSApplicationActivationPolicyRegular and a.localizedName()
    ]


def quit_app(name: str) -> str:
    match = match_app(name, running_apps())
    if match is None:
        return f"{name} isn't running."
    _osascript(["on run argv", "tell application (item 1 of argv) to quit", "end run"], match)
    return f"Quit {match}."


def quit_prompt(name: str) -> str:
    match = match_app(name, running_apps()) or name
    return f"Do you want me to quit {match}? Any unsaved work could be lost."


def system_tools(cfg: ToolsConfig) -> list[Tool]:
    return [
        Tool(
            name="battery",
            description="Get the Mac's battery level and whether it is charging.",
            parameters={"type": "object", "properties": {}},
            risk="safe",
            func=battery,
            triggers=(r"\bbattery\b", r"\bcharg(e|ing)\b"),
        ),
        Tool(
            name="volume",
            description="Get or change the Mac's sound volume.",
            parameters={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["get", "set", "up", "down", "mute", "unmute"]},
                    "level": {"type": "integer", "description": "0-100, only for action 'set'."},
                },
                "required": ["action"],
            },
            risk="safe",
            func=lambda action, level=None: volume(action, level, cfg.volume_step),
            triggers=(r"\bvolume\b", r"\b(un)?mute\b", r"\b(louder|quieter|softer)\b", r"\bturn (it|the sound) (up|down)\b"),
        ),
        Tool(
            name="open_app",
            description="Open an application on the Mac, e.g. Safari, Spotify, Slack.",
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string", "minLength": 2, "maxLength": 60}},
                "required": ["name"],
            },
            risk="safe",
            func=open_app,
            triggers=(r"\b(open|launch|start)\s+\w+",),
        ),
        Tool(
            name="quit_app",
            description="Quit a running application on the Mac.",
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string", "minLength": 2, "maxLength": 60}},
                "required": ["name"],
            },
            risk="confirm",
            func=quit_app,
            triggers=(r"\b(quit|close|exit)\s+\w+",),
            confirm_prompt=quit_prompt,
        ),
    ]
