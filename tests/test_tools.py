from datetime import date, datetime
from typing import Any

import pytest

from ximz.config import load_config
from ximz.conversation import is_yes
from ximz.tools import Tool, ToolRegistry
from ximz.tools import calendar as cal
from ximz.tools import system, weather

CFG = load_config()


# -- confirmation flow -------------------------------------------------------

def _risky(ran: list[str]) -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(Tool(
        name="quit_app", description="", risk="confirm",
        parameters={"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
        func=lambda name: ran.append(name) or f"Quit {name}.",
        confirm_prompt=lambda name: f"Quit {name}?",
    ))
    return reg


def test_confirm_tool_runs_only_after_yes() -> None:
    ran: list[str] = []
    reg = _risky(ran)
    asked: list[str] = []
    assert "said no" in reg.call("quit_app", {"name": "Safari"}, confirm=lambda q: asked.append(q) or False)
    assert ran == [] and asked == ["Quit Safari?"]
    assert reg.call("quit_app", {"name": "Safari"}, confirm=lambda q: True) == "Quit Safari."
    assert ran == ["Safari"]


def test_confirm_tool_refused_without_confirmer() -> None:
    ran: list[str] = []
    assert reg_result_is_error(_risky(ran).call("quit_app", {"name": "Safari"}))
    assert ran == []


def reg_result_is_error(text: str) -> bool:
    return text.startswith("Error")


@pytest.mark.parametrize(
    ("answer", "ok"),
    [("Yes.", True), ("Yeah, go ahead.", True), ("Sure!", True), ("No.", False),
     ("Yes, actually no wait.", False), ("Don't.", False), ("", False), (None, False), ("Banana.", False)],
)
def test_is_yes(answer: str | None, ok: bool) -> None:
    assert is_yes(answer, CFG.tools.yes_words, CFG.tools.no_words) is ok


# -- system ------------------------------------------------------------------

def test_parse_battery() -> None:
    out = "Now drawing from 'AC Power'\n -InternalBattery-0 (id=1)\t100%; charged; 0:00 remaining present: true"
    assert system.parse_battery(out) == "Battery is at 100 percent, charged, on power adapter."
    assert "doesn't report" in system.parse_battery("Now drawing from 'AC Power'")


@pytest.mark.parametrize(
    ("spoken", "expected"),
    [("chrome", "Google Chrome"), ("Spotify", "Spotify"), ("slak", "Slack"), ("vs code", "Visual Studio Code"),
     ("whatsapp", "WhatsApp"), ("photoshop", None)],
)
def test_match_app(spoken: str, expected: str | None) -> None:
    names = ["Google Chrome", "Spotify", "Slack", "Visual Studio Code", "WhatsApp", "Safari"]
    assert system.match_app(spoken, names) == expected


def test_match_running_app_short_name() -> None:
    assert system.match_app("vs code", ["Code", "Terminal", "Finder"]) == "Code"


def test_volume_rejects_bad_action_via_schema() -> None:
    reg = ToolRegistry()
    for t in system.system_tools(CFG.tools):
        reg.register(t)
    assert reg.call("volume", {"action": "explode"}).startswith("Error")
    assert reg.call("volume", {"action": "set", "level": "loud"}).startswith("Error")


# -- weather -----------------------------------------------------------------

def fake_fetch(url: str, params: dict[str, Any]) -> dict[str, Any]:
    if "geocoding" in url:
        return {"results": [{"name": "Lagos", "country": "Nigeria", "latitude": 6.45, "longitude": 3.39}]
                if params["name"].lower() == "lagos" else []}
    return {
        "current": {"temperature_2m": 30.4, "apparent_temperature": 35.1, "weather_code": 2},
        "daily": {"temperature_2m_max": [32.0, 31.2], "temperature_2m_min": [25.0, 24.6],
                  "precipitation_probability_max": [40, 70], "weather_code": [80, 61]},
    }


def test_weather_today_and_tomorrow() -> None:
    today = weather.weather("Lagos", "today", CFG.tools, fetch=fake_fetch)
    assert "Lagos, Nigeria" in today and "partly cloudy" in today and "30 degrees" in today
    tomorrow = weather.weather("Lagos", "tomorrow", CFG.tools, fetch=fake_fetch)
    assert "light rain" in tomorrow and "70 percent" in tomorrow


def test_weather_unknown_place_and_no_default() -> None:
    assert "couldn't find" in weather.weather("Atlantis", "today", CFG.tools, fetch=fake_fetch)
    assert "call weather again" in weather.weather(None, "today", CFG.tools, fetch=fake_fetch)


@pytest.mark.parametrize(
    ("text", "tool"),
    [("Remind me to call Funmi at 6", "add_reminder"), ("Mute the sound", "volume"),
     ("What's the weather in Lagos?", "weather"), ("Open Spotify", "open_app"),
     ("What's on my calendar today?", "calendar_events"), ("How much battery is left?", "battery"),
     ("Quit Safari", "quit_app")],
)
def test_tool_hints(text: str, tool: str) -> None:
    reg = ToolRegistry()
    for t in (system.system_tools(CFG.tools) + weather.weather_tools(CFG.tools)
              + cal.calendar_tools(CFG.tools, store=FakeEK())):  # type: ignore[arg-type]
        reg.register(t)
    assert tool in reg.suggest(text)
    assert reg.suggest("What's the capital of Nigeria?") == []


# -- calendar / reminders ------------------------------------------------------

def test_parse_day() -> None:
    today = date(2026, 11, 4)
    assert cal.parse_day("today", today) == today
    assert cal.parse_day("tomorrow", today) == date(2026, 11, 5)
    assert cal.parse_day("2026-12-25", today) == date(2026, 12, 25)
    with pytest.raises(ValueError):
        cal.parse_day("next blursday", today)


def test_parse_due() -> None:
    now = datetime(2026, 11, 4, 12, 0)
    assert cal.parse_due("2026-11-04 18:00", now) == datetime(2026, 11, 4, 18, 0)
    assert cal.parse_due("2026-11-05T09:30", now) == datetime(2026, 11, 5, 9, 30)
    with pytest.raises(ValueError, match="past"):
        cal.parse_due("2026-11-03 18:00", now)


class FakeEK:
    def __init__(self) -> None:
        self.added: list[tuple[str, str | None]] = []

    def events(self, day: str, days: int) -> str:
        return f"events {day} {days}"

    def reminders(self) -> str:
        return "No open reminders."

    def add_reminder(self, title: str, due: str | None) -> str:
        self.added.append((title, due))
        return f"Reminder set: {title}."


def test_calendar_tools_validate_and_clamp() -> None:
    ek = FakeEK()
    reg = ToolRegistry()
    for t in cal.calendar_tools(CFG.tools, store=ek):  # type: ignore[arg-type]
        reg.register(t)
    assert reg.call("calendar_events", {}) == "events today 1"
    assert reg.call("calendar_events", {"day": "tomorrow", "days": 999}) == f"events tomorrow {CFG.tools.calendar_max_days}"
    assert reg.call("add_reminder", {"title": "Call Funmi", "due": "2026-11-04 18:00"}).startswith("Reminder set")
    assert reg.call("add_reminder", {"due": "2026-11-04 18:00"}).startswith("Error")  # missing title
    assert ek.added == [("Call Funmi", "2026-11-04 18:00")]
