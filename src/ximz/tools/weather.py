"""Weather via Open-Meteo (free, no API key). Only a place name/coordinates
leave the Mac — never audio or conversation text."""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Any

from ximz.config import ToolsConfig
from ximz.tools import Tool

log = logging.getLogger(__name__)

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_TIMEOUT_S = 8

# WMO weather interpretation codes → spoken description.
WMO = {
    0: "clear skies", 1: "mostly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "freezing fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 56: "freezing drizzle", 57: "freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain", 66: "freezing rain", 67: "freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains", 80: "light showers",
    81: "showers", 82: "heavy showers", 85: "snow showers", 86: "heavy snow showers",
    95: "thunderstorms", 96: "thunderstorms with hail", 99: "thunderstorms with hail",
}

Fetch = Callable[[str, dict[str, Any]], dict[str, Any]]


def http_get_json(url: str, params: dict[str, Any]) -> dict[str, Any]:
    full = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(full, headers={"User-Agent": "ximz-voice-assistant"})
    with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:  # noqa: S310 (fixed https host)
        return json.loads(resp.read().decode())


def weather(place: str | None, day: str, cfg: ToolsConfig, fetch: Fetch = http_get_json) -> str:
    place = (place or cfg.weather_location).strip()
    if not place:
        return ("No place was given. If the user named a city, call weather again with place set to "
                "that city; otherwise ask the user which city.")
    geo = fetch(GEOCODE_URL, {"name": place, "count": 1, "language": "en", "format": "json"})
    results = geo.get("results") or []
    if not results:
        return f"I couldn't find a place called {place}."
    loc = results[0]
    name = ", ".join(x for x in (loc.get("name"), loc.get("country")) if x)
    unit = "fahrenheit" if cfg.weather_units == "fahrenheit" else "celsius"
    data = fetch(FORECAST_URL, {
        "latitude": loc["latitude"], "longitude": loc["longitude"], "timezone": "auto",
        "temperature_unit": unit, "forecast_days": 2,
        "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,weather_code",
    })
    daily = data["daily"]
    i = 1 if day == "tomorrow" else 0
    high, low = round(daily["temperature_2m_max"][i]), round(daily["temperature_2m_min"][i])
    rain = daily["precipitation_probability_max"][i]
    desc = WMO.get(daily["weather_code"][i], "mixed conditions")
    deg = "degrees"
    if day == "tomorrow":
        return (f"Tomorrow in {name}: {desc}, high of {high} and low of {low} {deg}, "
                f"{rain} percent chance of rain.")
    cur = data["current"]
    now = WMO.get(cur["weather_code"], "mixed conditions")
    return (f"Right now in {name}: {now}, {round(cur['temperature_2m'])} {deg}, feels like "
            f"{round(cur['apparent_temperature'])}. Today: high {high}, low {low}, "
            f"{rain} percent chance of rain.")


def weather_tools(cfg: ToolsConfig) -> list[Tool]:
    return [
        Tool(
            name="weather",
            description=("Current weather or tomorrow's forecast for a city. Leave place empty to use "
                         "the user's home city if you know it from memory or settings."),
            parameters={
                "type": "object",
                "properties": {
                    "place": {"type": "string", "maxLength": 80,
                              "description": "City name, e.g. Lagos. Always pass the city if the user named one."},
                    "day": {"type": "string", "enum": ["today", "tomorrow"]},
                },
                "required": [],
            },
            risk="safe",
            func=lambda place=None, day="today": weather(place, day, cfg),
            triggers=(r"\bweather\b", r"\b(rain|raining|temperature|forecast|sunny|hot|cold)\b"),
        ),
    ]
