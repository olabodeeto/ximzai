"""Load and validate config.yaml into typed, immutable dataclasses."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
CONFIG_ENV_VAR = "XIMZ_CONFIG"


class ConfigError(ValueError):
    """Raised when config.yaml is missing, malformed or invalid."""


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int
    channels: int
    chunk_samples: int
    queue_seconds: float
    silent_mic_warn_s: float


@dataclass(frozen=True)
class VADConfig:
    threshold: float
    neg_threshold: float
    min_speech_ms: int
    silence_ms: int
    pre_roll_ms: int
    listen_timeout_s: float
    max_command_s: float


@dataclass(frozen=True)
class WakeConfig:
    greetings: list[str]
    names: list[str]
    name_similarity: float
    max_leading_words: int
    name_alone: bool
    name_alone_names: list[str]
    name_alone_similarity: float
    min_segment_s: float
    max_segment_s: float
    min_command_words: int
    greeting: str
    stt_prompt: str


@dataclass(frozen=True)
class STTConfig:
    model: str
    language: str
    vocabulary: list[str]


@dataclass(frozen=True)
class LLMConfig:
    host: str
    model: str
    keep_alive: str
    temperature: float
    request_timeout_s: float
    max_history_messages: int
    max_tool_steps: int
    startup_wait_s: float
    retry_initial_s: float
    retry_max_s: float
    system_prompt: str


@dataclass(frozen=True)
class ToolsConfig:
    enabled: list[str]
    volume_step: int
    weather_location: str
    weather_units: str
    calendar_max_days: int
    permission_timeout_s: float
    confirm_timeout_s: float
    yes_words: list[str]
    no_words: list[str]


@dataclass(frozen=True)
class ConversationConfig:
    follow_up_s: float
    end_phrases: list[str]
    end_reply: str


@dataclass(frozen=True)
class MemoryConfig:
    enabled: bool
    path: str
    max_facts: int
    max_fact_chars: int
    auto_extract: bool
    max_extract_facts: int
    extract_prompt: str


@dataclass(frozen=True)
class TTSConfig:
    backend: str
    voice: str
    speed: float
    lang: str
    model_path: str
    voices_path: str
    say_voice: str | None
    say_rate: int


@dataclass(frozen=True)
class GuiConfig:
    enabled: bool
    position: str
    width: int
    height: int
    margin: int
    fade_s: float
    hide_delay_s: float
    menu_bar_icon: bool


@dataclass(frozen=True)
class MessagesConfig:
    error: str
    llm_unavailable: str
    no_speech: str
    tool_limit: str


@dataclass(frozen=True)
class LoggingConfig:
    level: str


@dataclass(frozen=True)
class DebugConfig:
    save_audio: bool
    audio_dir: str


@dataclass(frozen=True)
class Config:
    audio: AudioConfig
    vad: VADConfig
    wake: WakeConfig
    stt: STTConfig
    llm: LLMConfig
    tools: ToolsConfig
    conversation: ConversationConfig
    memory: MemoryConfig
    tts: TTSConfig
    gui: GuiConfig
    messages: MessagesConfig
    logging: LoggingConfig
    debug: DebugConfig


T = TypeVar("T")


def _section(cls: type[T], data: dict[str, Any], name: str) -> T:
    if not isinstance(data, dict):
        raise ConfigError(f"[{name}] must be a mapping")
    expected = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    missing = expected - data.keys()
    unknown = data.keys() - expected
    if missing:
        raise ConfigError(f"[{name}] missing keys: {sorted(missing)}")
    if unknown:
        raise ConfigError(f"[{name}] unknown keys: {sorted(unknown)}")
    return cls(**data)


def _validate(cfg: Config) -> None:
    if cfg.audio.sample_rate <= 0 or cfg.audio.channels <= 0:
        raise ConfigError("audio.sample_rate and audio.channels must be positive")
    if cfg.audio.chunk_samples <= 0 or cfg.audio.queue_seconds <= 0:
        raise ConfigError("audio.chunk_samples and audio.queue_seconds must be positive")
    if not 0 < cfg.vad.neg_threshold <= cfg.vad.threshold < 1:
        raise ConfigError("need 0 < vad.neg_threshold <= vad.threshold < 1")
    if min(cfg.vad.silence_ms, cfg.vad.listen_timeout_s, cfg.vad.max_command_s) <= 0:
        raise ConfigError("vad.silence_ms, listen_timeout_s and max_command_s must be positive")
    if not cfg.wake.greetings or not cfg.wake.names:
        raise ConfigError("wake.greetings and wake.names must not be empty")
    if not 0 < cfg.wake.name_similarity <= 1:
        raise ConfigError("wake.name_similarity must be in (0, 1]")
    if not 0 < cfg.wake.min_segment_s < cfg.wake.max_segment_s:
        raise ConfigError("need 0 < wake.min_segment_s < wake.max_segment_s")
    unknown_tools = set(cfg.tools.enabled) - {"system", "weather", "calendar"}
    if unknown_tools:
        raise ConfigError(f"tools.enabled has unknown groups: {sorted(unknown_tools)}")
    word_lists = {
        "tools.yes_words": cfg.tools.yes_words, "tools.no_words": cfg.tools.no_words,
        "conversation.end_phrases": cfg.conversation.end_phrases,
        "wake.greetings": cfg.wake.greetings, "wake.names": cfg.wake.names,
    }
    for key, words in word_lists.items():
        if not all(isinstance(w, str) for w in words):
            raise ConfigError(f"{key} must contain only quoted strings (YAML reads bare yes/no/on/off as booleans)")
    if cfg.tools.weather_units not in {"celsius", "fahrenheit"}:
        raise ConfigError("tools.weather_units must be 'celsius' or 'fahrenheit'")
    if min(cfg.tools.volume_step, cfg.tools.calendar_max_days, cfg.tools.confirm_timeout_s) <= 0:
        raise ConfigError("tools.volume_step, calendar_max_days and confirm_timeout_s must be positive")
    if cfg.llm.max_tool_steps <= 0:
        raise ConfigError("llm.max_tool_steps must be positive")
    if cfg.conversation.follow_up_s < 0:
        raise ConfigError("conversation.follow_up_s must be >= 0 (0 disables follow-ups)")
    if cfg.memory.max_facts <= 0 or cfg.memory.max_fact_chars <= 0:
        raise ConfigError("memory.max_facts and memory.max_fact_chars must be positive")
    if not 0 < cfg.llm.retry_initial_s <= cfg.llm.retry_max_s:
        raise ConfigError("need 0 < llm.retry_initial_s <= llm.retry_max_s")
    if cfg.tts.backend not in {"say", "kokoro"}:
        raise ConfigError(f"tts.backend must be 'kokoro' or 'say', not '{cfg.tts.backend}'")
    if cfg.tts.say_rate <= 0 or not 0.5 <= cfg.tts.speed <= 2.0:
        raise ConfigError("tts.say_rate must be positive and tts.speed within 0.5–2.0")
    if cfg.llm.max_history_messages < 0:
        raise ConfigError("llm.max_history_messages must be >= 0")
    if cfg.gui.position not in {"center", "top", "bottom"}:
        raise ConfigError("gui.position must be 'center', 'top' or 'bottom'")
    if cfg.gui.width <= 0 or cfg.gui.height <= 0:
        raise ConfigError("gui.width and gui.height must be positive")
    if cfg.gui.margin < 0:
        raise ConfigError("gui.margin must be >= 0")
    if cfg.gui.fade_s < 0 or cfg.gui.hide_delay_s < 0:
        raise ConfigError("gui.fade_s and gui.hide_delay_s must be >= 0")
    if cfg.llm.request_timeout_s <= 0:
        raise ConfigError("llm.request_timeout_s must be positive")


def load_config(path: str | Path | None = None) -> Config:
    """Load config from `path`, $XIMZ_CONFIG, or the project's config.yaml."""
    resolved = Path(path or os.environ.get(CONFIG_ENV_VAR) or DEFAULT_CONFIG_PATH)
    try:
        raw = yaml.safe_load(resolved.read_text())
    except FileNotFoundError as e:
        raise ConfigError(f"config file not found: {resolved}") from e
    except yaml.YAMLError as e:
        raise ConfigError(f"invalid YAML in {resolved}: {e}") from e
    if not isinstance(raw, dict):
        raise ConfigError(f"{resolved} must contain a mapping at the top level")

    section_classes = {
        "audio": AudioConfig,
        "vad": VADConfig,
        "wake": WakeConfig,
        "stt": STTConfig,
        "llm": LLMConfig,
        "tools": ToolsConfig,
        "conversation": ConversationConfig,
        "memory": MemoryConfig,
        "tts": TTSConfig,
        "gui": GuiConfig,
        "messages": MessagesConfig,
        "logging": LoggingConfig,
        "debug": DebugConfig,
    }
    unknown = raw.keys() - section_classes.keys()
    if unknown:
        raise ConfigError(f"unknown top-level sections: {sorted(unknown)}")
    missing = section_classes.keys() - raw.keys()
    if missing:
        raise ConfigError(f"missing top-level sections: {sorted(missing)}")

    cfg = Config(**{name: _section(cls, raw[name], name) for name, cls in section_classes.items()})
    _validate(cfg)
    return cfg
