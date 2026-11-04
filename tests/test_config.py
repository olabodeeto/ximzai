from pathlib import Path

import pytest
import yaml

from ximz.config import DEFAULT_CONFIG_PATH, ConfigError, load_config


def test_repo_config_loads() -> None:
    cfg = load_config(DEFAULT_CONFIG_PATH)
    assert cfg.audio.sample_rate == 16000
    assert cfg.llm.model
    assert cfg.debug.save_audio is False


def _write(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def _base() -> dict:
    return yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())


def test_unknown_key_rejected(tmp_path: Path) -> None:
    data = _base()
    data["llm"]["modle"] = "typo"
    with pytest.raises(ConfigError, match="unknown keys"):
        load_config(_write(tmp_path, data))


def test_missing_section_rejected(tmp_path: Path) -> None:
    data = _base()
    del data["tts"]
    with pytest.raises(ConfigError, match="missing top-level"):
        load_config(_write(tmp_path, data))


def test_invalid_value_rejected(tmp_path: Path) -> None:
    data = _base()
    data["vad"]["max_command_s"] = 0
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, data))
