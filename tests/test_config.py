"""Tests for runtime configuration via environment variables (Req 13.1–13.5)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reload_settings(env: dict[str, str]):
    """Reload server.config with a patched environment and return the Settings."""
    # Remove cached module so pydantic-settings re-reads env
    for mod in list(sys.modules):
        if mod.startswith("server.config"):
            del sys.modules[mod]
    with patch.dict("os.environ", env, clear=True):
        import server.config as cfg

        return cfg.Settings()


# ---------------------------------------------------------------------------
# Req 13.1 — LOG_PATH
# ---------------------------------------------------------------------------


def test_log_path_default() -> None:
    """LOG_PATH defaults to ./logs when not set."""
    settings = _reload_settings({})
    assert settings.log_path == Path("./logs")


def test_log_path_custom() -> None:
    """LOG_PATH is read from the environment variable."""
    settings = _reload_settings({"LOG_PATH": "/tmp/custom_logs"})
    assert settings.log_path == Path("/tmp/custom_logs")


# ---------------------------------------------------------------------------
# Req 13.2 — DEFAULT_SEED
# ---------------------------------------------------------------------------


def test_default_seed_absent() -> None:
    """DEFAULT_SEED is None when not set (optional, no default required)."""
    settings = _reload_settings({})
    assert settings.default_seed is None


def test_default_seed_custom() -> None:
    """DEFAULT_SEED is read from the environment variable as an integer."""
    settings = _reload_settings({"DEFAULT_SEED": "42"})
    assert settings.default_seed == 42


# ---------------------------------------------------------------------------
# Req 13.3 — HOST and PORT
# ---------------------------------------------------------------------------


def test_host_default() -> None:
    """HOST defaults to 0.0.0.0 when not set."""
    settings = _reload_settings({})
    assert settings.host == "0.0.0.0"


def test_host_custom() -> None:
    """HOST is read from the environment variable."""
    settings = _reload_settings({"HOST": "127.0.0.1"})
    assert settings.host == "127.0.0.1"


def test_port_default() -> None:
    """PORT defaults to 8000 when not set."""
    settings = _reload_settings({})
    assert settings.port == 8000


def test_port_custom() -> None:
    """PORT is read from the environment variable as an integer."""
    settings = _reload_settings({"PORT": "9090"})
    assert settings.port == 9090


# ---------------------------------------------------------------------------
# Req 13.4 — CURRICULUM_START_TIER
# ---------------------------------------------------------------------------


def test_curriculum_start_tier_default() -> None:
    """CURRICULUM_START_TIER defaults to 0 when not set."""
    settings = _reload_settings({})
    assert settings.curriculum_start_tier == 0


def test_curriculum_start_tier_custom() -> None:
    """CURRICULUM_START_TIER is read from the environment variable as an integer."""
    settings = _reload_settings({"CURRICULUM_START_TIER": "2"})
    assert settings.curriculum_start_tier == 2


# ---------------------------------------------------------------------------
# Req 13.5 — Missing required vars exit non-zero with descriptive error
# ---------------------------------------------------------------------------


def test_invalid_port_exits_nonzero(capsys) -> None:
    """A non-integer PORT value triggers a descriptive error and sys.exit(1)."""
    import pytest

    for mod in list(sys.modules):
        if mod.startswith("server.config"):
            del sys.modules[mod]

    with patch.dict("os.environ", {"PORT": "not_a_number"}, clear=True):
        with pytest.raises(SystemExit) as exc_info:
            import server.config  # noqa: F401

        assert exc_info.value.code != 0


def test_invalid_curriculum_tier_exits_nonzero(capsys) -> None:
    """A non-integer CURRICULUM_START_TIER triggers a descriptive error and sys.exit(1)."""
    import pytest

    for mod in list(sys.modules):
        if mod.startswith("server.config"):
            del sys.modules[mod]

    with patch.dict("os.environ", {"CURRICULUM_START_TIER": "bad"}, clear=True):
        with pytest.raises(SystemExit) as exc_info:
            import server.config  # noqa: F401

        assert exc_info.value.code != 0


# ---------------------------------------------------------------------------
# End-to-end wiring: EpisodeManager uses settings
# ---------------------------------------------------------------------------


def test_episode_manager_uses_curriculum_start_tier() -> None:
    """EpisodeManager._curriculum_tier is initialised from CURRICULUM_START_TIER."""
    # Reload config with tier=3, then reload episode_manager
    for mod in list(sys.modules):
        if mod.startswith("server"):
            del sys.modules[mod]

    with patch.dict("os.environ", {"CURRICULUM_START_TIER": "3"}, clear=True):
        from server.episode_manager import EpisodeManager

        mgr = EpisodeManager()
        assert mgr._curriculum_tier == 3


def test_episode_manager_uses_default_seed() -> None:
    """EpisodeManager.reset() uses DEFAULT_SEED when no explicit seed is given."""
    for mod in list(sys.modules):
        if mod.startswith("server"):
            del sys.modules[mod]

    with patch.dict("os.environ", {"DEFAULT_SEED": "99"}, clear=True):
        from server.episode_manager import EpisodeManager

        mgr = EpisodeManager()
        mgr.reset()  # no seed argument — should use DEFAULT_SEED=99
        # The latent state seed must equal 99
        assert mgr._latent is not None
        assert mgr._latent.seed == 99


def test_episode_manager_explicit_seed_overrides_default() -> None:
    """An explicit seed passed to reset() overrides DEFAULT_SEED."""
    for mod in list(sys.modules):
        if mod.startswith("server"):
            del sys.modules[mod]

    with patch.dict("os.environ", {"DEFAULT_SEED": "99"}, clear=True):
        from server.episode_manager import EpisodeManager

        mgr = EpisodeManager()
        mgr.reset(seed=7)
        assert mgr._latent is not None
        assert mgr._latent.seed == 7


def test_logger_uses_log_path() -> None:
    """EpisodeLogger uses LOG_PATH from settings for its log directory."""
    for mod in list(sys.modules):
        if mod.startswith("server"):
            del sys.modules[mod]

    with patch.dict("os.environ", {"LOG_PATH": "/tmp/test_logs_cfg"}, clear=True):
        from server.logger import EpisodeLogger

        el = EpisodeLogger()
        assert el._log_path == Path("/tmp/test_logs_cfg")
