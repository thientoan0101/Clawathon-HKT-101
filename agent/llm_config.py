"""Resolve LLM settings from environment / .env."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"
_dotenv_loaded = False


def ensure_env_loaded() -> None:
    global _dotenv_loaded
    if not _dotenv_loaded:
        load_dotenv(_ENV_FILE)
        _dotenv_loaded = True


def get_llm_settings() -> tuple[str, str, str]:
    """Return (model, base_url, api_key). Raises ValueError if incomplete."""
    ensure_env_loaded()

    model = os.environ.get("LLM_MODEL", "").strip()
    base_url = os.environ.get("LLM_BASE_URL", "").strip()
    api_key = (
        os.environ.get("LLM_API_KEY", "").strip()
        or os.environ.get("AIP_API_KEY", "").strip()
        or os.environ.get("OPENAI_API_KEY", "").strip()
    )

    missing = [
        name
        for name, value in [
            ("LLM_MODEL", model),
            ("LLM_BASE_URL", base_url),
            ("LLM_API_KEY", api_key),
        ]
        if not value
    ]
    if missing:
        raise ValueError(
            "Missing LLM configuration: "
            + ", ".join(missing)
            + ". Add them to .env (see .env.example). "
            "For GreenNode AIP use /agentbase-llm to create an API key."
        )

    return model, base_url, api_key


def llm_config_status() -> dict[str, bool]:
    """Non-secret status for startup logs."""
    ensure_env_loaded()
    api_key = (
        os.environ.get("LLM_API_KEY", "").strip()
        or os.environ.get("AIP_API_KEY", "").strip()
        or os.environ.get("OPENAI_API_KEY", "").strip()
    )
    return {
        "LLM_MODEL": bool(os.environ.get("LLM_MODEL", "").strip()),
        "LLM_BASE_URL": bool(os.environ.get("LLM_BASE_URL", "").strip()),
        "LLM_API_KEY": bool(api_key),
    }
