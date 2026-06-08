from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI


DEFAULT_BASE_URL = ""
DEFAULT_MODEL = ""
DEFAULT_API_KEY = ""
DEFAULT_TIMEOUT_SECONDS = 60.0


def default_llm_settings_path() -> Path:
    raw_path = os.getenv("ACTSOFT_LLM_SETTINGS_PATH")
    if raw_path:
        return Path(raw_path).expanduser()
    return Path(__file__).resolve().parents[1] / "runs" / "llm_settings.json"


def load_persisted_llm_settings(settings_path: Path | None = None) -> dict[str, Any]:
    path = settings_path or default_llm_settings_path()
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def save_persisted_llm_settings(
    settings: dict[str, Any],
    settings_path: Path | None = None,
    *,
    allow_empty: bool = False,
) -> dict[str, Any]:
    base_url = str(settings.get("base_url") or "").strip()
    model = str(settings.get("model") or "").strip()
    if not base_url and not allow_empty:
        raise ValueError("base_url is required")
    if not model and not allow_empty:
        raise ValueError("model is required")
    normalized = {
        "base_url": normalize_base_url(base_url) if base_url else "",
        "model": model,
        "api_key": str(settings.get("api_key") or "").strip(),
    }
    if settings.get("timeout_seconds") not in {None, ""}:
        normalized["timeout_seconds"] = float(settings["timeout_seconds"])
    path = settings_path or default_llm_settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(normalized, indent=2, ensure_ascii=False), encoding="utf-8")
    return normalized


@dataclass(frozen=True)
class LLMConfig:
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    api_key: str = DEFAULT_API_KEY
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS

    @classmethod
    def from_env(cls, *, settings_path: Path | None = None) -> "LLMConfig":
        config = cls(
            base_url=normalize_base_url(
                os.getenv("ACTSOFT_LLM_BASE_URL", DEFAULT_BASE_URL)
            ),
            model=os.getenv("ACTSOFT_LLM_MODEL", DEFAULT_MODEL),
            api_key=os.getenv("ACTSOFT_LLM_API_KEY", DEFAULT_API_KEY),
            timeout_seconds=float(os.getenv("ACTSOFT_LLM_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS))),
        )
        persisted = load_persisted_llm_settings(settings_path)
        if not persisted:
            return config
        base_url = config.base_url
        if "base_url" in persisted:
            persisted_base_url = str(persisted.get("base_url") or "").strip()
            base_url = normalize_base_url(persisted_base_url) if persisted_base_url else ""
        model = config.model
        if "model" in persisted:
            model = str(persisted.get("model") or "").strip()
        return cls(
            base_url=base_url,
            model=model,
            api_key=str(persisted["api_key"]) if "api_key" in persisted else config.api_key,
            timeout_seconds=float(persisted.get("timeout_seconds") or config.timeout_seconds),
        )


def normalize_base_url(value: str) -> str:
    stripped = value.strip().rstrip("/")
    if not stripped:
        return ""
    if not stripped.startswith(("http://", "https://")):
        stripped = f"http://{stripped}"
    return stripped


class LLMClient:
    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig.from_env()
        self.client: OpenAI | None = None
        if self.config.base_url and self.config.model:
            self.client = OpenAI(
                base_url=self.config.base_url,
                api_key=self.config.api_key,
                timeout=self.config.timeout_seconds,
            )

    def chat(self, system: str, user: str, temperature: float = 0.1) -> str:
        if self.client is None or not self.config.model:
            raise ValueError("LLM base_url or model is not configured.")
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
        )
        content = response.choices[0].message.content
        return content or ""

    def tool_call(
        self,
        system: str,
        user: str,
        tools: list[dict[str, Any]],
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        if self.client is None or not self.config.model:
            raise ValueError("LLM base_url or model is not configured.")
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            tools=tools,
            tool_choice="auto",
            temperature=temperature,
        )
        message = response.choices[0].message
        tool_calls = getattr(message, "tool_calls", None) or []
        if tool_calls:
            first = tool_calls[0]
            function = getattr(first, "function", None)
            return {
                "tool_call": {
                    "id": getattr(first, "id", None),
                    "name": getattr(function, "name", None),
                    "arguments": getattr(function, "arguments", "{}"),
                },
                "content": message.content or "",
            }
        return {"content": message.content or ""}
