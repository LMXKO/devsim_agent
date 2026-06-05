from __future__ import annotations

import time
from enum import Enum
from typing import Protocol

from pydantic import BaseModel

from tcad_agent.llm import LLMClient, LLMConfig


class LLMHealthStatus(str, Enum):
    UNCONFIGURED = "unconfigured"
    CONFIGURED = "configured"
    PASSED = "passed"
    FAILED = "failed"


class ChatClient(Protocol):
    config: LLMConfig

    def chat(self, system: str, user: str, temperature: float = 0.1) -> str:
        ...


class LLMHealthResult(BaseModel):
    status: LLMHealthStatus
    base_url: str
    model: str
    api_format: str = "openai_chat_completions"
    checked_live: bool = False
    latency_seconds: float | None = None
    response_excerpt: str | None = None
    failure_reason: str | None = None


def configured_llm_status(config: LLMConfig | None = None) -> LLMHealthResult:
    actual = config or LLMConfig.from_env()
    return LLMHealthResult(
        status=LLMHealthStatus.CONFIGURED if actual.base_url and actual.model else LLMHealthStatus.UNCONFIGURED,
        base_url=actual.base_url,
        model=actual.model,
    )


def check_llm_health(client: ChatClient | None = None) -> LLMHealthResult:
    actual_client = client or LLMClient()
    if not actual_client.config.base_url or not actual_client.config.model:
        return LLMHealthResult(
            status=LLMHealthStatus.UNCONFIGURED,
            base_url=actual_client.config.base_url,
            model=actual_client.config.model,
            checked_live=True,
            failure_reason="LLM base_url or model is not configured.",
        )
    started = time.monotonic()
    try:
        response = actual_client.chat(
            "You are a health-check endpoint. Reply with JSON only.",
            '{"task":"health_check","reply":"ok"}',
            temperature=0.0,
        )
        latency = time.monotonic() - started
        if not response.strip():
            return LLMHealthResult(
                status=LLMHealthStatus.FAILED,
                base_url=actual_client.config.base_url,
                model=actual_client.config.model,
                checked_live=True,
                latency_seconds=latency,
                failure_reason="empty response",
            )
        return LLMHealthResult(
            status=LLMHealthStatus.PASSED,
            base_url=actual_client.config.base_url,
            model=actual_client.config.model,
            checked_live=True,
            latency_seconds=latency,
            response_excerpt=response.strip()[:240],
        )
    except Exception as exc:
        return LLMHealthResult(
            status=LLMHealthStatus.FAILED,
            base_url=actual_client.config.base_url,
            model=actual_client.config.model,
            checked_live=True,
            latency_seconds=time.monotonic() - started,
            failure_reason=str(exc),
        )
