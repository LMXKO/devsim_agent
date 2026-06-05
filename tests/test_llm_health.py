from __future__ import annotations

import unittest

from tcad_agent.llm import LLMConfig
from tcad_agent.llm_health import LLMHealthStatus, check_llm_health, configured_llm_status


class FakeHealthClient:
    config = LLMConfig(base_url="http://llm.local/v1", model="fake-health")

    def __init__(self, response: str = '{"ok":true}', failure: Exception | None = None) -> None:
        self.response = response
        self.failure = failure

    def chat(self, system: str, user: str, temperature: float = 0.1) -> str:
        if self.failure:
            raise self.failure
        return self.response


class LLMHealthTest(unittest.TestCase):
    def test_configured_status_uses_config_without_live_call(self) -> None:
        result = configured_llm_status(LLMConfig(base_url="http://llm.local/v1", model="local-chat-model"))

        self.assertEqual(result.status, LLMHealthStatus.CONFIGURED)
        self.assertEqual(result.base_url, "http://llm.local/v1")
        self.assertEqual(result.model, "local-chat-model")
        self.assertFalse(result.checked_live)

    def test_configured_status_reports_unconfigured_when_url_or_model_is_blank(self) -> None:
        result = configured_llm_status(LLMConfig(base_url="", model=""))

        self.assertEqual(result.status, LLMHealthStatus.UNCONFIGURED)
        self.assertEqual(result.base_url, "")
        self.assertEqual(result.model, "")
        self.assertFalse(result.checked_live)

    def test_live_health_passes_with_nonempty_response(self) -> None:
        result = check_llm_health(FakeHealthClient())

        self.assertEqual(result.status, LLMHealthStatus.PASSED)
        self.assertTrue(result.checked_live)
        self.assertEqual(result.model, "fake-health")

    def test_live_health_reports_failure(self) -> None:
        result = check_llm_health(FakeHealthClient(failure=RuntimeError("connection refused")))

        self.assertEqual(result.status, LLMHealthStatus.FAILED)
        self.assertIn("connection refused", result.failure_reason or "")


if __name__ == "__main__":
    unittest.main()
