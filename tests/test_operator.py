from __future__ import annotations

import unittest
from unittest.mock import patch

from ltm_memory.operator import (
    CallableLLMProvider,
    LLMProviderError,
    NoLLMProvider,
    Operator,
    OpenRouterLLMProvider,
    OPERATOR_PROMPT_VERSION,
    build_operator_user_prompt,
    build_provider_from_env,
)


SAMPLE_OBSERVATION = {
    "id": "obs_test_1",
    "source_type": "chat",
    "session_id": "s-bench",
    "observed_at": "2026-09-22T10:00:00Z",
    "content": "User decided to implement MCP memory with Python, SQLite FTS5, and LanceDB.",
}


class OperatorTests(unittest.TestCase):
    def test_default_provider_runs_deterministic_without_fallback_flag(self) -> None:
        result = Operator(provider=NoLLMProvider()).extract(SAMPLE_OBSERVATION)
        self.assertFalse(result.fallback_used)
        self.assertEqual(result.provider_name, "none")
        structured = result.proposal["structured"]
        self.assertEqual(structured["operator_provider"], "none")
        self.assertEqual(structured["operator_prompt_version"], OPERATOR_PROMPT_VERSION)
        self.assertGreater(len(result.proposal["entities"]), 0)

    def test_callable_provider_can_override_event_type_and_summary(self) -> None:
        def handler(system_prompt: str, user_prompt: str) -> dict:
            self.assertIn("Operator", system_prompt)
            self.assertIn("observation_content:", user_prompt)
            return {
                "event_type": "decision",
                "summary": "User picks Python + SQLite FTS5 + LanceDB stack.",
                "importance": 0.81,
                "novelty": 0.74,
                "confidence": 0.77,
                "entities": [
                    {"canonical_name": "user", "entity_type": "person", "role_in_event": "speaker", "confidence": 0.9},
                    {"canonical_name": "Python", "entity_type": "tool", "role_in_event": "mentioned", "confidence": 0.85},
                ],
                "roles": [{"entity": "user", "role_type": "decision_maker", "confidence": 0.85}],
                "states": [{"entity": "user", "state_type": "decision", "value": "use Python", "role_type": "decision_maker", "confidence": 0.8}],
                "actions": [{"actor": "user", "action_type": "decide", "action_text": "use Python", "object": "Python", "valence": "positive", "confidence": 0.78}],
                "anchors": [],
                "open_questions": [],
            }

        operator = Operator(provider=CallableLLMProvider(name="stub", handler=handler))
        result = operator.extract(SAMPLE_OBSERVATION)
        self.assertFalse(result.fallback_used)
        self.assertEqual(result.proposal["event_type"], "decision")
        self.assertIn("Python + SQLite FTS5", result.proposal["summary"])
        self.assertEqual(result.proposal["confidence"], 0.77)
        self.assertEqual(result.provider_name, "stub")

    def test_unparseable_marker_triggers_deterministic_fallback(self) -> None:
        operator = Operator(provider=CallableLLMProvider(name="stub", handler=lambda *_: {"_unparseable": True}))
        result = operator.extract(SAMPLE_OBSERVATION)
        self.assertTrue(result.fallback_used)
        self.assertIn("_unparseable", (result.fallback_reason or ""))
        self.assertGreater(len(result.proposal["entities"]), 0)

    def test_provider_error_falls_back_with_reason(self) -> None:
        def boom(*_: str) -> dict:
            raise LLMProviderError("network down")

        operator = Operator(provider=CallableLLMProvider(name="stub", handler=boom))
        result = operator.extract(SAMPLE_OBSERVATION)
        self.assertTrue(result.fallback_used)
        self.assertEqual(result.fallback_reason, "network down")

    def test_schema_mismatch_falls_back(self) -> None:
        bad = lambda *_: {"event_type": "decision", "entities": "not-a-list"}
        operator = Operator(provider=CallableLLMProvider(name="stub", handler=bad))
        result = operator.extract(SAMPLE_OBSERVATION)
        self.assertTrue(result.fallback_used)
        self.assertIn("schema mismatch", result.fallback_reason or "")

    def test_user_prompt_includes_chain_summary(self) -> None:
        prompt = build_operator_user_prompt(
            SAMPLE_OBSERVATION,
            context={
                "chain_summary": "user is mid-discussion about local-first MCP",
                "recent_pages": [
                    {"role": "user", "content": "Let's pick the storage backend."},
                    {"role": "assistant", "content": "SQLite + LanceDB is the local-first answer."},
                ],
            },
        )
        self.assertIn("chain_summary: user is mid-discussion", prompt)
        self.assertIn("Let's pick the storage backend.", prompt)
        self.assertIn("observation_content:", prompt)

    def test_build_provider_from_env_defaults_to_none(self) -> None:
        provider = build_provider_from_env({})
        self.assertIsInstance(provider, NoLLMProvider)

    def test_build_provider_from_env_openrouter(self) -> None:
        provider = build_provider_from_env(
            {
                "LTM_LLM_PROVIDER": "openrouter",
                "LTM_LLM_MODEL": "openai/gpt-4o-mini",
                "OPENROUTER_API_KEY": "test-key",
                "OPENROUTER_BASE_URL": "https://openrouter.ai/api/v1",
            }
        )
        self.assertIsInstance(provider, OpenRouterLLMProvider)
        self.assertEqual(provider.model, "openai/gpt-4o-mini")

    def test_openrouter_provider_parses_json_response(self) -> None:
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                return None

            def read(self) -> bytes:
                return (
                    b'{"choices":[{"message":{"content":"{\\"event_type\\":\\"decision\\",'
                    b'\\"summary\\":\\"LLM summary\\",\\"importance\\":0.8,\\"novelty\\":0.7,'
                    b'\\"confidence\\":0.75,\\"entities\\":[],\\"roles\\":[],\\"states\\":[],'
                    b'\\"actions\\":[],\\"anchors\\":[],\\"open_questions\\":[]}"}}]}'
                )

        provider = OpenRouterLLMProvider(api_key="test-key", model="openai/gpt-4o-mini")
        with patch("urllib.request.urlopen", return_value=FakeResponse()) as mocked:
            payload = provider.generate_json(system_prompt="system", user_prompt="user")

        self.assertEqual(payload["event_type"], "decision")
        request = mocked.call_args.args[0]
        self.assertEqual(request.headers["Authorization"], "Bearer test-key")

    def test_build_provider_from_env_missing_openrouter_key_raises(self) -> None:
        with self.assertRaises(LLMProviderError):
            build_provider_from_env({"LTM_LLM_PROVIDER": "openrouter", "LTM_LLM_MODEL": "openai/gpt-4o-mini"})

    def test_build_provider_from_env_unwired_provider_raises(self) -> None:
        with self.assertRaises(LLMProviderError):
            build_provider_from_env({"LTM_LLM_PROVIDER": "openai", "LTM_LLM_MODEL": "gpt-4o-mini"})


if __name__ == "__main__":
    unittest.main()
