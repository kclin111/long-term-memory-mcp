"""Operator / ExtractWorker LLM provider adapter.

The deterministic extractor in :mod:`ltm_memory.extractors` produces a
GSW-style workspace proposal (entities, roles, states, actions, anchors,
open questions). M2.5 adds a pluggable Operator so the same payload can be
produced by an LLM provider when one is configured. The deterministic
extractor remains the default and the only fully-online path; LLM
providers are scaffolded but degrade to deterministic output on any
failure.

Wiring is environment-driven:

* ``LTM_LLM_PROVIDER`` -- ``none`` (default), ``openai``, ``anthropic``,
  ``ollama``, or ``lmstudio``. ``none`` runs in deterministic-only mode.
* ``LTM_LLM_MODEL`` -- model identifier passed through to the provider.

The Operator never raises on a malformed LLM response. It logs the
failure inside the proposal payload and falls back to the deterministic
extractor so the IntegrateWorker can still make progress, matching the
PRD §12 "degraded mode" guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import urllib.error
import urllib.request
from typing import Any, Callable, Protocol

from .config import load_dotenv
from .extractors import extract_event_proposal


OPERATOR_PROMPT_VERSION = "ltm-operator-v1"


OPERATOR_SYSTEM_PROMPT = """You are the Operator for an event-centric long-term memory system.

Given a single observation (one chat turn, document chunk, or tool result) and
the recent SessionBuffer chain summary as background context, you must extract
a GSW-style workspace proposal as strict JSON. Do not invent facts that are
not supported by the observation or the chain summary.

Return ONLY a JSON object that matches this schema:

{
  "event_type": "decision | preference | correction | task_progress | project_state | document_claim | fact | interaction | error_workaround | commitment | state_change | episodic_event",
  "summary": "<= 240 chars, factual",
  "importance": float in [0, 1],
  "novelty": float in [0, 1],
  "confidence": float in [0, 1],
  "entities": [
    {"canonical_name": str, "entity_type": "person | place | tool | concept | project | document | task | organization | time", "role_in_event": str, "confidence": float}
  ],
  "roles": [
    {"entity": str, "role_type": str, "confidence": float}
  ],
  "states": [
    {"entity": str, "state_type": "preference | decision | status | trait | goal | constraint | fact | emotion | capability", "value": str, "role_type": str | null, "confidence": float}
  ],
  "actions": [
    {"actor": str, "action_type": str, "action_text": str, "object": str | null, "valence": "positive | neutral | negative", "confidence": float}
  ],
  "anchors": [
    {"anchor_type": "time | space | spatiotemporal", "label": str, "normalized_value": str, "granularity": str, "confidence": float, "status": "candidate | confirmed"}
  ],
  "open_questions": [
    {"subject": str, "question_type": "when | where | who | what | why | how", "question_text": str, "related_entities": [str], "priority": 1 | 2 | 3, "confidence": float}
  ]
}

Rules:
- Use entity canonical names exactly as referenced in the observation. Use "user" for the speaker of a chat observation when no other actor is named.
- Do NOT emit anchors that are not literally supported by the observation or chain summary.
- If you would only emit an anchor as a guess, emit an open question instead.
- Keep confidence calibrated. Use <= 0.6 for guesses; >= 0.8 only for explicit text.
- If you cannot produce JSON that matches the schema, return: {"_unparseable": true}
"""


def build_operator_user_prompt(observation: dict, *, context: dict | None = None) -> str:
    """Build the per-observation user message for the Operator prompt."""

    context = context or {}
    chain_summary = context.get("chain_summary") or ""
    recent_pages = context.get("recent_pages") or []
    parts = [
        f"observation_id: {observation.get('id')}",
        f"source_type: {observation.get('source_type')}",
        f"observed_at: {observation.get('observed_at')}",
    ]
    if observation.get("session_id"):
        parts.append(f"session_id: {observation['session_id']}")
    if chain_summary:
        parts.append(f"chain_summary: {chain_summary}")
    if recent_pages:
        parts.append("recent_pages:")
        for page in recent_pages[-5:]:
            role = page.get("role", "unknown")
            content = (page.get("content") or "").strip().replace("\n", " ")
            if len(content) > 240:
                content = content[:239] + "…"
            parts.append(f"  - {role}: {content}")
    parts.append("observation_content:")
    parts.append(observation.get("content", ""))
    return "\n".join(parts)


class LLMProvider(Protocol):
    """Minimal interface every LLM backend must implement.

    Implementations should return the parsed JSON object produced by the
    model. They MUST raise ``LLMProviderError`` (or a subclass) on
    transport/parsing errors so the Operator can fall back gracefully.
    """

    name: str

    def generate_json(self, *, system_prompt: str, user_prompt: str) -> dict:
        ...


class LLMProviderError(RuntimeError):
    """Raised by an LLMProvider when it cannot return a parseable JSON dict."""


@dataclass(frozen=True)
class NoLLMProvider:
    """Default provider: explicitly disables LLM extraction."""

    name: str = "none"

    def generate_json(self, *, system_prompt: str, user_prompt: str) -> dict:  # noqa: D401 - protocol stub
        raise LLMProviderError("LLM provider is disabled (LTM_LLM_PROVIDER=none)")


@dataclass(frozen=True)
class CallableLLMProvider:
    """Adapter that wraps a Python callable for testing or local stubs."""

    name: str
    handler: Callable[[str, str], dict]

    def generate_json(self, *, system_prompt: str, user_prompt: str) -> dict:
        try:
            payload = self.handler(system_prompt, user_prompt)
        except Exception as exc:  # noqa: BLE001 - propagate as provider error
            raise LLMProviderError(str(exc)) from exc
        if not isinstance(payload, dict):
            raise LLMProviderError(f"LLM handler must return dict, got {type(payload).__name__}")
        return payload


def _extract_json_object(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise LLMProviderError("model response did not contain a JSON object")
        try:
            payload = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMProviderError(f"model response was not parseable JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise LLMProviderError(f"model response must be a JSON object, got {type(payload).__name__}")
    return payload


@dataclass(frozen=True)
class OpenRouterLLMProvider:
    """OpenRouter chat-completions provider."""

    api_key: str
    model: str
    base_url: str = "https://openrouter.ai/api/v1"
    timeout_seconds: float = 60.0
    site_url: str | None = None
    app_name: str | None = None
    name: str = "openrouter"

    def generate_json(self, *, system_prompt: str, user_prompt: str) -> dict:
        endpoint = self.base_url.rstrip("/") + "/chat/completions"
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise LLMProviderError(f"OpenRouter HTTP {exc.code}: {detail[:500]}") from exc
        except urllib.error.URLError as exc:
            raise LLMProviderError(f"OpenRouter request failed: {exc.reason}") from exc
        except TimeoutError as exc:
            raise LLMProviderError("OpenRouter request timed out") from exc

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMProviderError(f"OpenRouter response was not JSON: {exc}") from exc
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMProviderError(f"OpenRouter response missing choices[0].message.content: {payload}") from exc
        if not isinstance(content, str):
            raise LLMProviderError(f"OpenRouter content must be str, got {type(content).__name__}")
        return _extract_json_object(content)

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.app_name:
            headers["X-Title"] = self.app_name
        return headers


def build_provider_from_env(env: dict[str, str] | None = None) -> LLMProvider:
    """Resolve an :class:`LLMProvider` from environment variables.

    The default is :class:`NoLLMProvider`. Real HTTP-backed providers
    (``openai``/``anthropic``/``ollama``/``lmstudio``) are intentionally
    skeleton-only in M2.5: they raise a clear error pointing at the
    expected wiring. This keeps the Operator behavior fully deterministic
    in CI while leaving a single, well-named seam for the M3 integration
    work.
    """

    load_dotenv()
    env = env if env is not None else os.environ
    provider = (env.get("LTM_LLM_PROVIDER") or "none").lower()
    if provider in {"", "none", "off", "disabled"}:
        return NoLLMProvider()
    model = env.get("LTM_LLM_MODEL")
    if provider == "openrouter":
        api_key = env.get("OPENROUTER_API_KEY")
        if not api_key:
            raise LLMProviderError("OPENROUTER_API_KEY is required when LTM_LLM_PROVIDER=openrouter")
        if not model:
            raise LLMProviderError("LTM_LLM_MODEL is required when LTM_LLM_PROVIDER=openrouter")
        return OpenRouterLLMProvider(
            api_key=api_key,
            model=model,
            base_url=env.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            timeout_seconds=float(env.get("LTM_LLM_TIMEOUT_SECONDS", "60")),
            site_url=env.get("OPENROUTER_SITE_URL"),
            app_name=env.get("OPENROUTER_APP_NAME", "long-term-memory-mcp"),
        )
    raise LLMProviderError(
        f"LLM provider '{provider}' (model={model!r}) is declared but its HTTP "
        "adapter is not wired in this build. Set LTM_LLM_PROVIDER=openrouter, LTM_LLM_PROVIDER=none, or "
        "inject an LLMProvider instance via Operator(provider=...)."
    )


@dataclass
class OperatorResult:
    """Wraps a workspace proposal and the provider that produced it."""

    proposal: dict
    provider_name: str
    fallback_used: bool
    fallback_reason: str | None = None


class Operator:
    """Pluggable workspace-proposal builder.

    The Operator first tries the configured :class:`LLMProvider`. On any
    failure -- provider error, malformed JSON, missing required fields --
    it falls back to the deterministic extractor and records the reason
    on the resulting proposal so callers can see why the LLM path did
    not run.
    """

    def __init__(self, provider: LLMProvider | None = None) -> None:
        self.provider = provider or build_provider_from_env()

    def extract(self, observation: dict, *, context: dict | None = None) -> OperatorResult:
        deterministic = extract_event_proposal(observation)
        provider_name = getattr(self.provider, "name", type(self.provider).__name__)

        if isinstance(self.provider, NoLLMProvider):
            return OperatorResult(
                proposal=self._tag(deterministic, provider_name, fallback=False),
                provider_name=provider_name,
                fallback_used=False,
            )

        user_prompt = build_operator_user_prompt(observation, context=context)
        try:
            llm_payload = self.provider.generate_json(
                system_prompt=OPERATOR_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
        except LLMProviderError as exc:
            return OperatorResult(
                proposal=self._tag(deterministic, provider_name, fallback=True, reason=str(exc)),
                provider_name=provider_name,
                fallback_used=True,
                fallback_reason=str(exc),
            )

        if llm_payload.get("_unparseable"):
            return OperatorResult(
                proposal=self._tag(deterministic, provider_name, fallback=True, reason="model returned _unparseable=true"),
                provider_name=provider_name,
                fallback_used=True,
                fallback_reason="model returned _unparseable=true",
            )

        try:
            merged = self._merge_with_deterministic(llm_payload, deterministic)
        except (KeyError, TypeError, ValueError) as exc:
            return OperatorResult(
                proposal=self._tag(deterministic, provider_name, fallback=True, reason=f"schema mismatch: {exc}"),
                provider_name=provider_name,
                fallback_used=True,
                fallback_reason=f"schema mismatch: {exc}",
            )

        return OperatorResult(
            proposal=self._tag(merged, provider_name, fallback=False),
            provider_name=provider_name,
            fallback_used=False,
        )

    @staticmethod
    def _tag(proposal: dict, provider_name: str, *, fallback: bool, reason: str | None = None) -> dict:
        structured = dict(proposal.get("structured") or {})
        structured.setdefault("extractor", "deterministic_m2")
        structured["operator_provider"] = provider_name
        structured["operator_prompt_version"] = OPERATOR_PROMPT_VERSION
        structured["operator_fallback_used"] = fallback
        if reason:
            structured["operator_fallback_reason"] = reason
        proposal = dict(proposal)
        proposal["structured"] = structured
        return proposal

    @staticmethod
    def _merge_with_deterministic(llm_payload: dict, deterministic: dict) -> dict:
        """Validate the LLM payload and fill any missing fields from deterministic output.

        The deterministic extractor is the source of truth for evidence excerpts
        and source ids. The LLM is allowed to override the higher-signal fields:
        event_type, summary, entities, roles, states, actions, anchors,
        open_questions, importance, novelty, confidence.
        """

        required_lists = ["entities", "roles", "states", "actions", "anchors", "open_questions"]
        for key in required_lists:
            value = llm_payload.get(key, deterministic.get(key, []))
            if not isinstance(value, list):
                raise TypeError(f"field {key!r} must be a list")
            llm_payload[key] = value

        merged = dict(deterministic)
        for key in (
            "event_type",
            "summary",
            "importance",
            "novelty",
            "confidence",
            "entities",
            "roles",
            "states",
            "actions",
            "anchors",
            "open_questions",
        ):
            if key in llm_payload and llm_payload[key] is not None:
                merged[key] = llm_payload[key]
        merged["evidence_excerpt"] = deterministic.get("evidence_excerpt", "")
        return merged


__all__ = [
    "Operator",
    "OperatorResult",
    "OPERATOR_PROMPT_VERSION",
    "OPERATOR_SYSTEM_PROMPT",
    "LLMProvider",
    "LLMProviderError",
    "NoLLMProvider",
    "CallableLLMProvider",
    "OpenRouterLLMProvider",
    "build_operator_user_prompt",
    "build_provider_from_env",
]
