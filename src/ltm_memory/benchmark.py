"""Synthetic memory-quality benchmark harness (PRD §14.3).

This module loads a benchmark scenario file, replays the observations
into a fresh isolated SQLite + LanceDB-shaped store, runs each
scenario's queries, and scores the results. The harness is deterministic
and self-contained: every scenario starts from an empty store under a
temporary directory, so results are reproducible across machines.

Metrics reported per scenario and aggregated overall:

* ``event_recall_f1`` -- precision/recall/F1 over per-query event-level
  expectations (substring or event_type matches).
* ``evidence_coverage`` -- fraction of returned events that have at
  least one ``evidence`` row in SQLite.
* ``hallucination_rate`` -- proxy: fraction of returned events whose
  summary cannot be traced to any backing observation in the seeded
  scenario. Should be 0.0 for the baseline.
* ``expectations_passed`` / ``expectations_failed`` -- raw count of
  per-query expectation checks.

The harness is intentionally deterministic-only (Operator runs in
``LTM_LLM_PROVIDER=none`` mode) so the benchmark reflects baseline
behavior, not the variance of any specific LLM provider.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any
from uuid import uuid4
import json
import shutil

from .config import Settings
from .store import MemoryStore
from .time import utc_now
from .worker import JobWorker


DEFAULT_SCENARIO_FILE = resources.files("ltm_memory").joinpath(
    "benchmarks",
    "synthetic",
    "m2_baseline.json",
)
EPISODIC_HARD_SCENARIO_FILE = resources.files("ltm_memory").joinpath(
    "benchmarks",
    "synthetic",
    "episodic_hard.json",
)

BENCHMARK_TMP_ROOT = Path.cwd() / ".codex-tmp" / "benchmark-runs"


@dataclass
class QueryResult:
    name: str
    query_type: str
    passed: int = 0
    failed: int = 0
    failures: list[str] = field(default_factory=list)
    raw: dict | None = None

    def add_check(self, ok: bool, message: str) -> None:
        if ok:
            self.passed += 1
        else:
            self.failed += 1
            self.failures.append(message)


@dataclass
class ScenarioResult:
    name: str
    description: str
    queries: list[QueryResult] = field(default_factory=list)
    event_true_positives: int = 0
    event_false_positives: int = 0
    event_false_negatives: int = 0
    returned_event_count: int = 0
    events_with_evidence: int = 0
    hallucinated_events: int = 0
    max_segment_heat: float = 0.0
    persistent_memory_count: int = 0
    open_conflicts: int = 0
    error: str | None = None

    @property
    def passed(self) -> int:
        return sum(q.passed for q in self.queries)

    @property
    def failed(self) -> int:
        return sum(q.failed for q in self.queries)

    @property
    def event_recall_f1(self) -> float:
        tp = self.event_true_positives
        fp = self.event_false_positives
        fn = self.event_false_negatives
        if tp == 0 and (fp == 0 or fn == 0):
            return 1.0 if tp + fp + fn == 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision + recall == 0:
            return 0.0
        return round(2 * precision * recall / (precision + recall), 4)

    @property
    def evidence_coverage(self) -> float:
        if self.returned_event_count == 0:
            return 1.0
        return round(self.events_with_evidence / self.returned_event_count, 4)

    @property
    def hallucination_rate(self) -> float:
        if self.returned_event_count == 0:
            return 0.0
        return round(self.hallucinated_events / self.returned_event_count, 4)


def run_benchmark(*, scenario_file: str | Path | None = None) -> dict[str, Any]:
    """Run all scenarios in ``scenario_file`` and return a JSON-serializable report."""

    path = Path(scenario_file) if scenario_file else DEFAULT_SCENARIO_FILE
    if not path.exists():
        raise FileNotFoundError(f"benchmark scenario file not found: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))

    scenarios = document.get("scenarios", [])
    results: list[ScenarioResult] = [
        _run_scenario(scenario) for scenario in scenarios
    ]

    return _build_report(document, path, results)


def run_episodic_benchmark(*, scenario_file: str | Path | None = None) -> dict[str, Any]:
    """Run the stricter GSW-style episodic synthetic benchmark."""

    return run_benchmark(scenario_file=scenario_file or EPISODIC_HARD_SCENARIO_FILE)


def _run_scenario(scenario: dict) -> ScenarioResult:
    result = ScenarioResult(
        name=scenario.get("name", "<unnamed>"),
        description=scenario.get("description", ""),
    )
    BENCHMARK_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    tmp_root = BENCHMARK_TMP_ROOT / f"ltm-bench-{uuid4().hex}"
    tmp_root.mkdir(parents=True, exist_ok=False)
    try:
        settings = Settings(
            home=tmp_root,
            sqlite_path=tmp_root / "memory.sqlite",
            lancedb_path=tmp_root / "lancedb",
            sqlite_journal_mode="MEMORY",
        )
        store = MemoryStore.open(settings)
        try:
            seeded_summaries = _seed_observations(store, scenario)
            JobWorker(store).process(max_jobs=200)
            result.returned_event_count = store.conn.execute(
                "SELECT COUNT(*) AS n FROM events"
            ).fetchone()["n"]
            result.events_with_evidence = store.conn.execute(
                """
                SELECT COUNT(*) AS n FROM events e
                WHERE EXISTS (SELECT 1 FROM evidence ev WHERE ev.event_id = e.id)
                """
            ).fetchone()["n"]
            result.hallucinated_events = _count_hallucinated_events(store, seeded_summaries)
            row = store.conn.execute(
                "SELECT COALESCE(MAX(heat), 0) AS h FROM memory_segments"
            ).fetchone()
            result.max_segment_heat = float(row["h"] or 0.0)
            result.persistent_memory_count = store.conn.execute(
                "SELECT COUNT(*) AS n FROM persistent_memories"
            ).fetchone()["n"]
            result.open_conflicts = store.conn.execute(
                "SELECT COUNT(*) AS n FROM memory_conflicts WHERE status = 'open'"
            ).fetchone()["n"]
            for query in scenario.get("queries", []):
                _run_query(store, query, result)
        finally:
            store.close()
    except Exception as exc:  # noqa: BLE001 - benchmark must report, not crash CI
        result.error = f"{type(exc).__name__}: {exc}"
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    return result


def _seed_observations(store: MemoryStore, scenario: dict) -> list[str]:
    """Ingest each observation. Returns the list of raw observation contents."""

    contents: list[str] = []
    for observation in scenario.get("observations", []):
        store.ingest_observation(
            source_type=observation.get("source_type", "chat"),
            content=observation["content"],
            session_id=observation.get("session_id") or scenario.get("session_id"),
            client_dedup_key=observation.get("client_dedup_key"),
            metadata=observation.get("metadata"),
        )
        contents.append(observation["content"])
    return contents


def _count_hallucinated_events(store: MemoryStore, seeded_contents: list[str]) -> int:
    """Heuristic: an event is "hallucinated" if its summary shares no
    significant lexical overlap with any seeded observation. Because the
    deterministic extractor only summarizes from the source content, this
    should always be 0 for the baseline. Any non-zero value flags an
    integrity regression."""

    if not seeded_contents:
        return 0
    normalized_sources = [content.lower() for content in seeded_contents]
    rows = store.conn.execute("SELECT summary FROM events").fetchall()
    hallucinated = 0
    for row in rows:
        summary = (row["summary"] or "").lower()
        tokens = {token for token in summary.split() if len(token) >= 4}
        if not tokens:
            continue
        if not any(any(token in source for token in tokens) for source in normalized_sources):
            hallucinated += 1
    return hallucinated


def _run_query(store: MemoryStore, query: dict, result: ScenarioResult) -> None:
    qtype = query.get("type", "recall")
    qresult = QueryResult(
        name=query.get("name") or qtype,
        query_type=qtype,
    )
    expectations = query.get("expectations") or {}
    if qtype == "recall":
        qresult.raw = store.recall(
            query["query"],
            session_id=query.get("session_id"),
            limits=query.get("limits"),
            scope=query.get("scope", "auto"),
            include_session_buffer=query.get("include_session_buffer", True),
        )
        _check_recall(qresult, expectations, result)
    elif qtype == "entity_timeline":
        qresult.raw = store.get_entity_timeline(
            query["entity_query"],
            since=query.get("since"),
            until=query.get("until"),
            event_types=query.get("event_types"),
            state_types=query.get("state_types"),
            limit=query.get("limit", 50),
        )
        _check_entity_timeline(qresult, expectations, result)
    elif qtype == "forget_event":
        qresult.raw = _run_forget_event_query(store, query)
        _check_forget_event(qresult, expectations, result)
    else:
        qresult.add_check(False, f"unknown query type: {qtype}")
    result.queries.append(qresult)


def _run_forget_event_query(store: MemoryStore, query: dict) -> dict:
    substring = query["event_summary_substring"]
    target = store.conn.execute(
        "SELECT id FROM events WHERE summary LIKE ? ORDER BY observed_at DESC LIMIT 1",
        (f"%{substring}%",),
    ).fetchone()
    if not target:
        return {"error": f"event not found for substring: {substring}"}
    target_id = target["id"]

    entity_rows = store.conn.execute(
        """
        SELECT entity_id
        FROM event_entities
        WHERE event_id = ?
        ORDER BY CASE role_in_event WHEN 'speaker' THEN 1 ELSE 0 END, confidence DESC
        """,
        (target_id,),
    ).fetchall()
    entity_ids = [row["entity_id"] for row in entity_rows]

    support_substrings = query.get("supporting_event_summary_substrings") or [substring]
    support_ids: list[str] = []
    for support in support_substrings:
        row = store.conn.execute(
            "SELECT id FROM events WHERE summary LIKE ? ORDER BY observed_at DESC LIMIT 1",
            (f"%{support}%",),
        ).fetchone()
        if row and row["id"] not in support_ids:
            support_ids.append(row["id"])
    if target_id not in support_ids:
        support_ids.append(target_id)

    now = utc_now()
    with store.conn:
        for entity_id in entity_ids[:2]:
            store.narratives.get_or_generate(
                entity_id=entity_id,
                scope="global",
                now=now,
                force_refresh=True,
            )
        store.conn.execute(
            """
            INSERT OR REPLACE INTO persistent_memories(
              id, memory_type, content, entity_ids_json, supporting_event_ids_json,
              confidence, stability, status, reinforcement_count,
              created_at, updated_at
            ) VALUES ('pm_benchmark_forget', 'design_decision', 'Benchmark forget cascade memory',
              ?, ?, 0.8, 0.8, 'active', ?, ?, ?)
            """,
            (
                json.dumps(entity_ids[:2]),
                json.dumps(support_ids),
                len(support_ids),
                now,
                now,
            ),
        )

    before_pm = store.conn.execute(
        "SELECT reinforcement_count FROM persistent_memories WHERE id = 'pm_benchmark_forget'"
    ).fetchone()
    report = store.forget(
        target_type="event",
        target_id=target_id,
        mode=query.get("mode", "soft"),
        cascade=query.get("cascade", True),
    )
    after_pm = store.conn.execute(
        """
        SELECT status, reinforcement_count, supporting_event_ids_json
        FROM persistent_memories
        WHERE id = 'pm_benchmark_forget'
        """
    ).fetchone()
    stale_narratives = store.conn.execute(
        "SELECT COUNT(*) AS n FROM entity_narratives WHERE status = 'stale'"
    ).fetchone()["n"]
    visible_after_forget = store.search_events(
        query=substring,
        include_forgotten=False,
        limit=5,
    )
    return {
        "target_event_id": target_id,
        "forget_report": report,
        "before_reinforcement_count": int(before_pm["reinforcement_count"]),
        "after_persistent_memory": dict(after_pm) if after_pm else None,
        "stale_narratives": int(stale_narratives),
        "visible_events_after_forget": visible_after_forget,
    }


def _check_recall(qresult: QueryResult, expectations: dict, scenario: ScenarioResult) -> None:
    raw = qresult.raw or {}
    events = raw.get("events", []) or []

    min_events = expectations.get("min_events")
    if isinstance(min_events, int):
        ok = len(events) >= min_events
        qresult.add_check(
            ok,
            f"expected >= {min_events} events, got {len(events)}",
        )
        if ok:
            scenario.event_true_positives += 1
        else:
            scenario.event_false_negatives += 1

    summary_substring = expectations.get("must_have_event_summary_substring")
    if isinstance(summary_substring, str):
        ok = any(summary_substring in (event.get("summary") or "") for event in events)
        qresult.add_check(
            ok,
            f"no event summary contained {summary_substring!r}",
        )
        if ok:
            scenario.event_true_positives += 1
        else:
            scenario.event_false_negatives += 1

    action_type = expectations.get("must_have_event_action_type")
    if isinstance(action_type, str):
        ok = any(
            any((act.get("action_type") or "") == action_type for act in event.get("actions") or [])
            for event in events
        )
        qresult.add_check(
            ok,
            f"no event had action_type={action_type!r}",
        )
        if ok:
            scenario.event_true_positives += 1
        else:
            scenario.event_false_negatives += 1

    min_coverage = expectations.get("min_evidence_coverage")
    if isinstance(min_coverage, (int, float)):
        coverage = scenario.evidence_coverage
        ok = coverage >= float(min_coverage)
        qresult.add_check(
            ok,
            f"evidence_coverage {coverage} < {min_coverage}",
        )

    forbidden_substring = expectations.get("must_not_have_event_summary_substring")
    if isinstance(forbidden_substring, str):
        offenders = [
            event for event in events if forbidden_substring in (event.get("summary") or "")
        ]
        if offenders:
            scenario.event_false_positives += len(offenders)
        qresult.add_check(
            not offenders,
            f"forbidden substring {forbidden_substring!r} appeared in {len(offenders)} events",
        )

    min_narratives = expectations.get("min_entity_narratives")
    if isinstance(min_narratives, int):
        narratives = raw.get("entity_narratives") or []
        ok = len(narratives) >= min_narratives
        qresult.add_check(
            ok,
            f"expected >= {min_narratives} entity_narratives, got {len(narratives)}",
        )

    narrative_substring = expectations.get("narrative_content_substring")
    if isinstance(narrative_substring, str):
        narratives = raw.get("entity_narratives") or []
        ok = any(narrative_substring in (n.get("content") or "") for n in narratives)
        qresult.add_check(ok, f"no entity narrative contained {narrative_substring!r}")

    min_segment_heat = expectations.get("min_segment_heat")
    if isinstance(min_segment_heat, (int, float)):
        # Look up the highest segment heat in the underlying store via the
        # shared scenario report — recall does not return segments directly.
        max_heat = scenario.max_segment_heat
        ok = max_heat >= float(min_segment_heat)
        qresult.add_check(
            ok,
            f"max segment heat {max_heat} < {min_segment_heat}",
        )


def _check_entity_timeline(qresult: QueryResult, expectations: dict, scenario: ScenarioResult) -> None:
    raw = qresult.raw or {}
    if not raw.get("entity"):
        qresult.add_check(False, "entity was not found")
        scenario.event_false_negatives += 1
        return

    min_events = expectations.get("min_events")
    if isinstance(min_events, int):
        events = raw.get("events", []) or []
        ok = len(events) >= min_events
        qresult.add_check(ok, f"expected >= {min_events} entity events, got {len(events)}")
        if ok:
            scenario.event_true_positives += 1
        else:
            scenario.event_false_negatives += 1

    must_include = expectations.get("anchor_labels_must_include") or []
    if must_include:
        labels = {anchor.get("label") for anchor in raw.get("anchors", []) or []}
        for label in must_include:
            ok = label in labels
            qresult.add_check(ok, f"anchor label {label!r} not on entity timeline")
            if ok:
                scenario.event_true_positives += 1
            else:
                scenario.event_false_negatives += 1

    if expectations.get("open_questions_status_should_include_answered"):
        questions = raw.get("open_questions", []) or []
        ok = any(question.get("status") == "answered" for question in questions)
        qresult.add_check(ok, "no open question reached status='answered'")
        if ok:
            scenario.event_true_positives += 1
        else:
            scenario.event_false_negatives += 1

    answer_substring = expectations.get("answer_text_substring")
    if isinstance(answer_substring, str):
        questions = raw.get("open_questions", []) or []
        ok = any(answer_substring in (question.get("answer_text") or "") for question in questions)
        qresult.add_check(ok, f"no open-question answer contained {answer_substring!r}")

    if expectations.get("narrative_must_be_present"):
        ok = bool(raw.get("narrative"))
        qresult.add_check(ok, "expected narrative to be present on the entity timeline")

    role_types = expectations.get("role_types_must_include") or []
    if role_types:
        observed = {role.get("role_type") for role in raw.get("roles", []) or []}
        for role_type in role_types:
            ok = role_type in observed
            qresult.add_check(ok, f"role_type {role_type!r} not on entity timeline")
            if ok:
                scenario.event_true_positives += 1
            else:
                scenario.event_false_negatives += 1

    state_types = expectations.get("state_types_must_include") or []
    if state_types:
        observed = {state.get("state_type") for state in raw.get("states", []) or []}
        for state_type in state_types:
            ok = state_type in observed
            qresult.add_check(ok, f"state_type {state_type!r} not on entity timeline")
            if ok:
                scenario.event_true_positives += 1
            else:
                scenario.event_false_negatives += 1

    state_values = expectations.get("state_values_must_include") or []
    if state_values:
        observed_values = [state.get("value") or "" for state in raw.get("states", []) or []]
        for value in state_values:
            ok = any(value in observed for observed in observed_values)
            qresult.add_check(ok, f"state value substring {value!r} not on entity timeline")
            if ok:
                scenario.event_true_positives += 1
            else:
                scenario.event_false_negatives += 1

    min_conflicts = expectations.get("min_conflicts")
    if isinstance(min_conflicts, int):
        conflicts = raw.get("memory_conflicts", []) or []
        ok = len(conflicts) >= min_conflicts
        qresult.add_check(ok, f"expected >= {min_conflicts} conflicts, got {len(conflicts)}")


def _check_forget_event(qresult: QueryResult, expectations: dict, scenario: ScenarioResult) -> None:
    raw = qresult.raw or {}
    if raw.get("error"):
        qresult.add_check(False, raw["error"])
        scenario.event_false_negatives += 1
        return

    if expectations.get("narratives_should_be_stale"):
        ok = int(raw.get("stale_narratives") or 0) >= 1
        qresult.add_check(ok, "expected at least one entity narrative to be marked stale")

    max_reinforcement = expectations.get("max_reinforcement_count_after_forget")
    if isinstance(max_reinforcement, int):
        pm = raw.get("after_persistent_memory") or {}
        actual = int(pm.get("reinforcement_count") or 0)
        ok = actual <= max_reinforcement
        qresult.add_check(
            ok,
            f"persistent memory reinforcement_count {actual} > {max_reinforcement}",
        )

    status = expectations.get("persistent_memory_status")
    if isinstance(status, str):
        pm = raw.get("after_persistent_memory") or {}
        ok = pm.get("status") == status
        qresult.add_check(ok, f"persistent memory status {pm.get('status')!r} != {status!r}")

    if expectations.get("forgotten_event_hidden_from_search"):
        visible = raw.get("visible_events_after_forget") or []
        target = raw.get("target_event_id")
        ok = all(event.get("id") != target for event in visible)
        qresult.add_check(ok, "forgotten event was still visible in default search_events")


def _build_report(
    document: dict,
    path: Path,
    results: list[ScenarioResult],
) -> dict[str, Any]:
    aggregate_passed = sum(result.passed for result in results)
    aggregate_failed = sum(result.failed for result in results)
    f1_values = [result.event_recall_f1 for result in results if result.error is None]
    coverage_values = [result.evidence_coverage for result in results if result.error is None]
    hallucination_values = [result.hallucination_rate for result in results if result.error is None]

    summary = {
        "scenario_count": len(results),
        "expectations_passed": aggregate_passed,
        "expectations_failed": aggregate_failed,
        "macro_event_recall_f1": _safe_mean(f1_values),
        "macro_evidence_coverage": _safe_mean(coverage_values),
        "macro_hallucination_rate": _safe_mean(hallucination_values),
    }

    return {
        "benchmark": document.get("name", path.stem),
        "scenario_file": str(path),
        "description": document.get("description", ""),
        "summary": summary,
        "scenarios": [_scenario_to_dict(result) for result in results],
    }


def _scenario_to_dict(result: ScenarioResult) -> dict:
    return {
        "name": result.name,
        "description": result.description,
        "error": result.error,
        "expectations_passed": result.passed,
        "expectations_failed": result.failed,
        "event_recall_f1": result.event_recall_f1,
        "evidence_coverage": result.evidence_coverage,
        "hallucination_rate": result.hallucination_rate,
        "max_segment_heat": round(result.max_segment_heat, 4),
        "persistent_memory_count": result.persistent_memory_count,
        "open_conflicts": result.open_conflicts,
        "queries": [
            {
                "name": query.name,
                "query_type": query.query_type,
                "passed": query.passed,
                "failed": query.failed,
                "failures": query.failures,
            }
            for query in result.queries
        ],
    }


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


__all__ = [
    "run_benchmark",
    "run_episodic_benchmark",
    "ScenarioResult",
    "QueryResult",
    "DEFAULT_SCENARIO_FILE",
    "EPISODIC_HARD_SCENARIO_FILE",
]
