"""LoCoMo-style retrieval benchmark harness.

The official LoCoMo dataset is intentionally small but long: ten
multi-session conversations with QA annotations and dialog-id evidence.
This harness evaluates the memory store's retrieval layer without making
LLM calls. It ingests a LoCoMo JSON file into an isolated local store,
runs the deterministic background workers, asks each QA question through
``MemoryStore.recall()``, and reports evidence/answer-string retrieval
metrics.

This is not an LLM-as-judge answer benchmark. It is the cheaper first
gate: did the memory system retrieve the right evidence and enough answer
surface to let an agent answer the question?
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4
import json
import re
import shutil

from .config import Settings
from .store import MemoryStore
from .worker import JobWorker


LOCOMO_TMP_ROOT = Path.cwd() / ".codex-tmp" / "benchmark-locomo-runs"
DEFAULT_LOCOMO_CATEGORIES = ("1", "2", "3", "4")


@dataclass
class LocomoQuestionResult:
    sample_id: str
    question: str
    answer: str
    category: str
    evidence: list[str]
    retrieved_dia_ids: list[str]
    evidence_hits: list[str]
    answer_string_hit: bool

    @property
    def evidence_recall(self) -> float | None:
        if not self.evidence:
            return None
        return round(len(set(self.evidence_hits)) / len(set(self.evidence)), 4)

    @property
    def any_evidence_hit(self) -> bool | None:
        if not self.evidence:
            return None
        return bool(set(self.evidence_hits))


@dataclass
class LocomoSampleResult:
    sample_id: str
    ingested_observations: int = 0
    processed_jobs: int = 0
    questions: list[LocomoQuestionResult] = field(default_factory=list)
    error: str | None = None


def run_locomo_benchmark(
    *,
    dataset_file: str | Path,
    sample_limit: int | None = None,
    max_questions: int | None = None,
    categories: list[str] | tuple[str, ...] | None = None,
    source_mode: str = "dialogs",
    recall_limit: int = 10,
    max_jobs: int = 1000,
) -> dict[str, Any]:
    """Run a LoCoMo-style retrieval benchmark.

    Parameters are JSON-serializable by design so the CLI can expose them
    directly. ``source_mode`` may be one of:

    * ``dialogs``: ingest every dialog turn with session timestamp context.
    * ``observations``: ingest generated LoCoMo session observations.
    * ``session_summaries``: ingest generated LoCoMo session summaries.
    * a ``+`` combination, for example ``dialogs+observations``.
    """

    path = Path(dataset_file)
    if not path.exists():
        raise FileNotFoundError(f"LoCoMo dataset file not found: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    samples = _as_samples(document)
    if sample_limit is not None:
        samples = samples[: max(0, sample_limit)]
    category_set = {str(cat) for cat in (categories or DEFAULT_LOCOMO_CATEGORIES)}

    results: list[LocomoSampleResult] = []
    question_budget = max_questions
    for sample in samples:
        if question_budget is not None and question_budget <= 0:
            break
        result = _run_sample(
            sample,
            source_mode=source_mode,
            recall_limit=recall_limit,
            max_jobs=max_jobs,
            categories=category_set,
            max_questions=question_budget,
        )
        results.append(result)
        if question_budget is not None:
            question_budget -= len(result.questions)

    return _build_report(
        dataset_file=path,
        source_mode=source_mode,
        recall_limit=recall_limit,
        categories=sorted(category_set),
        results=results,
    )


def _run_sample(
    sample: dict,
    *,
    source_mode: str,
    recall_limit: int,
    max_jobs: int,
    categories: set[str],
    max_questions: int | None,
) -> LocomoSampleResult:
    sample_id = str(sample.get("sample_id") or sample.get("id") or uuid4().hex)
    result = LocomoSampleResult(sample_id=sample_id)
    LOCOMO_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    tmp_root = LOCOMO_TMP_ROOT / f"ltm-locomo-{uuid4().hex}"
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
            result.ingested_observations = _ingest_sample(
                store,
                sample,
                sample_id=sample_id,
                source_mode=source_mode,
            )
            result.processed_jobs = int(JobWorker(store).process(max_jobs=max_jobs)["processed"])
            for qa in _iter_qa(sample, categories=categories):
                if max_questions is not None and len(result.questions) >= max_questions:
                    break
                result.questions.append(
                    _evaluate_question(
                        store,
                        sample_id=sample_id,
                        qa=qa,
                        recall_limit=recall_limit,
                    )
                )
        finally:
            store.close()
    except Exception as exc:  # noqa: BLE001 - benchmark should report failures
        result.error = f"{type(exc).__name__}: {exc}"
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    return result


def _ingest_sample(
    store: MemoryStore,
    sample: dict,
    *,
    sample_id: str,
    source_mode: str,
) -> int:
    modes = {part.strip() for part in source_mode.split("+") if part.strip()}
    allowed_modes = {"dialogs", "observations", "session_summaries"}
    unknown_modes = sorted(modes - allowed_modes)
    if unknown_modes:
        raise ValueError(f"unknown LoCoMo source_mode value(s): {', '.join(unknown_modes)}")
    count = 0
    if "dialogs" in modes:
        for session_key, session_datetime, turn in _iter_dialog_turns(sample):
            dia_id = str(turn.get("dia_id") or f"{session_key}:{count + 1}")
            speaker = str(turn.get("speaker") or "speaker")
            text = str(turn.get("text") or turn.get("content") or "").strip()
            if not text:
                continue
            caption = _image_caption(turn)
            content = f"[{session_datetime}] {speaker}: {text}"
            if caption:
                content = f"{content}\nImage caption: {caption}"
            store.ingest_observation(
                source_type="locomo_dialog",
                content=content,
                session_id=f"{sample_id}:{session_key}",
                client_dedup_key=f"{sample_id}:{dia_id}",
                metadata={
                    "benchmark": "locomo",
                    "sample_id": sample_id,
                    "source_mode": "dialogs",
                    "session_key": session_key,
                    "session_datetime": session_datetime,
                    "dia_id": dia_id,
                    "speaker": speaker,
                },
            )
            count += 1
    if "observations" in modes:
        count += _ingest_session_level_records(
            store,
            sample,
            sample_id=sample_id,
            group_key="observation",
            suffix="_observation",
            source_mode="observations",
            source_type="locomo_observation",
        )
    if "session_summaries" in modes:
        count += _ingest_session_level_records(
            store,
            sample,
            sample_id=sample_id,
            group_key="session_summary",
            suffix="_summary",
            source_mode="session_summaries",
            source_type="locomo_session_summary",
        )
    return count


def _ingest_session_level_records(
    store: MemoryStore,
    sample: dict,
    *,
    sample_id: str,
    group_key: str,
    suffix: str,
    source_mode: str,
    source_type: str,
) -> int:
    group = sample.get(group_key) or {}
    if not isinstance(group, dict):
        return 0
    count = 0
    for key, value in sorted(group.items()):
        session_key = _session_key_from_annotation_key(key, suffix=suffix)
        content = _textify(value).strip()
        if not content:
            continue
        store.ingest_observation(
            source_type=source_type,
            content=content,
            session_id=f"{sample_id}:{session_key}",
            client_dedup_key=f"{sample_id}:{source_mode}:{key}",
            metadata={
                "benchmark": "locomo",
                "sample_id": sample_id,
                "source_mode": source_mode,
                "session_key": session_key,
            },
        )
        count += 1
    return count


def _evaluate_question(
    store: MemoryStore,
    *,
    sample_id: str,
    qa: dict,
    recall_limit: int,
) -> LocomoQuestionResult:
    question = str(qa.get("question") or qa.get("query") or "")
    answer = _answer_to_text(qa.get("answer"))
    evidence = [str(item) for item in (qa.get("evidence") or []) if item]
    category = str(qa.get("category") or qa.get("question_type") or "unknown")
    raw = store.recall(
        question,
        limit=recall_limit,
        limits={
            "observations": recall_limit,
            "events": recall_limit,
            "narratives": 5,
            "persistent_memories": 5,
            "open_questions": 5,
        },
        scope="global",
        include_session_buffer=False,
    )
    observation_ids = _retrieved_observation_ids(store, raw)
    retrieved_dia_ids = _dia_ids_for_observations(store, observation_ids)
    context = _retrieved_context_text(store, raw, observation_ids)
    evidence_set = set(evidence)
    return LocomoQuestionResult(
        sample_id=sample_id,
        question=question,
        answer=answer,
        category=category,
        evidence=evidence,
        retrieved_dia_ids=retrieved_dia_ids,
        evidence_hits=sorted(evidence_set & set(retrieved_dia_ids)),
        answer_string_hit=_answer_string_hit(answer, context),
    )


def _retrieved_observation_ids(store: MemoryStore, raw: dict) -> list[str]:
    ids: list[str] = []
    for observation in raw.get("observations") or []:
        obs_id = observation.get("id")
        if obs_id and obs_id not in ids:
            ids.append(obs_id)
    event_ids = [event.get("id") for event in raw.get("events") or [] if event.get("id")]
    if event_ids:
        placeholders = ",".join("?" for _ in event_ids)
        rows = store.conn.execute(
            f"""
            SELECT DISTINCT observation_id
            FROM evidence
            WHERE event_id IN ({placeholders}) AND status != 'stale'
            """,
            event_ids,
        ).fetchall()
        for row in rows:
            obs_id = row["observation_id"]
            if obs_id and obs_id not in ids:
                ids.append(obs_id)
    return ids


def _dia_ids_for_observations(store: MemoryStore, observation_ids: list[str]) -> list[str]:
    if not observation_ids:
        return []
    placeholders = ",".join("?" for _ in observation_ids)
    rows = store.conn.execute(
        f"SELECT id, metadata_json FROM observations WHERE id IN ({placeholders})",
        observation_ids,
    ).fetchall()
    dia_by_obs: dict[str, str] = {}
    for row in rows:
        try:
            metadata = json.loads(row["metadata_json"] or "{}")
        except json.JSONDecodeError:
            metadata = {}
        dia_id = metadata.get("dia_id")
        if dia_id:
            dia_by_obs[row["id"]] = str(dia_id)
    return [dia_by_obs[obs_id] for obs_id in observation_ids if obs_id in dia_by_obs]


def _retrieved_context_text(store: MemoryStore, raw: dict, observation_ids: list[str]) -> str:
    pieces: list[str] = []
    if observation_ids:
        placeholders = ",".join("?" for _ in observation_ids)
        rows = store.conn.execute(
            f"SELECT content FROM observations WHERE id IN ({placeholders})",
            observation_ids,
        ).fetchall()
        pieces.extend(str(row["content"]) for row in rows)
    pieces.extend(str(event.get("summary") or "") for event in raw.get("events") or [])
    pieces.extend(str(item.get("content") or "") for item in raw.get("entity_narratives") or [])
    pieces.extend(str(item.get("content") or "") for item in raw.get("persistent_memories") or [])
    pieces.extend(str(item.get("answer_text") or "") for item in raw.get("open_questions") or [])
    return "\n".join(piece for piece in pieces if piece)


def _build_report(
    *,
    dataset_file: Path,
    source_mode: str,
    recall_limit: int,
    categories: list[str],
    results: list[LocomoSampleResult],
) -> dict[str, Any]:
    questions = [question for result in results for question in result.questions]
    evidence_questions = [q for q in questions if q.evidence]
    summary = {
        "sample_count": len(results),
        "question_count": len(questions),
        "samples_with_errors": sum(1 for result in results if result.error),
        "ingested_observations": sum(result.ingested_observations for result in results),
        "processed_jobs": sum(result.processed_jobs for result in results),
        "evidence_recall_at_k": _safe_mean(
            [q.evidence_recall for q in evidence_questions if q.evidence_recall is not None]
        ),
        "any_evidence_hit_rate": _safe_mean(
            [1.0 if q.any_evidence_hit else 0.0 for q in evidence_questions]
        ),
        "answer_string_hit_rate": _safe_mean(
            [1.0 if q.answer_string_hit else 0.0 for q in questions]
        ),
    }
    return {
        "benchmark": "locomo_retrieval",
        "dataset_file": str(dataset_file),
        "source_mode": source_mode,
        "recall_limit": recall_limit,
        "categories": categories,
        "summary": summary,
        "category_metrics": _category_metrics(questions),
        "samples": [_sample_to_dict(result) for result in results],
    }


def _category_metrics(questions: list[LocomoQuestionResult]) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[LocomoQuestionResult]] = {}
    for question in questions:
        grouped.setdefault(question.category, []).append(question)
    metrics: dict[str, dict[str, float | int]] = {}
    for category, items in sorted(grouped.items()):
        evidence_items = [item for item in items if item.evidence]
        metrics[category] = {
            "question_count": len(items),
            "evidence_recall_at_k": _safe_mean(
                [item.evidence_recall for item in evidence_items if item.evidence_recall is not None]
            ),
            "any_evidence_hit_rate": _safe_mean(
                [1.0 if item.any_evidence_hit else 0.0 for item in evidence_items]
            ),
            "answer_string_hit_rate": _safe_mean(
                [1.0 if item.answer_string_hit else 0.0 for item in items]
            ),
        }
    return metrics


def _sample_to_dict(result: LocomoSampleResult) -> dict[str, Any]:
    return {
        "sample_id": result.sample_id,
        "error": result.error,
        "ingested_observations": result.ingested_observations,
        "processed_jobs": result.processed_jobs,
        "question_count": len(result.questions),
        "questions": [
            {
                "question": question.question,
                "answer": question.answer,
                "category": question.category,
                "evidence": question.evidence,
                "retrieved_dia_ids": question.retrieved_dia_ids,
                "evidence_hits": question.evidence_hits,
                "evidence_recall": question.evidence_recall,
                "answer_string_hit": question.answer_string_hit,
            }
            for question in result.questions
        ],
    }


def _as_samples(document: Any) -> list[dict]:
    if isinstance(document, list):
        return [sample for sample in document if isinstance(sample, dict)]
    if isinstance(document, dict):
        for key in ("samples", "data", "conversations"):
            value = document.get(key)
            if isinstance(value, list):
                return [sample for sample in value if isinstance(sample, dict)]
        if "conversation" in document:
            return [document]
    raise ValueError("LoCoMo dataset must be a list of samples or an object containing samples/data")


def _iter_dialog_turns(sample: dict):
    conversation = sample.get("conversation") or {}
    if not isinstance(conversation, dict):
        return
    for key in sorted(conversation.keys(), key=_session_sort_key):
        if not re.fullmatch(r"session_\d+", key):
            continue
        turns = conversation.get(key)
        if not isinstance(turns, list):
            continue
        session_datetime = str(conversation.get(f"{key}_date_time") or "unknown time")
        for turn in turns:
            if isinstance(turn, dict):
                yield key, session_datetime, turn


def _iter_qa(sample: dict, *, categories: set[str]):
    qa_rows = sample.get("qa") or sample.get("questions") or []
    if not isinstance(qa_rows, list):
        return
    for qa in qa_rows:
        if not isinstance(qa, dict):
            continue
        category = str(qa.get("category") or qa.get("question_type") or "unknown")
        if categories and category not in categories:
            continue
        if qa.get("question") or qa.get("query"):
            yield qa


def _session_sort_key(key: str) -> tuple[int, str]:
    match = re.search(r"session_(\d+)", key)
    if not match:
        return (10**9, key)
    return (int(match.group(1)), key)


def _session_key_from_annotation_key(key: str, *, suffix: str) -> str:
    normalized = key
    if normalized.endswith(suffix):
        normalized = normalized[: -len(suffix)]
    match = re.search(r"session_\d+", normalized)
    return match.group(0) if match else normalized


def _image_caption(turn: dict) -> str:
    caption = turn.get("blip_caption")
    if isinstance(caption, list):
        return "; ".join(str(item) for item in caption if item)
    if caption:
        return str(caption)
    return ""


def _textify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(_textify(item) for item in value if item)
    if isinstance(value, dict):
        pieces = []
        for key, item in sorted(value.items()):
            text = _textify(item)
            if text:
                pieces.append(f"{key}: {text}")
        return "\n".join(pieces)
    return str(value)


def _answer_to_text(value: Any) -> str:
    if isinstance(value, list):
        return "; ".join(str(item) for item in value if item)
    if value is None:
        return ""
    return str(value)


def _answer_string_hit(answer: str, context: str) -> bool:
    answer_norm = _normalize_text(answer)
    context_norm = _normalize_text(context)
    if not answer_norm:
        return False
    if answer_norm in context_norm:
        return True
    answer_tokens = _significant_tokens(answer_norm)
    if not answer_tokens:
        return False
    overlap = sum(1 for token in answer_tokens if token in context_norm)
    return (overlap / len(answer_tokens)) >= 0.6


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()


def _significant_tokens(value: str) -> list[str]:
    stop = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "to",
        "of",
        "in",
        "on",
        "at",
        "for",
        "with",
        "was",
        "is",
        "did",
        "do",
    }
    return [
        token
        for token in re.findall(r"[\w\u4e00-\u9fff]+", value)
        if len(token) >= 3 and token not in stop
    ]


def _safe_mean(values: list[float | None]) -> float:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return 0.0
    return round(sum(clean) / len(clean), 4)


__all__ = [
    "run_locomo_benchmark",
    "DEFAULT_LOCOMO_CATEGORIES",
    "LocomoSampleResult",
    "LocomoQuestionResult",
]
