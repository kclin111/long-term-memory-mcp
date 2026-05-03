from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
import json
import sqlite3
import traceback

from .lance_index import LanceMemoryIndex
from .lifecycle import (
    ConflictDetector,
    EntityNarrativeCache,
    PersistentMemoryStore,
    SegmentManager,
)
from .operator import Operator, OperatorResult
from .store import MemoryStore, new_id, normalize_entity_name
from .time import utc_now


def _json_dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _parse_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


class ExtractWorker:
    """Local observation -> workspace proposal.

    Wraps a pluggable :class:`Operator`. The Operator falls back to the
    deterministic extractor when no LLM provider is configured or when
    the LLM response is unusable, so this worker always returns a valid
    GSW-style payload (entities, roles, states, actions, anchors, open
    questions).
    """

    def __init__(self, operator: Operator | None = None) -> None:
        self.operator = operator or Operator()

    def extract(self, observation: sqlite3.Row, *, context: dict | None = None) -> OperatorResult:
        return self.operator.extract(dict(observation), context=context)


class IntegrateWorker:
    """Workspace proposal -> canonical SQLite memory graph.

    This is the M2 Reconciler boundary. It owns entity matching, anchor
    propagation, forward-falling question resolution, and the SQLite
    read-modify-write transaction.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self.segments = SegmentManager(conn)
        self.persistent = PersistentMemoryStore(conn)
        self.narratives = EntityNarrativeCache(conn)
        self.conflicts = ConflictDetector(conn)

    def integrate(self, observation: sqlite3.Row, proposal: dict) -> dict:
        existing = self.conn.execute(
            "SELECT event_id FROM evidence WHERE observation_id = ? AND event_id IS NOT NULL LIMIT 1",
            (observation["id"],),
        ).fetchone()
        if existing:
            return {"status": "deduplicated", "event_id": existing["event_id"]}

        now = utc_now()
        event_id = new_id("evt")
        evidence_id = new_id("evd")
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO events(
                  id, event_type, summary, structured_json, observed_at, occurred_at,
                  importance, novelty, confidence, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'extracted', ?, ?)
                """,
                (
                    event_id,
                    proposal["event_type"],
                    proposal["summary"],
                    _json_dumps(proposal["structured"]),
                    observation["observed_at"],
                    observation["observed_at"],
                    proposal["importance"],
                    proposal["novelty"],
                    proposal["confidence"],
                    now,
                    now,
                ),
            )
            self.conn.execute(
                """
                INSERT INTO evidence(id, observation_id, event_id, span_start, span_end, content_excerpt, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    evidence_id,
                    observation["id"],
                    event_id,
                    0,
                    len(observation["content"]),
                    proposal["evidence_excerpt"],
                    now,
                ),
            )

            entity_ids = self._reify_entities(event_id, proposal["entities"], now)
            explicit_anchor_ids = self._reify_anchors(observation["id"], proposal["anchors"], now)
            propagated_anchor_ids = []
            if not explicit_anchor_ids:
                propagated_anchor_ids = self._propagate_anchors_from_entities(entity_ids.values())
            all_anchor_ids = list(dict.fromkeys([*explicit_anchor_ids, *propagated_anchor_ids]))

            self._link_event_and_entity_anchors(
                event_id=event_id,
                entity_ids=entity_ids.values(),
                explicit_anchor_ids=explicit_anchor_ids,
                propagated_anchor_ids=propagated_anchor_ids,
                now=now,
            )
            role_ids = self._reify_roles(
                event_id=event_id,
                roles=proposal["roles"],
                entity_ids=entity_ids,
                valid_from=observation["observed_at"],
                now=now,
            )
            self._reify_states(
                event_id=event_id,
                states=proposal["states"],
                entity_ids=entity_ids,
                role_ids=role_ids,
                valid_from=observation["observed_at"],
                now=now,
            )
            self._reify_actions(
                event_id=event_id,
                actions=proposal["actions"],
                entity_ids=entity_ids,
                now=now,
            )
            answered = self._resolve_open_questions(
                event_id=event_id,
                entity_ids=list(entity_ids.values()),
                anchor_ids=all_anchor_ids,
                now=now,
            )
            created_questions = self._reify_open_questions(
                questions=proposal["open_questions"],
                entity_ids=entity_ids,
                anchor_ids=all_anchor_ids,
                now=now,
            )

            assignment = self.segments.assign_event(
                event_id=event_id,
                event_summary=proposal["summary"],
                event_type=proposal["event_type"],
                event_importance=float(proposal["importance"]),
                entity_ids=entity_ids.values(),
                observed_at=observation["observed_at"],
                now=now,
            )

            promoted_memory: dict | None = None
            if assignment.promoted:
                promoted_memory = self.persistent.promote_segment(assignment.segment_id, now=now)

            for entity_id in entity_ids.values():
                self.narratives.invalidate_for_entity(entity_id, now=now)

        return {
            "status": "created",
            "event_id": event_id,
            "summary": proposal["summary"],
            "event_type": proposal["event_type"],
            "importance": proposal["importance"],
            "confidence": proposal["confidence"],
            "answered_open_questions": answered,
            "created_open_questions": created_questions,
            "segment_id": assignment.segment_id,
            "segment_heat": assignment.heat,
            "segment_promoted": assignment.promoted,
            "persistent_memory": promoted_memory,
        }

    def _reify_entities(self, event_id: str, entities: list[dict], now: str) -> dict[str, str]:
        entity_ids: dict[str, str] = {}
        for entity in entities:
            entity_id = self._get_or_create_entity(
                canonical_name=entity["canonical_name"],
                entity_type=entity["entity_type"],
                now=now,
            )
            entity_ids[normalize_entity_name(entity["canonical_name"])] = entity_id
            self.conn.execute(
                """
                INSERT OR IGNORE INTO event_entities(event_id, entity_id, role_in_event, confidence, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    entity_id,
                    entity["role_in_event"],
                    entity["confidence"],
                    now,
                ),
            )
        return entity_ids

    def _get_or_create_entity(self, *, canonical_name: str, entity_type: str, now: str) -> str:
        normalized = normalize_entity_name(canonical_name)
        row = self.conn.execute(
            "SELECT id FROM entities WHERE entity_type = ? AND normalized_name = ?",
            (entity_type, normalized),
        ).fetchone()
        if row:
            self.conn.execute("UPDATE entities SET last_seen_at = ? WHERE id = ?", (now, row["id"]))
            return row["id"]
        entity_id = new_id("ent")
        self.conn.execute(
            """
            INSERT INTO entities(id, entity_type, canonical_name, normalized_name, aliases_json, description, created_at, last_seen_at)
            VALUES (?, ?, ?, ?, '[]', '', ?, ?)
            """,
            (entity_id, entity_type, canonical_name, normalized, now, now),
        )
        return entity_id

    def _reify_anchors(self, observation_id: str, anchors: list[dict], now: str) -> list[str]:
        anchor_ids: list[str] = []
        for anchor in anchors:
            row = self.conn.execute(
                """
                SELECT id, confidence, status FROM spatiotemporal_anchors
                WHERE anchor_type = ? AND normalized_value = ? AND granularity = ?
                """,
                (anchor["anchor_type"], anchor["normalized_value"], anchor["granularity"]),
            ).fetchone()
            if row:
                anchor_id = row["id"]
                if anchor["confidence"] > float(row["confidence"]):
                    self.conn.execute(
                        """
                        UPDATE spatiotemporal_anchors
                        SET label = ?, confidence = ?, status = ?, updated_at = ?
                        WHERE id = ?
                        """,
                        (anchor["label"], anchor["confidence"], anchor["status"], now, anchor_id),
                    )
            else:
                anchor_id = new_id("anc")
                self.conn.execute(
                    """
                    INSERT INTO spatiotemporal_anchors(
                      id, anchor_type, label, normalized_value, granularity, source_observation_id,
                      confidence, status, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        anchor_id,
                        anchor["anchor_type"],
                        anchor["label"],
                        anchor["normalized_value"],
                        anchor["granularity"],
                        observation_id,
                        anchor["confidence"],
                        anchor["status"],
                        now,
                        now,
                    ),
                )
            anchor_ids.append(anchor_id)
        return anchor_ids

    def _propagate_anchors_from_entities(self, entity_ids: Iterable[str]) -> list[str]:
        ids = list(entity_ids)
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        rows = self.conn.execute(
            f"""
            SELECT DISTINCT anchor_id
            FROM entity_anchors
            WHERE entity_id IN ({placeholders})
            ORDER BY confidence DESC
            LIMIT 8
            """,
            ids,
        ).fetchall()
        return [row["anchor_id"] for row in rows]

    def _link_event_and_entity_anchors(
        self,
        *,
        event_id: str,
        entity_ids: Iterable[str],
        explicit_anchor_ids: list[str],
        propagated_anchor_ids: list[str],
        now: str,
    ) -> None:
        for anchor_id in explicit_anchor_ids:
            self.conn.execute(
                """
                INSERT OR IGNORE INTO event_anchors(event_id, anchor_id, relation, confidence, created_at)
                VALUES (?, ?, 'explicit_context', 0.82, ?)
                """,
                (event_id, anchor_id, now),
            )
        for anchor_id in propagated_anchor_ids:
            self.conn.execute(
                """
                INSERT OR IGNORE INTO event_anchors(event_id, anchor_id, relation, confidence, created_at)
                VALUES (?, ?, 'propagated_context', 0.58, ?)
                """,
                (event_id, anchor_id, now),
            )
        for entity_id in entity_ids:
            for anchor_id in explicit_anchor_ids:
                self.conn.execute(
                    """
                    INSERT OR IGNORE INTO entity_anchors(entity_id, anchor_id, relation, confidence, source_event_id, created_at)
                    VALUES (?, ?, 'explicit_coupled', 0.78, ?, ?)
                    """,
                    (entity_id, anchor_id, event_id, now),
                )
            for anchor_id in propagated_anchor_ids:
                self.conn.execute(
                    """
                    INSERT OR IGNORE INTO entity_anchors(entity_id, anchor_id, relation, confidence, source_event_id, created_at)
                    VALUES (?, ?, 'propagated_coupled', 0.56, ?, ?)
                    """,
                    (entity_id, anchor_id, event_id, now),
                )

    def _reify_roles(
        self,
        *,
        event_id: str,
        roles: list[dict],
        entity_ids: dict[str, str],
        valid_from: str,
        now: str,
    ) -> dict[tuple[str, str], str]:
        role_ids: dict[tuple[str, str], str] = {}
        for role in roles:
            entity_id = entity_ids.get(normalize_entity_name(role["entity"]))
            if not entity_id:
                continue
            row = self.conn.execute(
                """
                SELECT id, confidence FROM entity_roles
                WHERE entity_id = ? AND role_type = ? AND scope_entity_id IS NULL AND status = 'active'
                ORDER BY confidence DESC
                LIMIT 1
                """,
                (entity_id, role["role_type"]),
            ).fetchone()
            if row:
                role_id = row["id"]
                if role["confidence"] > float(row["confidence"]):
                    self.conn.execute(
                        "UPDATE entity_roles SET confidence = ?, updated_at = ? WHERE id = ?",
                        (role["confidence"], now, role_id),
                    )
            else:
                role_id = new_id("role")
                self.conn.execute(
                    """
                    INSERT INTO entity_roles(
                      id, entity_id, role_type, scope_entity_id, source_event_id, confidence,
                      valid_from, valid_to, status, created_at, updated_at
                    ) VALUES (?, ?, ?, NULL, ?, ?, ?, NULL, 'active', ?, ?)
                    """,
                    (role_id, entity_id, role["role_type"], event_id, role["confidence"], valid_from, now, now),
                )
            role_ids[(entity_id, role["role_type"])] = role_id
        return role_ids

    def _reify_states(
        self,
        *,
        event_id: str,
        states: list[dict],
        entity_ids: dict[str, str],
        role_ids: dict[tuple[str, str], str],
        valid_from: str,
        now: str,
    ) -> None:
        for state in states:
            entity_id = entity_ids.get(normalize_entity_name(state["entity"]))
            if not entity_id:
                continue
            role_id = None
            if state.get("role_type"):
                role_id = role_ids.get((entity_id, state["role_type"]))

            existing_same = self.conn.execute(
                """
                SELECT id, confidence FROM entity_states
                WHERE entity_id = ? AND state_type = ? AND value = ? AND status = 'active'
                ORDER BY confidence DESC
                LIMIT 1
                """,
                (entity_id, state["state_type"], state["value"]),
            ).fetchone()
            if existing_same:
                if state["confidence"] > float(existing_same["confidence"]):
                    self.conn.execute(
                        "UPDATE entity_states SET confidence = ?, updated_at = ? WHERE id = ?",
                        (state["confidence"], now, existing_same["id"]),
                    )
                continue

            if state["confidence"] >= 0.65:
                self.conn.execute(
                    """
                    UPDATE entity_states
                    SET status = 'superseded', valid_to = ?, updated_at = ?
                    WHERE entity_id = ? AND state_type = ? AND status = 'active' AND confidence <= ?
                    """,
                    (valid_from, now, entity_id, state["state_type"], state["confidence"]),
                )

            state_id = new_id("state")
            self.conn.execute(
                """
                INSERT INTO entity_states(
                  id, entity_id, role_id, state_type, value, value_json, scope_entity_id,
                  source_event_id, confidence, valid_from, valid_to, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, NULL, 'active', ?, ?)
                """,
                (
                    state_id,
                    entity_id,
                    role_id,
                    state["state_type"],
                    state["value"],
                    _json_dumps({"extractor": "deterministic_m2"}),
                    event_id,
                    state["confidence"],
                    valid_from,
                    now,
                    now,
                ),
            )
            self.conflicts.evaluate_state(
                entity_id=entity_id,
                state_type=state["state_type"],
                new_value=state["value"],
                new_confidence=float(state["confidence"]),
                new_state_id=state_id,
                valid_from=valid_from,
                now=now,
            )

    def _reify_actions(self, *, event_id: str, actions: list[dict], entity_ids: dict[str, str], now: str) -> None:
        for action in actions:
            actor_id = entity_ids.get(normalize_entity_name(action["actor"])) if action.get("actor") else None
            object_id = entity_ids.get(normalize_entity_name(action["object"])) if action.get("object") else None
            self.conn.execute(
                """
                INSERT INTO event_actions(
                  id, event_id, actor_entity_id, action_type, action_text,
                  object_entity_id, valence, confidence, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    new_id("act"),
                    event_id,
                    actor_id,
                    action["action_type"],
                    action["action_text"],
                    object_id,
                    action["valence"],
                    action["confidence"],
                    now,
                ),
            )

    def _resolve_open_questions(self, *, event_id: str, entity_ids: list[str], anchor_ids: list[str], now: str) -> int:
        if not entity_ids or not anchor_ids:
            return 0
        anchors = self._anchors_by_type(anchor_ids)
        if not anchors:
            return 0
        rows = self.conn.execute(
            "SELECT * FROM open_questions WHERE status = 'open' ORDER BY priority DESC, created_at"
        ).fetchall()
        answered = 0
        entity_set = set(entity_ids)
        for question in rows:
            related = set(json.loads(question["related_entity_ids_json"]))
            subject = question["subject_entity_id"]
            overlaps = (subject in entity_set) or bool(related & entity_set)
            if not overlaps:
                continue
            wanted_anchor_type = "time" if question["question_type"] == "when" else "space"
            labels = anchors.get(wanted_anchor_type, [])
            if not labels:
                continue
            answer = "; ".join(labels)
            self.conn.execute(
                """
                UPDATE open_questions
                SET status = 'answered', answer_event_id = ?, answer_text = ?,
                    confidence = MAX(confidence, 0.68), updated_at = ?, answered_at = ?
                WHERE id = ?
                """,
                (event_id, answer, now, now, question["id"]),
            )
            answered += 1
        return answered

    def _reify_open_questions(
        self,
        *,
        questions: list[dict],
        entity_ids: dict[str, str],
        anchor_ids: list[str],
        now: str,
    ) -> int:
        if not questions:
            return 0
        anchors = self._anchors_by_type(anchor_ids)
        created = 0
        for question in questions:
            if question["question_type"] == "when" and anchors.get("time"):
                continue
            if question["question_type"] == "where" and anchors.get("space"):
                continue
            subject_id = entity_ids.get(normalize_entity_name(question["subject"]))
            related_ids = [
                entity_ids[normalize_entity_name(name)]
                for name in question["related_entities"]
                if normalize_entity_name(name) in entity_ids
            ]
            dedup_key = self._open_question_dedup_key(subject_id, question["question_type"], related_ids)
            cursor = self.conn.execute(
                """
                INSERT OR IGNORE INTO open_questions(
                  id, subject_entity_id, question_type, question_text, dedup_key,
                  related_entity_ids_json, status, priority, confidence, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'open', ?, ?, ?, ?)
                """,
                (
                    new_id("oq"),
                    subject_id,
                    question["question_type"],
                    question["question_text"],
                    dedup_key,
                    _json_dumps(sorted(set(related_ids))),
                    int(question["priority"]),
                    question["confidence"],
                    now,
                    now,
                ),
            )
            if cursor.rowcount > 0:
                created += 1
        return created

    def _anchors_by_type(self, anchor_ids: list[str]) -> dict[str, list[str]]:
        if not anchor_ids:
            return {}
        placeholders = ",".join("?" for _ in anchor_ids)
        rows = self.conn.execute(
            f"SELECT anchor_type, label FROM spatiotemporal_anchors WHERE id IN ({placeholders})",
            anchor_ids,
        ).fetchall()
        grouped: dict[str, list[str]] = {}
        for row in rows:
            grouped.setdefault(row["anchor_type"], []).append(row["label"])
        return grouped

    @staticmethod
    def _open_question_dedup_key(subject_id: str | None, question_type: str, related_ids: list[str]) -> str:
        subject = subject_id or "none"
        related = ",".join(sorted(set(related_ids)))
        return f"{subject}|{question_type}|{related}"


class JobWorker:
    def __init__(self, store: MemoryStore, *, operator: Operator | None = None) -> None:
        self.store = store
        self.conn = store.conn
        self.index = LanceMemoryIndex(store.settings)
        self.extract_worker = ExtractWorker(operator=operator)
        self.integrate_worker = IntegrateWorker(self.conn)

    def process(self, *, max_jobs: int = 50) -> dict:
        self.store.init()
        recovered = self._recover_stale_running_jobs()
        processed = 0
        failed = 0
        skipped = 0
        for job in self._pending_jobs(max_jobs):
            result = self._process_one(job)
            processed += 1
            failed += 1 if result == "failed" else 0
            skipped += 1 if result == "skipped" else 0
        return {
            "processed": processed,
            "failed": failed,
            "skipped": skipped,
            "recovered_running_jobs": recovered,
            "lancedb": self.index.status.__dict__,
        }

    def rebuild_index(self) -> dict:
        self.store.init()
        records = []
        for row in self.conn.execute("SELECT id, event_type, summary, importance, confidence, observed_at FROM events"):
            records.append(
                {
                    "sqlite_table": "events",
                    "sqlite_id": row["id"],
                    "record_type": "event",
                    "text": row["summary"],
                    "metadata": {
                        "event_type": row["event_type"],
                        "importance": row["importance"],
                        "confidence": row["confidence"],
                        "observed_at": row["observed_at"],
                    },
                }
            )
        for row in self.conn.execute("SELECT id, question_type, question_text, status, priority FROM open_questions"):
            records.append(
                {
                    "sqlite_table": "open_questions",
                    "sqlite_id": row["id"],
                    "record_type": "open_question",
                    "text": row["question_text"],
                    "metadata": {
                        "question_type": row["question_type"],
                        "status": row["status"],
                        "priority": row["priority"],
                    },
                }
            )
        status = self.index.upsert(records)
        with self.conn:
            self.conn.execute(
                """
                UPDATE settings
                SET value_json = ?, updated_at = ?
                WHERE key = 'embedding_model_signature'
                """,
                (_json_dumps(self.store.settings.embedding_signature), utc_now()),
            )
        return {"records": len(records), "lancedb": status.__dict__}

    def _pending_jobs(self, max_jobs: int) -> list[sqlite3.Row]:
        return self.conn.execute(
            """
            SELECT * FROM index_jobs
            WHERE status = 'pending'
            ORDER BY scheduled_at, created_at
            LIMIT ?
            """,
            (max_jobs,),
        ).fetchall()

    def _recover_stale_running_jobs(self) -> int:
        timeout = timedelta(seconds=max(1, int(self.store.settings.job_running_timeout_seconds)))
        cutoff = datetime.now(timezone.utc) - timeout
        rows = self.conn.execute(
            """
            SELECT id, started_at
            FROM index_jobs
            WHERE status = 'running'
            """
        ).fetchall()
        stale_ids = [
            row["id"]
            for row in rows
            if (_parse_utc(row["started_at"]) or datetime.min.replace(tzinfo=timezone.utc)) <= cutoff
        ]
        if not stale_ids:
            return 0
        placeholders = ",".join("?" for _ in stale_ids)
        with self.conn:
            self.conn.execute(
                f"""
                UPDATE index_jobs
                SET status = 'pending',
                    started_at = NULL,
                    last_error = COALESCE(last_error, 'recovered stale running job')
                WHERE id IN ({placeholders})
                """,
                stale_ids,
            )
        return len(stale_ids)

    def _process_one(self, job: sqlite3.Row) -> str:
        now = utc_now()
        with self.conn:
            self.conn.execute(
                "UPDATE index_jobs SET status = 'running', started_at = ? WHERE id = ?",
                (now, job["id"]),
            )
        try:
            if job["job_type"] == "extract_event":
                payload = json.loads(job["payload_json"])
                integrated = self._handle_extract_event(payload["observation_id"])
                if integrated.get("status") == "deduplicated":
                    self._complete(job["id"], status="skipped")
                    return "skipped"
            elif job["job_type"] == "upsert_lancedb":
                self.rebuild_index()
            elif job["job_type"] == "delete_lancedb":
                payload = json.loads(job["payload_json"])
                status = self.index.delete(
                    record_type=payload["record_type"],
                    sqlite_id=payload["sqlite_id"],
                )
                if not status.available and status.message.startswith("lancedb delete failed"):
                    raise RuntimeError(status.message)
            elif job["job_type"] == "invalidate_entity_narrative":
                payload = json.loads(job["payload_json"])
                self._handle_invalidate_entity_narrative(payload)
            elif job["job_type"] == "refresh_entity_narrative":
                payload = json.loads(job["payload_json"])
                self._handle_refresh_entity_narrative(payload)
            elif job["job_type"] == "extract_persistent_memory":
                payload = json.loads(job["payload_json"])
                self._handle_extract_persistent_memory(payload)
            elif job["job_type"] == "rebuild_index":
                self.rebuild_index()
            else:
                self._complete(job["id"], status="skipped")
                return "skipped"
        except Exception as exc:
            self._fail(job, exc)
            return "failed"
        self._complete(job["id"])
        return "completed"

    def _handle_extract_event(self, observation_id: str) -> dict:
        observation = self.conn.execute(
            "SELECT * FROM observations WHERE id = ?",
            (observation_id,),
        ).fetchone()
        if not observation:
            return {"status": "missing_observation"}

        operator_result = self.extract_worker.extract(observation)
        integrated = self.integrate_worker.integrate(observation, operator_result.proposal)
        if integrated.get("status") != "created":
            return integrated

        status = self.index.upsert(
            [
                {
                    "sqlite_table": "events",
                    "sqlite_id": integrated["event_id"],
                    "record_type": "event",
                    "text": integrated["summary"],
                    "metadata": {
                        "event_type": integrated["event_type"],
                        "importance": integrated["importance"],
                        "confidence": integrated["confidence"],
                        "observed_at": observation["observed_at"],
                    },
                }
            ]
        )
        if not status.available:
            self._enqueue_lancedb_retry(integrated["event_id"])
        return integrated

    def _handle_invalidate_entity_narrative(self, payload: dict) -> None:
        now = utc_now()
        entity_ids = self._entity_ids_from_payload(payload)
        if not entity_ids:
            return
        with self.conn:
            for entity_id in entity_ids:
                self.store.narratives.invalidate_for_entity(entity_id, now=now)

    def _handle_refresh_entity_narrative(self, payload: dict) -> None:
        now = utc_now()
        entity_ids = self._entity_ids_from_payload(payload)
        scope = payload.get("scope", "global")
        query = payload.get("query")
        with self.conn:
            for entity_id in entity_ids:
                self.store.narratives.invalidate_for_entity(entity_id, now=now)
                self.store.narratives.get_or_generate(
                    entity_id=entity_id,
                    scope=scope,
                    query=query,
                    now=now,
                    force_refresh=True,
                )

    def _handle_extract_persistent_memory(self, payload: dict) -> None:
        now = utc_now()
        segment_id = payload.get("segment_id")
        if not segment_id and payload.get("event_id"):
            segment = self.store.segments.get_event_segment(payload["event_id"])
            segment_id = segment["id"] if segment else None
        with self.conn:
            if segment_id:
                self.store.persistent.promote_segment(segment_id, now=now)
            else:
                for segment in self.store.segments.promotable_segments():
                    self.store.persistent.promote_segment(segment["id"], now=now)

    def _entity_ids_from_payload(self, payload: dict) -> list[str]:
        entity_ids = payload.get("entity_ids")
        if isinstance(entity_ids, list):
            return [str(entity_id) for entity_id in entity_ids]
        if payload.get("entity_id"):
            return [str(payload["entity_id"])]
        if payload.get("event_id"):
            rows = self.conn.execute(
                "SELECT entity_id FROM event_entities WHERE event_id = ?",
                (payload["event_id"],),
            ).fetchall()
            return [row["entity_id"] for row in rows]
        return []

    def _enqueue_lancedb_retry(self, event_id: str) -> None:
        now = utc_now()
        with self.conn:
            self.conn.execute(
                """
                INSERT OR IGNORE INTO index_jobs(id, idempotency_key, job_type, payload_json, status, scheduled_at, created_at)
                VALUES (?, ?, 'upsert_lancedb', ?, 'pending', ?, ?)
                """,
                (
                    new_id("job"),
                    f"upsert_lancedb:{event_id}",
                    _json_dumps({"event_id": event_id}),
                    now,
                    now,
                ),
            )

    def _complete(self, job_id: str, *, status: str = "completed") -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE index_jobs SET status = ?, completed_at = ? WHERE id = ?",
                (status, utc_now(), job_id),
            )

    def _fail(self, job: sqlite3.Row, exc: Exception) -> None:
        retry_count = int(job["retry_count"]) + 1
        status = "failed" if retry_count >= int(job["max_retries"]) else "pending"
        error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        with self.conn:
            self.conn.execute(
                """
                UPDATE index_jobs
                SET status = ?, retry_count = ?, last_error = ?, completed_at = CASE WHEN ? = 'failed' THEN ? ELSE completed_at END
                WHERE id = ?
                """,
                (status, retry_count, error, status, utc_now(), job["id"]),
            )
