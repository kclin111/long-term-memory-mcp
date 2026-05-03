from __future__ import annotations

import json
import re
import sqlite3
import unicodedata
import warnings
from uuid import uuid4

from .config import Settings
from .db import connect, rows_to_dicts
from .lifecycle import (
    ConflictDetector,
    EntityNarrativeCache,
    PersistentMemoryStore,
    SegmentManager,
)
from .migrations import MIGRATIONS, apply_migrations
from .scoring import score_importance, score_novelty
from .time import utc_now


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def normalize_entity_name(name: str) -> str:
    return unicodedata.normalize("NFC", name).strip().lower()


def _json_dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _fts_query(query: str) -> str:
    terms = re.findall(r"[\w\u4e00-\u9fff]+", query, flags=re.UNICODE)
    if not terms:
        return ""
    return " OR ".join(f'"{term}"' for term in terms[:12])


class MemoryStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.settings.ensure_dirs()
        self.conn = connect(self.settings.sqlite_path, self.settings.sqlite_journal_mode)
        self.segments = SegmentManager(self.conn)
        self.persistent = PersistentMemoryStore(self.conn)
        self.narratives = EntityNarrativeCache(self.conn)
        self.conflicts = ConflictDetector(self.conn)
        self.embedding_signature_change_detected = False

    @classmethod
    def open(cls, settings: Settings | None = None) -> "MemoryStore":
        return cls(settings or Settings.from_env())

    def close(self) -> None:
        self.conn.close()

    def init(self) -> None:
        apply_migrations(self.conn)
        now = utc_now()
        defaults = {
            "schema_version": MIGRATIONS[-1].version,
            "daily_llm_cost_usd": 0,
            "embedding_model_signature": None,
        }
        with self.conn:
            for key, value in defaults.items():
                self.conn.execute(
                    "INSERT OR IGNORE INTO settings(key, value_json, updated_at) VALUES (?, ?, ?)",
                    (key, _json_dumps(value), now),
                )
            self.conn.execute(
                "UPDATE settings SET value_json = ?, updated_at = ? WHERE key = 'schema_version'",
                (_json_dumps(MIGRATIONS[-1].version), now),
            )
            self.embedding_signature_change_detected = self._check_embedding_signature(now=now)

    def _check_embedding_signature(self, *, now: str) -> bool:
        current = self.settings.embedding_signature
        row = self.conn.execute(
            "SELECT value_json FROM settings WHERE key = 'embedding_model_signature'"
        ).fetchone()
        stored = json.loads(row["value_json"]) if row else None
        if not stored:
            self.conn.execute(
                "UPDATE settings SET value_json = ?, updated_at = ? WHERE key = 'embedding_model_signature'",
                (_json_dumps(current), now),
            )
            return False
        if stored != current:
            warnings.warn(
                "Embedding model signature changed; run rebuild_index before trusting LanceDB recall "
                f"(stored={stored!r}, current={current!r})",
                RuntimeWarning,
                stacklevel=2,
            )
            return True
        return False

    def ingest_observation(
        self,
        *,
        source_type: str,
        content: str,
        session_id: str | None = None,
        source_uri: str | None = None,
        client_dedup_key: str | None = None,
        metadata: dict | None = None,
        role: str = "user",
    ) -> dict:
        self.init()
        now = utc_now()
        metadata_json = _json_dumps(metadata or {})

        if client_dedup_key:
            existing = self.conn.execute(
                """
                SELECT id FROM observations
                WHERE source_type = ? AND session_id IS ? AND client_dedup_key = ?
                """,
                (source_type, session_id, client_dedup_key),
            ).fetchone()
            if existing:
                return {
                    "observation_id": existing["id"],
                    "session_page_id": None,
                    "importance": None,
                    "novelty": None,
                    "queued_jobs": [],
                    "status": "deduplicated",
                }

        observation_id = new_id("obs")
        page_id = new_id("page") if session_id and source_type == "chat" else None
        importance = score_importance(content)
        novelty = score_novelty(content)
        queued_jobs: list[str] = []

        with self.conn:
            self.conn.execute(
                """
                INSERT INTO observations(
                  id, client_dedup_key, source_type, source_uri, content, metadata_json,
                  observed_at, session_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    observation_id,
                    client_dedup_key,
                    source_type,
                    source_uri,
                    content,
                    metadata_json,
                    now,
                    session_id,
                    now,
                ),
            )
            if page_id and session_id:
                turn = self._next_turn_index(session_id)
                self.conn.execute(
                    """
                    INSERT INTO session_pages(
                      id, observation_id, session_id, role, content, timestamp,
                      turn_index, metadata_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (page_id, observation_id, session_id, role, content, now, turn, metadata_json, now),
                )
                self._insert_dialogue_chain(page_id=page_id, session_id=session_id, turn_index=turn)

            if (
                importance >= self.settings.importance_promote_threshold
                and novelty >= self.settings.novelty_promote_threshold
            ):
                job_id = new_id("job")
                queued_jobs.append(job_id)
                self.conn.execute(
                    """
                    INSERT INTO index_jobs(
                      id, idempotency_key, job_type, payload_json, status,
                      scheduled_at, created_at
                    ) VALUES (?, ?, ?, ?, 'pending', ?, ?)
                    """,
                    (
                        job_id,
                        f"extract_event:{observation_id}",
                        "extract_event",
                        _json_dumps({"observation_id": observation_id}),
                        now,
                        now,
                    ),
                )

        return {
            "observation_id": observation_id,
            "session_page_id": page_id,
            "importance": importance,
            "novelty": novelty,
            "queued_jobs": queued_jobs,
            "status": "queued" if queued_jobs else "stored",
        }

    def recall(
        self,
        query: str,
        *,
        session_id: str | None = None,
        limit: int = 10,
        limits: dict | None = None,
        scope: str = "auto",
        include_session_buffer: bool = True,
        max_response_tokens: int | None = None,
    ) -> dict:
        self.init()
        limits = limits or {}
        max_observations = int(limits.get("observations", limit))
        max_events = int(limits.get("events", limit))
        max_open_questions = int(limits.get("open_questions", 5))
        max_session_pages = int(limits.get("session_pages", 10))
        current_session_only = scope == "current_session"
        scoped_session_id = session_id if current_session_only and session_id else None

        fts = _fts_query(query)
        if current_session_only and not session_id:
            rows = []
        elif fts:
            try:
                if scoped_session_id:
                    rows = self.conn.execute(
                        """
                        SELECT o.id, o.source_type, o.session_id, o.content, o.observed_at
                        FROM observations_fts f
                        JOIN observations o ON o.id = f.observation_id
                        WHERE observations_fts MATCH ? AND o.session_id = ? AND o.status != 'forgotten'
                        ORDER BY rank
                        LIMIT ?
                        """,
                        (fts, scoped_session_id, max_observations),
                    ).fetchall()
                else:
                    rows = self.conn.execute(
                        """
                        SELECT o.id, o.source_type, o.session_id, o.content, o.observed_at
                        FROM observations_fts f
                        JOIN observations o ON o.id = f.observation_id
                        WHERE observations_fts MATCH ? AND o.status != 'forgotten'
                        ORDER BY rank
                        LIMIT ?
                        """,
                        (fts, max_observations),
                    ).fetchall()
            except sqlite3.OperationalError:
                rows = self._like_recall(query, max_observations, session_id=scoped_session_id)
        else:
            rows = self._like_recall(query, max_observations, session_id=scoped_session_id)

        session_rows = []
        if include_session_buffer and session_id:
            session_rows = self.conn.execute(
                """
                SELECT id, role, content, timestamp, turn_index
                FROM session_pages
                WHERE session_id = ?
                ORDER BY turn_index DESC
                LIMIT ?
                """,
                (session_id, max_session_pages),
            ).fetchall()

        event_rows = [] if current_session_only and not session_id else self._event_recall(
            query,
            max_events,
            session_id=scoped_session_id,
        )
        events = self._enrich_events(event_rows)
        open_questions = self._open_question_recall(query, max_open_questions)
        max_narratives = int(limits.get("narratives", 5))
        max_persistent = int(limits.get("persistent_memories", 5))

        event_ids = [event["id"] for event in events]
        now = utc_now()
        with self.conn:
            self.segments.visit_segments_for_events(event_ids, now=now)

        primary_entities = self._primary_entities_for_events(events, limit=max_narratives)
        entity_narratives = self._build_query_narratives(
            query=query,
            entity_ids=primary_entities,
            now=now,
        )
        persistent_memories = self._persistent_memories_for_entities(
            entity_ids=primary_entities,
            event_ids=event_ids,
            limit=max_persistent,
        )
        conflicts = self._conflicts_for_entities(primary_entities, limit=max_persistent)

        return {
            "query": query,
            "scope": scope,
            "limits": {
                "observations": max_observations,
                "events": max_events,
                "open_questions": max_open_questions,
                "session_pages": max_session_pages,
                "narratives": max_narratives,
                "persistent_memories": max_persistent,
            },
            "max_response_tokens": max_response_tokens or self.settings.recall_max_response_tokens,
            "session_context": rows_to_dicts(reversed(session_rows)),
            "observations": rows_to_dicts(rows),
            "events": events,
            "entity_narratives": entity_narratives,
            "persistent_memories": persistent_memories,
            "open_questions": rows_to_dicts(open_questions),
            "memory_conflicts": conflicts,
            "evidence": [],
            "confidence": 0.0,
        }

    def merge_entities(
        self,
        *,
        source_id: str,
        target_id: str,
        notes: str | None = None,
    ) -> dict:
        """Merge ``source`` entity into ``target``.

        All foreign references are rewritten to point at ``target``. The
        ``source`` row is deleted. Aliases from ``source`` (canonical name plus
        existing aliases) are appended to ``target.aliases_json`` so that future
        entity resolution can hit the merged record by either name.

        Returns a summary of the moved rows so callers can audit the merge.
        """

        self.init()
        if source_id == target_id:
            raise ValueError("source_id and target_id must differ")

        source = self.conn.execute(
            "SELECT id, entity_type, canonical_name, aliases_json FROM entities WHERE id = ?",
            (source_id,),
        ).fetchone()
        target = self.conn.execute(
            "SELECT id, entity_type, canonical_name, aliases_json FROM entities WHERE id = ?",
            (target_id,),
        ).fetchone()
        if not source:
            raise ValueError(f"source entity not found: {source_id}")
        if not target:
            raise ValueError(f"target entity not found: {target_id}")
        if source["entity_type"] != target["entity_type"]:
            raise ValueError(
                "entity_type mismatch: "
                f"source={source['entity_type']} target={target['entity_type']}"
            )

        now = utc_now()
        moved: dict[str, int] = {}
        with self.conn:
            moved["event_entities"] = self._merge_pk_table(
                table="event_entities",
                key_columns=("event_id", "role_in_event"),
                source_id=source_id,
                target_id=target_id,
                entity_column="entity_id",
            )
            moved["entity_anchors"] = self._merge_pk_table(
                table="entity_anchors",
                key_columns=("anchor_id", "relation"),
                source_id=source_id,
                target_id=target_id,
                entity_column="entity_id",
            )
            moved["entity_roles"] = self.conn.execute(
                "UPDATE entity_roles SET entity_id = ?, updated_at = ? WHERE entity_id = ?",
                (target_id, now, source_id),
            ).rowcount
            self.conn.execute(
                "UPDATE entity_roles SET scope_entity_id = ?, updated_at = ? WHERE scope_entity_id = ?",
                (target_id, now, source_id),
            )
            moved["entity_states"] = self.conn.execute(
                "UPDATE entity_states SET entity_id = ?, updated_at = ? WHERE entity_id = ?",
                (target_id, now, source_id),
            ).rowcount
            self.conn.execute(
                "UPDATE entity_states SET scope_entity_id = ?, updated_at = ? WHERE scope_entity_id = ?",
                (target_id, now, source_id),
            )
            moved["event_actions_actor"] = self.conn.execute(
                "UPDATE event_actions SET actor_entity_id = ? WHERE actor_entity_id = ?",
                (target_id, source_id),
            ).rowcount
            moved["event_actions_object"] = self.conn.execute(
                "UPDATE event_actions SET object_entity_id = ? WHERE object_entity_id = ?",
                (target_id, source_id),
            ).rowcount
            moved["open_questions"] = self._rewrite_open_questions(
                source_id=source_id,
                target_id=target_id,
                now=now,
            )

            new_aliases = self._merge_aliases(source, target)
            self.conn.execute(
                """
                UPDATE entities
                SET aliases_json = ?, last_seen_at = ?
                WHERE id = ?
                """,
                (_json_dumps(new_aliases), now, target_id),
            )
            self.conn.execute("DELETE FROM entities WHERE id = ?", (source_id,))

            self.conn.execute(
                """
                INSERT INTO entity_merges(
                  id, source_entity_id, target_entity_id,
                  source_canonical_name, source_entity_type,
                  moved_counts_json, notes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    new_id("merge"),
                    source_id,
                    target_id,
                    source["canonical_name"],
                    source["entity_type"],
                    _json_dumps(moved),
                    notes,
                    now,
                ),
            )

        return {
            "status": "merged",
            "source_id": source_id,
            "target_id": target_id,
            "target_entity_type": target["entity_type"],
            "target_canonical_name": target["canonical_name"],
            "aliases": new_aliases,
            "moved": moved,
        }

    def _merge_pk_table(
        self,
        *,
        table: str,
        key_columns: tuple[str, ...],
        entity_column: str,
        source_id: str,
        target_id: str,
    ) -> int:
        join_predicate = " AND ".join(
            f"e2.{column} = {table}.{column}" for column in key_columns
        )
        moved_cursor = self.conn.execute(
            f"""
            UPDATE {table}
            SET {entity_column} = ?
            WHERE {entity_column} = ?
              AND NOT EXISTS (
                SELECT 1 FROM {table} e2
                WHERE e2.{entity_column} = ?
                  AND {join_predicate}
              )
            """,
            (target_id, source_id, target_id),
        )
        moved = moved_cursor.rowcount
        self.conn.execute(
            f"DELETE FROM {table} WHERE {entity_column} = ?",
            (source_id,),
        )
        return moved

    def _rewrite_open_questions(self, *, source_id: str, target_id: str, now: str) -> int:
        rows = self.conn.execute(
            """
            SELECT id, subject_entity_id, related_entity_ids_json, question_type, dedup_key, status
            FROM open_questions
            WHERE subject_entity_id = ? OR related_entity_ids_json LIKE ?
            """,
            (source_id, f"%{source_id}%"),
        ).fetchall()
        affected = 0
        for row in rows:
            related = json.loads(row["related_entity_ids_json"])
            new_related = sorted({target_id if rid == source_id else rid for rid in related})
            new_subject = target_id if row["subject_entity_id"] == source_id else row["subject_entity_id"]
            if new_subject == row["subject_entity_id"] and new_related == sorted(set(related)):
                continue
            new_dedup = self._open_question_dedup_key(new_subject, row["question_type"], new_related)
            existing = self.conn.execute(
                "SELECT id FROM open_questions WHERE dedup_key = ? AND id != ?",
                (new_dedup, row["id"]),
            ).fetchone()
            if existing:
                self.conn.execute(
                    """
                    UPDATE open_questions
                    SET status = 'obsolete', updated_at = ?
                    WHERE id = ?
                    """,
                    (now, row["id"]),
                )
            else:
                self.conn.execute(
                    """
                    UPDATE open_questions
                    SET subject_entity_id = ?,
                        related_entity_ids_json = ?,
                        dedup_key = ?,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (new_subject, _json_dumps(new_related), new_dedup, now, row["id"]),
                )
            affected += 1
        return affected

    @staticmethod
    def _open_question_dedup_key(
        subject_entity_id: str | None,
        question_type: str,
        related_ids: list[str],
    ) -> str:
        subject = subject_entity_id or "none"
        related = ",".join(sorted(set(related_ids)))
        return f"{subject}|{question_type}|{related}"

    @staticmethod
    def _merge_aliases(source, target) -> list[str]:
        source_aliases = json.loads(source["aliases_json"] or "[]")
        target_aliases = json.loads(target["aliases_json"] or "[]")
        all_aliases = list(target_aliases) + list(source_aliases) + [source["canonical_name"]]
        target_canonical = target["canonical_name"]
        seen: set[str] = set()
        result: list[str] = []
        for alias in all_aliases:
            if not alias:
                continue
            if alias == target_canonical:
                continue
            if alias in seen:
                continue
            seen.add(alias)
            result.append(alias)
        return result

    def get_entity_timeline(
        self,
        entity_query: str,
        *,
        since: str | None = None,
        until: str | None = None,
        event_types: list[str] | None = None,
        state_types: list[str] | None = None,
        limit: int = 50,
    ) -> dict:
        self.init()
        entity = self._find_entity(entity_query)
        if not entity:
            return {
                "entity": None,
                "events": [],
                "states": [],
                "roles": [],
                "anchors": [],
                "open_questions": [],
            }

        event_rows = self._entity_event_rows(
            entity["id"],
            since=since,
            until=until,
            event_types=event_types,
            limit=limit,
        )
        states = self._entity_state_rows(
            entity["id"],
            since=since,
            until=until,
            state_types=state_types,
            limit=limit,
        )
        roles = self.conn.execute(
            """
            SELECT id, role_type, confidence, valid_from, valid_to, status
            FROM entity_roles
            WHERE entity_id = ?
            ORDER BY valid_from DESC
            LIMIT ?
            """,
            (entity["id"], limit),
        ).fetchall()
        anchors = self.conn.execute(
            """
            SELECT a.id, a.anchor_type, a.label, a.normalized_value, a.granularity,
                   ea.relation, ea.confidence
            FROM entity_anchors ea
            JOIN spatiotemporal_anchors a ON a.id = ea.anchor_id
            WHERE ea.entity_id = ?
            ORDER BY ea.created_at DESC
            LIMIT ?
            """,
            (entity["id"], limit),
        ).fetchall()
        questions = self.conn.execute(
            """
            SELECT id, question_type, question_text, status, priority, answer_text, confidence, created_at, answered_at
            FROM open_questions
            WHERE subject_entity_id = ? OR related_entity_ids_json LIKE ?
            ORDER BY status, priority DESC, created_at DESC
            LIMIT ?
            """,
            (entity["id"], f"%{entity['id']}%", limit),
        ).fetchall()

        now = utc_now()
        with self.conn:
            narrative = self.narratives.get_or_generate(
                entity_id=entity["id"],
                scope="global",
                now=now,
            )
        persistent = self._persistent_memories_for_entities(
            entity_ids=[entity["id"]],
            event_ids=[],
            limit=limit,
        )
        conflicts = self._conflicts_for_entities([entity["id"]], limit=limit)

        return {
            "entity": dict(entity),
            "events": self._enrich_events(event_rows),
            "states": rows_to_dicts(states),
            "roles": rows_to_dicts(roles),
            "anchors": rows_to_dicts(anchors),
            "open_questions": rows_to_dicts(questions),
            "narrative": narrative,
            "persistent_memories": persistent,
            "memory_conflicts": conflicts,
        }

    def status(self) -> dict:
        self.init()
        counts = {}
        for table in [
            "observations",
            "session_pages",
            "events",
            "entities",
            "entity_roles",
            "entity_states",
            "event_actions",
            "spatiotemporal_anchors",
            "open_questions",
            "memory_segments",
            "persistent_memories",
            "entity_narratives",
            "memory_conflicts",
            "index_jobs",
        ]:
            counts[table] = self.conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
        pending_jobs = self.conn.execute(
            "SELECT COUNT(*) AS n FROM index_jobs WHERE status = 'pending'"
        ).fetchone()["n"]
        open_conflicts = self.conn.execute(
            "SELECT COUNT(*) AS n FROM memory_conflicts WHERE status = 'open'"
        ).fetchone()["n"]
        signature_row = self.conn.execute(
            "SELECT value_json FROM settings WHERE key = 'embedding_model_signature'"
        ).fetchone()
        stored_signature = json.loads(signature_row["value_json"]) if signature_row else None
        current_signature = self.settings.embedding_signature
        signature_changed = bool(stored_signature and stored_signature != current_signature)
        return {
            "sqlite_path": str(self.settings.sqlite_path),
            "lancedb_path": str(self.settings.lancedb_path),
            "counts": counts,
            "pending_jobs": pending_jobs,
            "open_conflicts": open_conflicts,
            "embedding_model_signature": stored_signature,
            "current_embedding_signature": current_signature,
            "embedding_signature_change_detected": signature_changed,
        }

    def get_open_questions(
        self,
        *,
        entity_id: str | None = None,
        status: str | None = None,
        priority: int | None = None,
        limit: int = 50,
    ) -> list[dict]:
        self.init()
        clauses: list[str] = []
        params: list[object] = []
        if status:
            clauses.append("status = ?")
            params.append(status)
        if priority is not None:
            clauses.append("priority = ?")
            params.append(priority)
        if entity_id:
            clauses.append("(subject_entity_id = ? OR related_entity_ids_json LIKE ?)")
            params.extend([entity_id, f"%{entity_id}%"])
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        rows = self.conn.execute(
            f"""
            SELECT id, subject_entity_id, question_type, question_text, related_entity_ids_json,
                   status, priority, answer_event_id, answer_text, confidence,
                   created_at, updated_at, answered_at
            FROM open_questions
            {where}
            ORDER BY status, priority DESC, created_at DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return rows_to_dicts(rows)

    def get_conflicts(
        self,
        *,
        entity_id: str | None = None,
        status: str | None = "open",
        conflict_type: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        self.init()
        clauses: list[str] = []
        params: list[object] = []
        if status:
            clauses.append("status = ?")
            params.append(status)
        if entity_id:
            clauses.append("entity_id = ?")
            params.append(entity_id)
        if conflict_type:
            clauses.append("conflict_type = ?")
            params.append(conflict_type)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        rows = self.conn.execute(
            f"""
            SELECT id, conflict_type, entity_id, new_record_type, new_record_id,
                   old_record_type, old_record_id, description, severity, status,
                   created_at, resolved_at
            FROM memory_conflicts
            {where}
            ORDER BY status, severity DESC, created_at DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return rows_to_dicts(rows)

    FORGETTABLE_TYPES: tuple[str, ...] = (
        "event",
        "observation",
        "persistent_memory",
        "memory_segment",
        "open_question",
    )

    def forget(
        self,
        *,
        target_type: str,
        target_id: str,
        mode: str = "soft",
        cascade: bool = True,
    ) -> dict:
        """Soft-forget or hard-delete a record with PRD §8.6 cascade rules.

        ``mode='soft'`` flips the record's status/lifecycle to ``'forgotten'``
        without touching foreign references; default recall queries can then
        skip it. ``mode='hard'`` deletes the SQLite row (FK ON DELETE CASCADE
        handles the dependent edges) and enqueues a ``delete_lancedb`` job so
        the retrieval index can drop the same id asynchronously.

        Cascade for events: dependent ``entity_narratives`` are marked stale
        and ``persistent_memories`` are recomputed (supporting event removed
        from the supporting list; if no supporting events remain, the memory
        is also marked forgotten). Cascade for observations: dependent
        ``evidence`` rows are marked stale, and the originating event (if
        any) is forgotten with the same mode unless ``cascade=False``.
        """

        self.init()
        if target_type not in self.FORGETTABLE_TYPES:
            raise ValueError(f"unsupported target_type: {target_type}")
        if mode not in ("soft", "hard"):
            raise ValueError(f"mode must be 'soft' or 'hard', got {mode!r}")

        now = utc_now()
        report: dict = {
            "target_type": target_type,
            "target_id": target_id,
            "mode": mode,
            "cascade": cascade,
            "actions": [],
            "now": now,
        }
        with self.conn:
            if target_type == "event":
                self._forget_event(target_id, mode=mode, cascade=cascade, now=now, report=report)
            elif target_type == "observation":
                self._forget_observation(target_id, mode=mode, cascade=cascade, now=now, report=report)
            elif target_type == "persistent_memory":
                self._forget_persistent_memory(target_id, mode=mode, now=now, report=report)
            elif target_type == "memory_segment":
                self._forget_segment(target_id, mode=mode, now=now, report=report)
            elif target_type == "open_question":
                self._forget_open_question(target_id, mode=mode, now=now, report=report)
        return report

    def _forget_event(
        self,
        event_id: str,
        *,
        mode: str,
        cascade: bool,
        now: str,
        report: dict,
    ) -> None:
        row = self.conn.execute(
            "SELECT id, status FROM events WHERE id = ?",
            (event_id,),
        ).fetchone()
        if not row:
            report["actions"].append({"action": "missing_event", "event_id": event_id})
            return

        affected_entities = [
            r["entity_id"]
            for r in self.conn.execute(
                "SELECT entity_id FROM event_entities WHERE event_id = ?",
                (event_id,),
            ).fetchall()
        ]

        if cascade:
            stale_narratives = self.conn.execute(
                """
                UPDATE entity_narratives
                SET status = 'stale'
                WHERE entity_id IN (SELECT entity_id FROM event_entities WHERE event_id = ?)
                  AND status = 'fresh'
                """,
                (event_id,),
            ).rowcount
            report["actions"].append({"action": "narratives_marked_stale", "count": stale_narratives})

            self._recompute_persistent_memories_after_event_drop(event_id=event_id, now=now, report=report)

            self.conn.execute(
                "UPDATE memory_segments SET updated_at = ? "
                "WHERE id IN (SELECT segment_id FROM segment_events WHERE event_id = ?)",
                (now, event_id),
            )

        if mode == "soft":
            self.conn.execute(
                "UPDATE events SET status = 'forgotten', updated_at = ? WHERE id = ?",
                (now, event_id),
            )
            report["actions"].append({"action": "event_marked_forgotten", "event_id": event_id})
        else:
            self.conn.execute("DELETE FROM events WHERE id = ?", (event_id,))
            report["actions"].append({"action": "event_deleted", "event_id": event_id})
            self._enqueue_delete_lancedb(record_type="event", sqlite_id=event_id, now=now)

        report["affected_entity_ids"] = affected_entities

    def _forget_observation(
        self,
        observation_id: str,
        *,
        mode: str,
        cascade: bool,
        now: str,
        report: dict,
    ) -> None:
        row = self.conn.execute(
            "SELECT id FROM observations WHERE id = ?",
            (observation_id,),
        ).fetchone()
        if not row:
            report["actions"].append({"action": "missing_observation", "observation_id": observation_id})
            return

        if cascade:
            stale_evidence = self.conn.execute(
                "UPDATE evidence SET status = 'stale', updated_at = ? "
                "WHERE observation_id = ? AND status = 'active'",
                (now, observation_id),
            ).rowcount
            report["actions"].append({"action": "evidence_marked_stale", "count": stale_evidence})

            descendant_event_rows = self.conn.execute(
                "SELECT DISTINCT event_id FROM evidence "
                "WHERE observation_id = ? AND event_id IS NOT NULL",
                (observation_id,),
            ).fetchall()
            for descendant in descendant_event_rows:
                self._forget_event(
                    descendant["event_id"],
                    mode=mode,
                    cascade=True,
                    now=now,
                    report=report,
                )

        if mode == "soft":
            self.conn.execute(
                "UPDATE observations SET status = 'forgotten', updated_at = ? WHERE id = ?",
                (now, observation_id),
            )
            report["actions"].append({"action": "observation_marked_forgotten", "observation_id": observation_id})
        else:
            self.conn.execute("DELETE FROM observations WHERE id = ?", (observation_id,))
            report["actions"].append({"action": "observation_deleted", "observation_id": observation_id})
            self._enqueue_delete_lancedb(record_type="observation", sqlite_id=observation_id, now=now)

    def _forget_persistent_memory(
        self,
        pm_id: str,
        *,
        mode: str,
        now: str,
        report: dict,
    ) -> None:
        row = self.conn.execute(
            "SELECT id FROM persistent_memories WHERE id = ?",
            (pm_id,),
        ).fetchone()
        if not row:
            report["actions"].append({"action": "missing_persistent_memory", "id": pm_id})
            return
        if mode == "soft":
            self.conn.execute(
                "UPDATE persistent_memories SET status = 'forgotten', updated_at = ? WHERE id = ?",
                (now, pm_id),
            )
            report["actions"].append({"action": "persistent_memory_marked_forgotten", "id": pm_id})
        else:
            self.conn.execute("DELETE FROM persistent_memories WHERE id = ?", (pm_id,))
            report["actions"].append({"action": "persistent_memory_deleted", "id": pm_id})
            self._enqueue_delete_lancedb(record_type="persistent_memory", sqlite_id=pm_id, now=now)

    def _forget_segment(
        self,
        segment_id: str,
        *,
        mode: str,
        now: str,
        report: dict,
    ) -> None:
        row = self.conn.execute(
            "SELECT id FROM memory_segments WHERE id = ?",
            (segment_id,),
        ).fetchone()
        if not row:
            report["actions"].append({"action": "missing_segment", "id": segment_id})
            return
        if mode == "soft":
            self.conn.execute(
                "UPDATE memory_segments SET lifecycle = 'forgotten', updated_at = ? WHERE id = ?",
                (now, segment_id),
            )
            report["actions"].append({"action": "segment_marked_forgotten", "id": segment_id})
        else:
            self.conn.execute("DELETE FROM memory_segments WHERE id = ?", (segment_id,))
            report["actions"].append({"action": "segment_deleted", "id": segment_id})
            self._enqueue_delete_lancedb(record_type="memory_segment", sqlite_id=segment_id, now=now)

    def _forget_open_question(
        self,
        question_id: str,
        *,
        mode: str,
        now: str,
        report: dict,
    ) -> None:
        row = self.conn.execute(
            "SELECT id FROM open_questions WHERE id = ?",
            (question_id,),
        ).fetchone()
        if not row:
            report["actions"].append({"action": "missing_open_question", "id": question_id})
            return
        if mode == "soft":
            self.conn.execute(
                "UPDATE open_questions SET status = 'forgotten', updated_at = ? WHERE id = ?",
                (now, question_id),
            )
            report["actions"].append({"action": "open_question_marked_forgotten", "id": question_id})
        else:
            self.conn.execute("DELETE FROM open_questions WHERE id = ?", (question_id,))
            report["actions"].append({"action": "open_question_deleted", "id": question_id})
            self._enqueue_delete_lancedb(record_type="open_question", sqlite_id=question_id, now=now)

    def _recompute_persistent_memories_after_event_drop(
        self,
        *,
        event_id: str,
        now: str,
        report: dict,
    ) -> None:
        rows = self.conn.execute(
            """
            SELECT id, supporting_event_ids_json, reinforcement_count
            FROM persistent_memories
            WHERE status = 'active' AND supporting_event_ids_json LIKE ?
            """,
            (f"%\"{event_id}\"%",),
        ).fetchall()
        affected = 0
        forgotten = 0
        for row in rows:
            supporting = [eid for eid in json.loads(row["supporting_event_ids_json"]) if eid != event_id]
            if not supporting:
                self.conn.execute(
                    "UPDATE persistent_memories SET status = 'forgotten', updated_at = ?, "
                    "supporting_event_ids_json = '[]' WHERE id = ?",
                    (now, row["id"]),
                )
                forgotten += 1
                continue
            new_count = max(1, len(supporting))
            new_stability = min(1.0, round(0.4 + 0.1 * new_count, 4))
            self.conn.execute(
                """
                UPDATE persistent_memories
                SET supporting_event_ids_json = ?,
                    reinforcement_count = ?,
                    stability = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    _json_dumps(supporting),
                    new_count,
                    new_stability,
                    now,
                    row["id"],
                ),
            )
            affected += 1
        if affected or forgotten:
            report["actions"].append(
                {
                    "action": "persistent_memories_recomputed",
                    "recomputed": affected,
                    "forgotten": forgotten,
                }
            )

    def _enqueue_delete_lancedb(self, *, record_type: str, sqlite_id: str, now: str) -> None:
        self.conn.execute(
            """
            INSERT OR IGNORE INTO index_jobs(
              id, idempotency_key, job_type, payload_json, status, scheduled_at, created_at
            ) VALUES (?, ?, 'delete_lancedb', ?, 'pending', ?, ?)
            """,
            (
                new_id("job"),
                f"delete_lancedb:{record_type}:{sqlite_id}",
                _json_dumps({"record_type": record_type, "sqlite_id": sqlite_id}),
                now,
                now,
            ),
        )

    def search_events(
        self,
        *,
        query: str | None = None,
        entity_id: str | None = None,
        event_type: str | None = None,
        since: str | None = None,
        until: str | None = None,
        min_importance: float | None = None,
        include_forgotten: bool = False,
        limit: int = 25,
    ) -> list[dict]:
        """Search events by free text + structured filters (PRD §8.3)."""

        self.init()
        clauses: list[str] = []
        params: list[object] = []

        if not include_forgotten:
            clauses.append("e.status != 'forgotten'")

        if event_type:
            clauses.append("e.event_type = ?")
            params.append(event_type)
        if since:
            clauses.append("e.observed_at >= ?")
            params.append(since)
        if until:
            clauses.append("e.observed_at <= ?")
            params.append(until)
        if min_importance is not None:
            clauses.append("e.importance >= ?")
            params.append(float(min_importance))

        if query:
            terms = re.findall(r"[\w\u4e00-\u9fff]+", query, flags=re.UNICODE)[:12] or [query]
            placeholders = " OR ".join("e.summary LIKE ?" for _ in terms)
            clauses.append(f"({placeholders})")
            params.extend(f"%{term}%" for term in terms)

        join = ""
        if entity_id:
            join = "JOIN event_entities ee ON ee.event_id = e.id"
            clauses.append("ee.entity_id = ?")
            params.append(entity_id)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        rows = self.conn.execute(
            f"""
            SELECT DISTINCT e.id, e.event_type, e.summary, e.observed_at, e.importance,
                   e.confidence, e.status
            FROM events e
            {join}
            {where}
            ORDER BY e.importance DESC, e.observed_at DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return self._enrich_events(rows)

    def consolidate_memory(
        self,
        *,
        tasks: list[str] | None = None,
        scope: str = "auto",
        entity_id: str | None = None,
        segment_id: str | None = None,
    ) -> dict:
        """Run the safe MemoryOS-style consolidation tasks (PRD §8.5)."""

        self.init()
        valid_tasks = {
            "heat_update",
            "persistent_memory",
            "narrative_refresh",
            "open_question_resolution",
            "archive",
        }
        chosen = set(tasks or valid_tasks)
        unknown = chosen - valid_tasks
        if unknown:
            raise ValueError(f"unknown consolidation tasks: {sorted(unknown)}")

        now = utc_now()
        report: dict = {"tasks_run": sorted(chosen), "scope": scope}

        if "heat_update" in chosen:
            report["heat_update"] = self._consolidate_heat_update(now=now, segment_id=segment_id)

        if "persistent_memory" in chosen:
            report["persistent_memory"] = self._consolidate_persistent_memory(now=now, segment_id=segment_id)

        if "narrative_refresh" in chosen:
            report["narrative_refresh"] = self._consolidate_narratives(
                now=now, entity_id=entity_id
            )

        if "open_question_resolution" in chosen:
            report["open_question_resolution"] = self._consolidate_open_questions(now=now)

        if "archive" in chosen:
            report["archive"] = self._consolidate_archive(now=now)

        return report

    def flush_session_buffer(self, session_id: str) -> dict:
        """Mark the session's pages as flushed and return a count.

        SessionBuffer is already mirrored into ``session_pages`` on write,
        so the SQL state is durable. This entry point exists so that
        ``flush_session_buffer`` can be used as a deterministic
        synchronization point by callers (PRD §8.11).
        """

        self.init()
        rows = self.conn.execute(
            "SELECT COUNT(*) AS n FROM session_pages WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return {
            "session_id": session_id,
            "session_page_count": int(rows["n"]) if rows else 0,
            "flushed_at": utc_now(),
        }

    def _primary_entities_for_events(self, events: list[dict], *, limit: int) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for event in events:
            for entity in event.get("entities", []) or []:
                eid = entity.get("id")
                if not eid or eid in seen:
                    continue
                if entity.get("canonical_name") == "user":
                    continue
                seen.add(eid)
                ordered.append(eid)
                if len(ordered) >= limit:
                    return ordered
        if len(ordered) < limit:
            for event in events:
                for entity in event.get("entities", []) or []:
                    eid = entity.get("id")
                    if eid and eid not in seen:
                        seen.add(eid)
                        ordered.append(eid)
                        if len(ordered) >= limit:
                            return ordered
        return ordered

    def _build_query_narratives(self, *, query: str, entity_ids: list[str], now: str) -> list[dict]:
        narratives: list[dict] = []
        with self.conn:
            for entity_id in entity_ids:
                narrative = self.narratives.get_or_generate(
                    entity_id=entity_id,
                    scope="query_specific",
                    query=query,
                    now=now,
                )
                if narrative:
                    narratives.append(narrative)
        return narratives

    def _persistent_memories_for_entities(
        self,
        *,
        entity_ids: list[str],
        event_ids: list[str],
        limit: int,
    ) -> list[dict]:
        if not entity_ids and not event_ids:
            return []
        rows: list[sqlite3.Row] = []
        if entity_ids:
            seen_ids: set[str] = set()
            for entity_id in entity_ids:
                pattern = f"%\"{entity_id}\"%"
                for row in self.conn.execute(
                    """
                    SELECT id, memory_type, content, entity_ids_json, supporting_event_ids_json,
                           confidence, stability, status, reinforcement_count, last_reinforced_at,
                           created_at, updated_at
                    FROM persistent_memories
                    WHERE status = 'active' AND entity_ids_json LIKE ?
                    ORDER BY stability DESC, last_reinforced_at DESC
                    LIMIT ?
                    """,
                    (pattern, limit),
                ).fetchall():
                    if row["id"] in seen_ids:
                        continue
                    seen_ids.add(row["id"])
                    rows.append(row)
                    if len(rows) >= limit:
                        break
                if len(rows) >= limit:
                    break
        return rows_to_dicts(rows[:limit])

    def _conflicts_for_entities(self, entity_ids: list[str], *, limit: int) -> list[dict]:
        if not entity_ids:
            return []
        placeholders = ",".join("?" for _ in entity_ids)
        rows = self.conn.execute(
            f"""
            SELECT id, conflict_type, entity_id, new_record_type, new_record_id,
                   old_record_type, old_record_id, description, severity, status, created_at
            FROM memory_conflicts
            WHERE status = 'open' AND entity_id IN ({placeholders})
            ORDER BY severity DESC, created_at DESC
            LIMIT ?
            """,
            (*entity_ids, limit),
        ).fetchall()
        return rows_to_dicts(rows)

    def _consolidate_heat_update(self, *, now: str, segment_id: str | None) -> dict:
        if segment_id:
            row = self.conn.execute(
                "SELECT id FROM memory_segments WHERE id = ?",
                (segment_id,),
            ).fetchone()
            if not row:
                return {"updated": 0}
            with self.conn:
                self.segments.visit_segments_for_events(
                    [r["event_id"] for r in self.conn.execute(
                        "SELECT event_id FROM segment_events WHERE segment_id = ?",
                        (segment_id,),
                    ).fetchall()],
                    now=now,
                )
            return {"updated": 1}

        rows = self.conn.execute(
            "SELECT DISTINCT segment_id, event_id FROM segment_events"
        ).fetchall()
        per_segment: dict[str, list[str]] = {}
        for row in rows:
            per_segment.setdefault(row["segment_id"], []).append(row["event_id"])
        updated = 0
        with self.conn:
            for events in per_segment.values():
                touched = self.segments.visit_segments_for_events(events, now=now)
                updated += len(touched)
        return {"updated": updated}

    def _consolidate_persistent_memory(self, *, now: str, segment_id: str | None) -> dict:
        promoted: list[dict] = []
        with self.conn:
            if segment_id:
                result = self.persistent.promote_segment(segment_id, now=now)
                if result:
                    promoted.append(result)
            else:
                for segment in self.segments.promotable_segments():
                    result = self.persistent.promote_segment(segment["id"], now=now)
                    if result:
                        promoted.append(result)
        return {"promoted": promoted}

    def _consolidate_narratives(self, *, now: str, entity_id: str | None) -> dict:
        refreshed = 0
        with self.conn:
            if entity_id:
                self.narratives.invalidate_for_entity(entity_id, now=now)
                narrative = self.narratives.get_or_generate(
                    entity_id=entity_id,
                    scope="global",
                    now=now,
                    force_refresh=True,
                )
                if narrative:
                    refreshed += 1
            else:
                rows = self.conn.execute(
                    """
                    SELECT DISTINCT entity_id FROM event_entities
                    """
                ).fetchall()
                for row in rows:
                    self.narratives.invalidate_for_entity(row["entity_id"], now=now)
                    narrative = self.narratives.get_or_generate(
                        entity_id=row["entity_id"],
                        scope="global",
                        now=now,
                        force_refresh=True,
                    )
                    if narrative:
                        refreshed += 1
        return {"refreshed": refreshed}

    def _consolidate_open_questions(self, *, now: str) -> dict:
        rows = self.conn.execute(
            """
            SELECT q.id, q.subject_entity_id, q.question_type, q.related_entity_ids_json
            FROM open_questions q
            WHERE q.status = 'open'
            """
        ).fetchall()
        answered = 0
        with self.conn:
            for question in rows:
                anchor_type = "time" if question["question_type"] == "when" else "space" if question["question_type"] == "where" else None
                if not anchor_type:
                    continue
                related = json.loads(question["related_entity_ids_json"] or "[]")
                entity_pool = [eid for eid in [question["subject_entity_id"], *related] if eid]
                if not entity_pool:
                    continue
                placeholders = ",".join("?" for _ in entity_pool)
                anchor_rows = self.conn.execute(
                    f"""
                    SELECT a.label, ea.source_event_id
                    FROM entity_anchors ea
                    JOIN spatiotemporal_anchors a ON a.id = ea.anchor_id
                    WHERE ea.entity_id IN ({placeholders}) AND a.anchor_type = ?
                    ORDER BY ea.confidence DESC LIMIT 5
                    """,
                    (*entity_pool, anchor_type),
                ).fetchall()
                if not anchor_rows:
                    continue
                answer_text = "; ".join(row["label"] for row in anchor_rows)
                source_event_id = next(
                    (row["source_event_id"] for row in anchor_rows if row["source_event_id"]),
                    None,
                )
                self.conn.execute(
                    """
                    UPDATE open_questions
                    SET status = 'answered',
                        answer_text = ?,
                        answer_event_id = COALESCE(answer_event_id, ?),
                        confidence = MAX(confidence, 0.66),
                        updated_at = ?,
                        answered_at = ?
                    WHERE id = ?
                    """,
                    (answer_text, source_event_id, now, now, question["id"]),
                )
                answered += 1
        return {"answered": answered}

    def _consolidate_archive(self, *, now: str) -> dict:
        cursor = self.conn.execute(
            """
            UPDATE entity_narratives
            SET status = 'expired'
            WHERE status = 'fresh' AND expires_at IS NOT NULL AND expires_at <= ?
            """,
            (now,),
        )
        return {"narratives_expired": cursor.rowcount}

    def _next_turn_index(self, session_id: str) -> int:
        row = self.conn.execute(
            "SELECT COALESCE(MAX(turn_index), 0) + 1 AS next_turn FROM session_pages WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return int(row["next_turn"])

    def _insert_dialogue_chain(self, *, page_id: str, session_id: str, turn_index: int) -> None:
        prev = self.conn.execute(
            """
            SELECT sp.id, dc.chain_id
            FROM session_pages sp
            LEFT JOIN dialogue_chains dc ON dc.page_id = sp.id
            WHERE sp.session_id = ? AND sp.turn_index < ?
            ORDER BY sp.turn_index DESC
            LIMIT 1
            """,
            (session_id, turn_index),
        ).fetchone()
        previous_page_id = prev["id"] if prev else None
        chain_id = prev["chain_id"] if prev and prev["chain_id"] else new_id("chain")
        self.conn.execute(
            """
            INSERT INTO dialogue_chains(
              page_id, chain_id, previous_page_id, continuity_score, chain_summary, topic_shift_detected
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (page_id, chain_id, previous_page_id, 1.0, "", 0),
        )

    def _like_recall(self, query: str, limit: int, *, session_id: str | None = None) -> list[sqlite3.Row]:
        pattern = f"%{query}%"
        clauses = ["content LIKE ?", "status != 'forgotten'"]
        params: list[object] = [pattern]
        if session_id:
            clauses.append("session_id = ?")
            params.append(session_id)
        params.append(limit)
        return self.conn.execute(
            f"""
            SELECT id, source_type, session_id, content, observed_at
            FROM observations
            WHERE {' AND '.join(clauses)}
            ORDER BY observed_at DESC
            LIMIT ?
            """,
            params,
        ).fetchall()

    def _event_recall(self, query: str, limit: int, *, session_id: str | None = None) -> list[sqlite3.Row]:
        terms = re.findall(r"[\w\u4e00-\u9fff]+", query, flags=re.UNICODE)
        if not terms:
            terms = [query]
        clauses = ["e.status != 'forgotten'"]
        clauses.append("(" + " OR ".join("e.summary LIKE ?" for _ in terms[:12]) + ")")
        params: list[object] = [f"%{term}%" for term in terms[:12]]
        if session_id:
            clauses.append(
                """
                EXISTS (
                  SELECT 1 FROM evidence ev
                  JOIN observations o ON o.id = ev.observation_id
                  WHERE ev.event_id = e.id AND o.session_id = ?
                )
                """
            )
            params.append(session_id)
        params.append(limit)
        return self.conn.execute(
            f"""
            SELECT e.id, e.event_type, e.summary, e.observed_at, e.importance, e.confidence, e.status
            FROM events e
            WHERE {' AND '.join(clauses)}
            ORDER BY importance DESC, observed_at DESC
            LIMIT ?
            """,
            params,
        ).fetchall()

    def _open_question_recall(self, query: str, limit: int) -> list[sqlite3.Row]:
        terms = re.findall(r"[\w\u4e00-\u9fff]+", query, flags=re.UNICODE)
        if not terms:
            terms = [query]
        where = " OR ".join("question_text LIKE ?" for _ in terms[:12])
        params = [f"%{term}%" for term in terms[:12]]
        return self.conn.execute(
            f"""
            SELECT id, question_type, question_text, status, priority, answer_text, confidence, created_at, answered_at
            FROM open_questions
            WHERE {where}
            ORDER BY status, priority DESC, created_at DESC
            LIMIT ?
            """,
            (*params, limit),
        ).fetchall()

    def _enrich_events(self, event_rows: list[sqlite3.Row]) -> list[dict]:
        events = rows_to_dicts(event_rows)
        for event in events:
            event["entities"] = rows_to_dicts(
                self.conn.execute(
                    """
                    SELECT ent.id, ent.entity_type, ent.canonical_name, ee.role_in_event, ee.confidence
                    FROM event_entities ee
                    JOIN entities ent ON ent.id = ee.entity_id
                    WHERE ee.event_id = ?
                    ORDER BY ee.confidence DESC, ent.canonical_name
                    """,
                    (event["id"],),
                ).fetchall()
            )
            event["actions"] = rows_to_dicts(
                self.conn.execute(
                    """
                    SELECT act.id, act.action_type, act.action_text, act.valence, act.confidence,
                           actor.canonical_name AS actor,
                           obj.canonical_name AS object
                    FROM event_actions act
                    LEFT JOIN entities actor ON actor.id = act.actor_entity_id
                    LEFT JOIN entities obj ON obj.id = act.object_entity_id
                    WHERE act.event_id = ?
                    ORDER BY act.confidence DESC
                    """,
                    (event["id"],),
                ).fetchall()
            )
            event["anchors"] = rows_to_dicts(
                self.conn.execute(
                    """
                    SELECT anc.id, anc.anchor_type, anc.label, anc.normalized_value, anc.granularity,
                           ea.relation, ea.confidence
                    FROM event_anchors ea
                    JOIN spatiotemporal_anchors anc ON anc.id = ea.anchor_id
                    WHERE ea.event_id = ?
                    ORDER BY anc.anchor_type, ea.confidence DESC
                    """,
                    (event["id"],),
                ).fetchall()
            )
        return events

    def _find_entity(self, entity_query: str) -> sqlite3.Row | None:
        normalized = normalize_entity_name(entity_query)
        row = self.conn.execute(
            """
            SELECT id, entity_type, canonical_name, normalized_name, aliases_json, description, created_at, last_seen_at
            FROM entities
            WHERE normalized_name = ?
            ORDER BY last_seen_at DESC
            LIMIT 1
            """,
            (normalized,),
        ).fetchone()
        if row:
            return row
        return self.conn.execute(
            """
            SELECT id, entity_type, canonical_name, normalized_name, aliases_json, description, created_at, last_seen_at
            FROM entities
            WHERE canonical_name LIKE ?
            ORDER BY last_seen_at DESC
            LIMIT 1
            """,
            (f"%{entity_query}%",),
        ).fetchone()

    def _entity_event_rows(
        self,
        entity_id: str,
        *,
        since: str | None,
        until: str | None,
        event_types: list[str] | None,
        limit: int,
    ) -> list[sqlite3.Row]:
        clauses = ["ee.entity_id = ?"]
        params: list[object] = [entity_id]
        if since:
            clauses.append("e.observed_at >= ?")
            params.append(since)
        if until:
            clauses.append("e.observed_at <= ?")
            params.append(until)
        if event_types:
            placeholders = ",".join("?" for _ in event_types)
            clauses.append(f"e.event_type IN ({placeholders})")
            params.extend(event_types)
        params.append(limit)
        return self.conn.execute(
            f"""
            SELECT e.id, e.event_type, e.summary, e.observed_at, e.importance, e.confidence, e.status
            FROM event_entities ee
            JOIN events e ON e.id = ee.event_id
            WHERE {' AND '.join(clauses)}
            ORDER BY e.observed_at DESC
            LIMIT ?
            """,
            params,
        ).fetchall()

    def _entity_state_rows(
        self,
        entity_id: str,
        *,
        since: str | None,
        until: str | None,
        state_types: list[str] | None,
        limit: int,
    ) -> list[sqlite3.Row]:
        clauses = ["entity_id = ?"]
        params: list[object] = [entity_id]
        if since:
            clauses.append("valid_from >= ?")
            params.append(since)
        if until:
            clauses.append("valid_from <= ?")
            params.append(until)
        if state_types:
            placeholders = ",".join("?" for _ in state_types)
            clauses.append(f"state_type IN ({placeholders})")
            params.extend(state_types)
        params.append(limit)
        return self.conn.execute(
            f"""
            SELECT id, state_type, value, confidence, valid_from, valid_to, status, source_event_id
            FROM entity_states
            WHERE {' AND '.join(clauses)}
            ORDER BY valid_from DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
