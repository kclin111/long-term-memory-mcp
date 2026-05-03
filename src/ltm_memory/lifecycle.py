"""MemoryOS-style lifecycle layer (PRD §7.11/§7.12/§7.15/§7.16).

This module owns four pieces of the M3 milestone that share a lot of
plumbing and are easier to evolve together than to spread across the
worker:

* :class:`SegmentManager` — simplified MTM. Assigns a freshly integrated
  event to a topic segment, recomputes heat from the MemoryOS formula,
  and promotes the segment when it crosses ``tau``.
* :class:`PersistentMemoryStore` — extracts/refreshes
  ``persistent_memories`` rows when a segment is ready or when a
  high-importance event lands directly (per PRD §7.15 "immediate
  event-level promotion").
* :class:`EntityNarrativeCache` — generates rule-based entity-level
  chronological narratives, caches them with the PRD §7.11 TTL, and
  invalidates on demand.
* :class:`ConflictDetector` — records ``memory_conflicts`` rows when a
  new ``EntityState`` does not pass the supersede checklist of PRD §7.6.

Heat formula (PRD §7.15):

    heat = alpha * N_visit + beta * L_interaction + gamma * R_recency
    R_recency = exp(-delta_t / mu)
    default mu = 1e7 seconds, default tau = 5

We use small alphas/betas because ``visit_count`` and
``interaction_length`` would otherwise dominate. The formula is
parameterized so future tuning does not require a schema change.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import sqlite3
from typing import Iterable
from uuid import uuid4

from .time import utc_now


SEGMENT_HEAT_ALPHA = 0.05    # weight of visit_count
SEGMENT_HEAT_BETA = 0.001    # weight of interaction_length (chars)
SEGMENT_HEAT_GAMMA = 1.0     # weight of recency
SEGMENT_HEAT_DELTA = 2.0     # weight of importance bonus (extended_heat)
SEGMENT_HEAT_MU = 1e7        # recency decay constant (seconds)
SEGMENT_PROMOTE_TAU = 5.0    # promotion threshold

PERSISTENT_MEMORY_IMPORTANCE_FLOOR = 0.65
PERSISTENT_MEMORY_CONFIDENCE_FLOOR = 0.50

NARRATIVE_TTL_SECONDS = {
    "global": 7 * 24 * 3600,
    "project": 3 * 24 * 3600,
    "session": 24 * 3600,
    "query_specific": 60 * 60,
}


_STOP_TOKENS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "have",
    "has", "had", "was", "were", "are", "but", "not", "you", "user",
    "important", "remember", "记住", "重要",
}


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def _json_dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    cleaned = ts.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(cleaned)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _seconds_since(ts: str | None, *, now: str) -> float:
    start = _parse_iso(ts)
    end = _parse_iso(now) or datetime.now(timezone.utc)
    if not start:
        return 0.0
    return max(0.0, (end - start).total_seconds())


def _topic_keywords(text: str) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()
    for raw in text.replace(".", " ").replace(",", " ").split():
        token = raw.strip().lower()
        token = "".join(ch for ch in token if ch.isalnum())
        if len(token) < 4:
            continue
        if token in _STOP_TOKENS:
            continue
        if token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return tokens[:8]


def _query_hash(query: str, *, scope: str) -> str:
    digest = hashlib.sha256(f"{scope}|{query.strip().lower()}".encode("utf-8")).hexdigest()
    return digest[:32]


@dataclass
class SegmentAssignment:
    segment_id: str
    created: bool
    heat: float
    lifecycle: str
    promoted: bool


class SegmentManager:
    """Owns memory_segments membership and heat updates."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def assign_event(
        self,
        *,
        event_id: str,
        event_summary: str,
        event_type: str,
        event_importance: float,
        entity_ids: Iterable[str],
        observed_at: str,
        now: str,
    ) -> SegmentAssignment:
        entity_ids = list(dict.fromkeys(entity_ids))
        keywords = _topic_keywords(event_summary)

        segment = self._find_matching_segment(entity_ids=entity_ids, keywords=keywords)
        created = segment is None
        if segment is None:
            segment_id = _new_id("seg")
            topic = self._choose_topic(
                entity_ids=entity_ids,
                keywords=keywords,
                event_type=event_type,
                entity_names=self._entity_display_names(entity_ids),
            )
            self.conn.execute(
                """
                INSERT INTO memory_segments(
                  id, topic, summary, entity_ids_json, event_ids_json, keywords_json,
                  heat, visit_count, interaction_length, importance_sum,
                  last_accessed_at, lifecycle, created_at, updated_at
                ) VALUES (?, ?, '', ?, ?, ?, 0.0, 0, 0, 0.0, ?, 'active', ?, ?)
                """,
                (
                    segment_id,
                    topic,
                    _json_dumps(entity_ids),
                    _json_dumps([]),
                    _json_dumps(keywords),
                    now,
                    now,
                    now,
                ),
            )
            segment = self._fetch(segment_id)

        if segment is None:
            raise RuntimeError("segment row missing immediately after insert")

        merged_entity_ids = sorted(set(json.loads(segment["entity_ids_json"]) + entity_ids))
        merged_event_ids = sorted(set(json.loads(segment["event_ids_json"]) + [event_id]))
        merged_keywords = sorted(set(json.loads(segment["keywords_json"]) + keywords))
        new_interaction_length = int(segment["interaction_length"]) + len(event_summary or "")
        new_importance_sum = float(segment["importance_sum"]) + max(0.0, event_importance or 0.0)

        self.conn.execute(
            "INSERT OR IGNORE INTO segment_events(segment_id, event_id, added_at) VALUES (?, ?, ?)",
            (segment["id"], event_id, now),
        )

        heat = self._compute_heat(
            visit_count=int(segment["visit_count"]),
            interaction_length=new_interaction_length,
            last_accessed_at=segment["last_accessed_at"] or now,
            importance_sum=new_importance_sum,
            now=now,
        )

        promoted = False
        lifecycle = segment["lifecycle"]
        if heat >= SEGMENT_PROMOTE_TAU and lifecycle == "active":
            lifecycle = "promoted"
            promoted = True

        self.conn.execute(
            """
            UPDATE memory_segments
            SET entity_ids_json = ?,
                event_ids_json = ?,
                keywords_json = ?,
                interaction_length = ?,
                importance_sum = ?,
                heat = ?,
                lifecycle = ?,
                promoted_at = COALESCE(promoted_at, ?),
                updated_at = ?
            WHERE id = ?
            """,
            (
                _json_dumps(merged_entity_ids),
                _json_dumps(merged_event_ids),
                _json_dumps(merged_keywords),
                new_interaction_length,
                new_importance_sum,
                heat,
                lifecycle,
                now if promoted else None,
                now,
                segment["id"],
            ),
        )

        return SegmentAssignment(
            segment_id=segment["id"],
            created=created,
            heat=heat,
            lifecycle=lifecycle,
            promoted=promoted,
        )

    def visit_segments_for_events(self, event_ids: Iterable[str], *, now: str) -> list[str]:
        """Increment visit counts for any segment containing one of the events.

        Each segment is incremented at most once per call, matching PRD §7.15.
        Returns the list of touched segment ids.
        """

        ids = list(dict.fromkeys(event_ids))
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        rows = self.conn.execute(
            f"""
            SELECT DISTINCT segment_id FROM segment_events
            WHERE event_id IN ({placeholders})
            """,
            ids,
        ).fetchall()
        touched: list[str] = []
        for row in rows:
            segment = self._fetch(row["segment_id"])
            if not segment:
                continue
            new_visits = int(segment["visit_count"]) + 1
            heat = self._compute_heat(
                visit_count=new_visits,
                interaction_length=int(segment["interaction_length"]),
                last_accessed_at=now,
                importance_sum=float(segment["importance_sum"]),
                now=now,
            )
            lifecycle = segment["lifecycle"]
            promoted_at = segment["promoted_at"]
            if heat >= SEGMENT_PROMOTE_TAU and lifecycle == "active":
                lifecycle = "promoted"
                promoted_at = promoted_at or now
            self.conn.execute(
                """
                UPDATE memory_segments
                SET visit_count = ?,
                    last_accessed_at = ?,
                    heat = ?,
                    lifecycle = ?,
                    promoted_at = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (new_visits, now, heat, lifecycle, promoted_at, now, segment["id"]),
            )
            touched.append(segment["id"])
        return touched

    def promotable_segments(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            """
            SELECT * FROM memory_segments
            WHERE lifecycle = 'promoted' AND heat >= ?
            ORDER BY heat DESC
            """,
            (SEGMENT_PROMOTE_TAU,),
        ).fetchall()

    def get_event_segment(self, event_id: str) -> sqlite3.Row | None:
        row = self.conn.execute(
            """
            SELECT s.* FROM memory_segments s
            JOIN segment_events se ON se.segment_id = s.id
            WHERE se.event_id = ?
            ORDER BY s.heat DESC
            LIMIT 1
            """,
            (event_id,),
        ).fetchone()
        return row

    def _fetch(self, segment_id: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM memory_segments WHERE id = ?",
            (segment_id,),
        ).fetchone()

    def _find_matching_segment(self, *, entity_ids: list[str], keywords: list[str]) -> sqlite3.Row | None:
        if not entity_ids and not keywords:
            return None
        rows = self.conn.execute(
            """
            SELECT * FROM memory_segments
            WHERE lifecycle IN ('active', 'promoted')
            ORDER BY updated_at DESC
            LIMIT 50
            """
        ).fetchall()
        best: sqlite3.Row | None = None
        best_score = 0
        for row in rows:
            seg_entities = set(json.loads(row["entity_ids_json"]))
            seg_keywords = set(json.loads(row["keywords_json"]))
            entity_overlap = len(seg_entities & set(entity_ids))
            keyword_overlap = len(seg_keywords & set(keywords))
            score = entity_overlap * 2 + keyword_overlap
            if entity_overlap >= 1 and score > best_score:
                best_score = score
                best = row
        return best

    def _entity_display_names(self, entity_ids: list[str]) -> dict[str, dict]:
        if not entity_ids:
            return {}
        placeholders = ",".join("?" for _ in entity_ids)
        rows = self.conn.execute(
            f"SELECT id, canonical_name, entity_type FROM entities WHERE id IN ({placeholders})",
            entity_ids,
        ).fetchall()
        return {row["id"]: {"name": row["canonical_name"], "type": row["entity_type"]} for row in rows}

    @staticmethod
    def _choose_topic(
        *,
        entity_ids: list[str],
        keywords: list[str],
        event_type: str,
        entity_names: dict[str, dict],
    ) -> str:
        non_user = [
            eid for eid in entity_ids
            if eid in entity_names and entity_names[eid]["name"] != "user"
        ]
        if non_user:
            return entity_names[non_user[0]]["name"]
        if entity_ids and entity_ids[0] in entity_names:
            return entity_names[entity_ids[0]]["name"]
        if keywords:
            return keywords[0]
        return event_type or "general"

    @staticmethod
    def _compute_heat(
        *,
        visit_count: int,
        interaction_length: int,
        last_accessed_at: str,
        importance_sum: float,
        now: str,
    ) -> float:
        delta_t = _seconds_since(last_accessed_at, now=now)
        recency = math.exp(-delta_t / SEGMENT_HEAT_MU)
        heat = (
            SEGMENT_HEAT_ALPHA * visit_count
            + SEGMENT_HEAT_BETA * interaction_length
            + SEGMENT_HEAT_GAMMA * recency
            + SEGMENT_HEAT_DELTA * max(0.0, importance_sum)
        )
        return round(heat, 4)


class PersistentMemoryStore:
    """Promotes high-heat segments and high-importance events into LPM."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def promote_segment(self, segment_id: str, *, now: str) -> dict | None:
        segment = self.conn.execute(
            "SELECT * FROM memory_segments WHERE id = ?",
            (segment_id,),
        ).fetchone()
        if not segment:
            return None
        if segment["lifecycle"] not in ("promoted", "active"):
            return None

        event_ids = json.loads(segment["event_ids_json"])
        if not event_ids:
            return None
        placeholders = ",".join("?" for _ in event_ids)
        events = self.conn.execute(
            f"""
            SELECT id, event_type, summary, importance, confidence, observed_at
            FROM events
            WHERE id IN ({placeholders})
            ORDER BY importance DESC, observed_at DESC
            """,
            event_ids,
        ).fetchall()
        qualifying = [
            event for event in events
            if float(event["importance"]) >= PERSISTENT_MEMORY_IMPORTANCE_FLOOR
            and float(event["confidence"]) >= PERSISTENT_MEMORY_CONFIDENCE_FLOOR
        ]
        if not qualifying:
            return None

        memory_type = self._infer_memory_type([row["event_type"] for row in qualifying])
        entity_ids = json.loads(segment["entity_ids_json"])
        supporting_ids = [row["id"] for row in qualifying]
        content = self._compose_content(memory_type, segment, qualifying)
        avg_confidence = sum(float(row["confidence"]) for row in qualifying) / len(qualifying)
        stability = min(1.0, round(0.4 + 0.1 * len(qualifying), 4))

        existing = self.conn.execute(
            """
            SELECT id, reinforcement_count FROM persistent_memories
            WHERE source_segment_id = ? AND memory_type = ?
            """,
            (segment_id, memory_type),
        ).fetchone()
        if existing:
            new_count = int(existing["reinforcement_count"]) + 1
            self.conn.execute(
                """
                UPDATE persistent_memories
                SET content = ?,
                    entity_ids_json = ?,
                    supporting_event_ids_json = ?,
                    confidence = ?,
                    stability = ?,
                    reinforcement_count = ?,
                    last_reinforced_at = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    content,
                    _json_dumps(entity_ids),
                    _json_dumps(supporting_ids),
                    round(avg_confidence, 4),
                    stability,
                    new_count,
                    now,
                    now,
                    existing["id"],
                ),
            )
            return {"id": existing["id"], "memory_type": memory_type, "reinforced": True}

        memory_id = _new_id("pm")
        self.conn.execute(
            """
            INSERT INTO persistent_memories(
              id, memory_type, content, entity_ids_json, supporting_event_ids_json,
              source_segment_id, confidence, stability, status, reinforcement_count,
              last_reinforced_at, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active', 1, ?, ?, ?)
            """,
            (
                memory_id,
                memory_type,
                content,
                _json_dumps(entity_ids),
                _json_dumps(supporting_ids),
                segment_id,
                round(avg_confidence, 4),
                stability,
                now,
                now,
                now,
            ),
        )
        return {"id": memory_id, "memory_type": memory_type, "reinforced": False}

    @staticmethod
    def _infer_memory_type(event_types: list[str]) -> str:
        order = ["preference", "decision", "project_state", "fact", "correction", "error_workaround", "commitment"]
        seen = {event_type for event_type in event_types}
        for candidate in order:
            if candidate in seen:
                return {
                    "preference": "user_preference",
                    "decision": "design_decision",
                    "project_state": "project_knowledge",
                    "fact": "stable_fact",
                    "correction": "user_preference",
                    "error_workaround": "workflow_pattern",
                    "commitment": "habit",
                }[candidate]
        return "stable_fact"

    @staticmethod
    def _compose_content(memory_type: str, segment: sqlite3.Row, events: list[sqlite3.Row]) -> str:
        topic = segment["topic"] or "general"
        bullets = [f"- {event['summary']}" for event in events[:5]]
        header = f"[{memory_type}] topic={topic} (segment_id={segment['id']})"
        return "\n".join([header, *bullets])


class EntityNarrativeCache:
    """Generates and caches entity-level chronological narratives."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def get_or_generate(
        self,
        *,
        entity_id: str,
        scope: str = "query_specific",
        query: str | None = None,
        now: str | None = None,
        force_refresh: bool = False,
    ) -> dict | None:
        now = now or utc_now()
        if scope == "query_specific" and not query:
            raise ValueError("query is required for scope=query_specific")
        if scope != "query_specific" and query is not None:
            query = None
        query_hash = _query_hash(query, scope=scope) if scope == "query_specific" else None

        if not force_refresh:
            cached = self._load_fresh(entity_id=entity_id, scope=scope, query_hash=query_hash, now=now)
            if cached:
                return cached

        return self._generate(
            entity_id=entity_id,
            scope=scope,
            query_hash=query_hash,
            now=now,
        )

    def invalidate_for_entity(self, entity_id: str, *, now: str | None = None) -> int:
        now = now or utc_now()
        cursor = self.conn.execute(
            "UPDATE entity_narratives SET status = 'stale' WHERE entity_id = ? AND status = 'fresh'",
            (entity_id,),
        )
        return cursor.rowcount

    def _load_fresh(
        self,
        *,
        entity_id: str,
        scope: str,
        query_hash: str | None,
        now: str,
    ) -> dict | None:
        if query_hash is None:
            row = self.conn.execute(
                """
                SELECT * FROM entity_narratives
                WHERE entity_id = ? AND scope = ? AND query_hash IS NULL
                  AND status = 'fresh'
                ORDER BY generated_at DESC LIMIT 1
                """,
                (entity_id, scope),
            ).fetchone()
        else:
            row = self.conn.execute(
                """
                SELECT * FROM entity_narratives
                WHERE entity_id = ? AND scope = ? AND query_hash = ?
                  AND status = 'fresh'
                ORDER BY generated_at DESC LIMIT 1
                """,
                (entity_id, scope, query_hash),
            ).fetchone()
        if not row:
            return None
        if row["expires_at"]:
            expires = _parse_iso(row["expires_at"])
            current = _parse_iso(now)
            if expires and current and current >= expires:
                self.conn.execute(
                    "UPDATE entity_narratives SET status = 'stale' WHERE id = ?",
                    (row["id"],),
                )
                return None
        return self._row_to_dict(row, cached=True)

    def _generate(
        self,
        *,
        entity_id: str,
        scope: str,
        query_hash: str | None,
        now: str,
    ) -> dict | None:
        entity = self.conn.execute(
            "SELECT id, canonical_name, entity_type FROM entities WHERE id = ?",
            (entity_id,),
        ).fetchone()
        if not entity:
            return None

        events = self.conn.execute(
            """
            SELECT e.id, e.event_type, e.summary, e.observed_at, e.importance, e.confidence
            FROM events e
            JOIN event_entities ee ON ee.event_id = e.id
            WHERE ee.entity_id = ?
            ORDER BY e.observed_at DESC
            LIMIT 6
            """,
            (entity_id,),
        ).fetchall()
        roles = self.conn.execute(
            """
            SELECT role_type, status FROM entity_roles
            WHERE entity_id = ? AND status = 'active'
            ORDER BY confidence DESC LIMIT 5
            """,
            (entity_id,),
        ).fetchall()
        states = self.conn.execute(
            """
            SELECT state_type, value FROM entity_states
            WHERE entity_id = ? AND status = 'active'
            ORDER BY valid_from DESC LIMIT 5
            """,
            (entity_id,),
        ).fetchall()
        anchors = self.conn.execute(
            """
            SELECT a.id, a.anchor_type, a.label
            FROM entity_anchors ea
            JOIN spatiotemporal_anchors a ON a.id = ea.anchor_id
            WHERE ea.entity_id = ?
            ORDER BY ea.confidence DESC LIMIT 5
            """,
            (entity_id,),
        ).fetchall()

        if not events and not roles and not states:
            return None

        lines = [f"{entity['canonical_name']} ({entity['entity_type']})"]
        if roles:
            lines.append("Roles: " + ", ".join(row["role_type"] for row in roles))
        if states:
            lines.append("States: " + "; ".join(f"{row['state_type']}={row['value']}" for row in states))
        if anchors:
            lines.append("Anchors: " + ", ".join(f"{row['anchor_type']}={row['label']}" for row in anchors))
        if events:
            lines.append("Recent events:")
            for event in events:
                lines.append(f"  - [{event['observed_at']}] {event['event_type']}: {event['summary']}")
        content = "\n".join(lines)

        confidence = round(
            (
                sum(float(event["confidence"]) for event in events) / max(1, len(events))
                if events else 0.5
            ),
            4,
        )
        ttl = NARRATIVE_TTL_SECONDS.get(scope, NARRATIVE_TTL_SECONDS["query_specific"])
        expires_at = self._add_seconds(now, ttl)

        narrative_id = _new_id("nar")
        supporting_event_ids = [event["id"] for event in events]
        supporting_anchor_ids = [anchor["id"] for anchor in anchors]
        self.conn.execute(
            """
            INSERT INTO entity_narratives(
              id, entity_id, scope, query_hash, content,
              supporting_event_ids_json, supporting_anchor_ids_json,
              generated_at, expires_at, confidence, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'fresh')
            """,
            (
                narrative_id,
                entity_id,
                scope,
                query_hash,
                content,
                _json_dumps(supporting_event_ids),
                _json_dumps(supporting_anchor_ids),
                now,
                expires_at,
                confidence,
            ),
        )
        row = self.conn.execute("SELECT * FROM entity_narratives WHERE id = ?", (narrative_id,)).fetchone()
        return self._row_to_dict(row, cached=False)

    @staticmethod
    def _row_to_dict(row: sqlite3.Row, *, cached: bool) -> dict:
        data = dict(row)
        data["cached"] = cached
        data["supporting_event_ids"] = json.loads(data.pop("supporting_event_ids_json"))
        data["supporting_anchor_ids"] = json.loads(data.pop("supporting_anchor_ids_json"))
        return data

    @staticmethod
    def _add_seconds(now: str, seconds: int) -> str:
        current = _parse_iso(now) or datetime.now(timezone.utc)
        future = current.timestamp() + seconds
        return datetime.fromtimestamp(future, tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class ConflictDetector:
    """Records contradictions between EntityState rows (PRD §7.6/§7.16)."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def evaluate_state(
        self,
        *,
        entity_id: str,
        state_type: str,
        new_value: str,
        new_confidence: float,
        new_state_id: str,
        valid_from: str,
        now: str,
    ) -> dict | None:
        existing = self.conn.execute(
            """
            SELECT id, value, confidence, valid_from
            FROM entity_states
            WHERE entity_id = ? AND state_type = ? AND status = 'active' AND id != ?
            ORDER BY valid_from DESC
            """,
            (entity_id, state_type, new_state_id),
        ).fetchall()
        if not existing:
            return None
        for row in existing:
            if row["value"] == new_value:
                continue
            if not self._supersedes(row, new_confidence=new_confidence, new_valid_from=valid_from):
                return self._record(
                    conflict_type="state_contradiction",
                    entity_id=entity_id,
                    new_record_type="entity_state",
                    new_record_id=new_state_id,
                    old_record_type="entity_state",
                    old_record_id=row["id"],
                    description=(
                        f"new state value {new_value!r} contradicts existing {row['value']!r} "
                        f"on entity={entity_id} state_type={state_type}"
                    ),
                    severity="medium",
                    now=now,
                )
        return None

    @staticmethod
    def _supersedes(existing_row: sqlite3.Row, *, new_confidence: float, new_valid_from: str) -> bool:
        if new_confidence < float(existing_row["confidence"]):
            return False
        old_ts = _parse_iso(existing_row["valid_from"])
        new_ts = _parse_iso(new_valid_from)
        if old_ts and new_ts and new_ts < old_ts:
            return False
        return new_confidence >= 0.65

    def _record(
        self,
        *,
        conflict_type: str,
        entity_id: str | None,
        new_record_type: str,
        new_record_id: str,
        old_record_type: str,
        old_record_id: str,
        description: str,
        severity: str,
        now: str,
    ) -> dict:
        existing = self.conn.execute(
            """
            SELECT id FROM memory_conflicts
            WHERE conflict_type = ? AND new_record_id = ? AND old_record_id = ? AND status = 'open'
            """,
            (conflict_type, new_record_id, old_record_id),
        ).fetchone()
        if existing:
            return {"id": existing["id"], "conflict_type": conflict_type, "deduplicated": True}
        conflict_id = _new_id("conf")
        self.conn.execute(
            """
            INSERT INTO memory_conflicts(
              id, conflict_type, entity_id, new_record_type, new_record_id,
              old_record_type, old_record_id, description, severity, status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?)
            """,
            (
                conflict_id,
                conflict_type,
                entity_id,
                new_record_type,
                new_record_id,
                old_record_type,
                old_record_id,
                description,
                severity,
                now,
            ),
        )
        return {"id": conflict_id, "conflict_type": conflict_type, "deduplicated": False}


__all__ = [
    "SegmentManager",
    "SegmentAssignment",
    "PersistentMemoryStore",
    "EntityNarrativeCache",
    "ConflictDetector",
    "SEGMENT_PROMOTE_TAU",
    "PERSISTENT_MEMORY_IMPORTANCE_FLOOR",
    "PERSISTENT_MEMORY_CONFIDENCE_FLOOR",
    "NARRATIVE_TTL_SECONDS",
]
