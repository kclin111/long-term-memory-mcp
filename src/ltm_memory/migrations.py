from __future__ import annotations

from dataclasses import dataclass
import hashlib
import sqlite3

from .db import execute_script
from .time import utc_now


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    sql: str

    @property
    def checksum(self) -> str:
        return hashlib.sha256(self.sql.encode("utf-8")).hexdigest()


MIGRATIONS: tuple[Migration, ...] = (
    Migration(
        1,
        "initial_core_schema",
        r"""
CREATE TABLE IF NOT EXISTS schema_migrations (
  version INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  applied_at TEXT NOT NULL,
  checksum TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS settings (
  key TEXT PRIMARY KEY,
  value_json TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS observations (
  id TEXT PRIMARY KEY,
  client_dedup_key TEXT,
  source_type TEXT NOT NULL,
  source_uri TEXT,
  content TEXT NOT NULL,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  observed_at TEXT NOT NULL,
  session_id TEXT,
  created_at TEXT NOT NULL,
  UNIQUE(source_type, session_id, client_dedup_key)
);

CREATE VIRTUAL TABLE IF NOT EXISTS observations_fts USING fts5(
  content,
  observation_id UNINDEXED,
  session_id UNINDEXED,
  source_type UNINDEXED,
  tokenize='unicode61'
);

CREATE TRIGGER IF NOT EXISTS observations_ai AFTER INSERT ON observations BEGIN
  INSERT INTO observations_fts(content, observation_id, session_id, source_type)
  VALUES (new.content, new.id, new.session_id, new.source_type);
END;

CREATE TRIGGER IF NOT EXISTS observations_ad AFTER DELETE ON observations BEGIN
  DELETE FROM observations_fts WHERE observation_id = old.id;
END;

CREATE TRIGGER IF NOT EXISTS observations_au AFTER UPDATE OF content, session_id, source_type ON observations BEGIN
  DELETE FROM observations_fts WHERE observation_id = old.id;
  INSERT INTO observations_fts(content, observation_id, session_id, source_type)
  VALUES (new.content, new.id, new.session_id, new.source_type);
END;

CREATE TABLE IF NOT EXISTS session_pages (
  id TEXT PRIMARY KEY,
  observation_id TEXT,
  session_id TEXT NOT NULL,
  role TEXT NOT NULL,
  content TEXT NOT NULL,
  timestamp TEXT NOT NULL,
  turn_index INTEGER NOT NULL,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  FOREIGN KEY(observation_id) REFERENCES observations(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS dialogue_chains (
  page_id TEXT PRIMARY KEY,
  chain_id TEXT NOT NULL,
  previous_page_id TEXT,
  continuity_score REAL NOT NULL DEFAULT 1.0,
  chain_summary TEXT NOT NULL DEFAULT '',
  topic_shift_detected INTEGER NOT NULL DEFAULT 0,
  FOREIGN KEY(page_id) REFERENCES session_pages(id) ON DELETE CASCADE,
  FOREIGN KEY(previous_page_id) REFERENCES session_pages(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS events (
  id TEXT PRIMARY KEY,
  event_type TEXT NOT NULL,
  summary TEXT NOT NULL,
  structured_json TEXT NOT NULL DEFAULT '{}',
  observed_at TEXT NOT NULL,
  occurred_at TEXT,
  importance REAL NOT NULL DEFAULT 0.0,
  novelty REAL NOT NULL DEFAULT 0.0,
  confidence REAL NOT NULL DEFAULT 0.0,
  status TEXT NOT NULL DEFAULT 'observed',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS entities (
  id TEXT PRIMARY KEY,
  entity_type TEXT NOT NULL,
  canonical_name TEXT NOT NULL,
  normalized_name TEXT NOT NULL,
  aliases_json TEXT NOT NULL DEFAULT '[]',
  description TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL,
  last_seen_at TEXT NOT NULL,
  UNIQUE(entity_type, normalized_name)
);

CREATE TABLE IF NOT EXISTS event_entities (
  event_id TEXT NOT NULL,
  entity_id TEXT NOT NULL,
  role_in_event TEXT,
  confidence REAL NOT NULL DEFAULT 0.0,
  created_at TEXT NOT NULL,
  PRIMARY KEY(event_id, entity_id, role_in_event),
  FOREIGN KEY(event_id) REFERENCES events(id) ON DELETE CASCADE,
  FOREIGN KEY(entity_id) REFERENCES entities(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS evidence (
  id TEXT PRIMARY KEY,
  observation_id TEXT,
  event_id TEXT,
  span_start INTEGER,
  span_end INTEGER,
  content_excerpt TEXT NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY(observation_id) REFERENCES observations(id) ON DELETE SET NULL,
  FOREIGN KEY(event_id) REFERENCES events(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS index_jobs (
  id TEXT PRIMARY KEY,
  idempotency_key TEXT NOT NULL UNIQUE,
  job_type TEXT NOT NULL,
  payload_json TEXT NOT NULL DEFAULT '{}',
  status TEXT NOT NULL DEFAULT 'pending',
  retry_count INTEGER NOT NULL DEFAULT 0,
  max_retries INTEGER NOT NULL DEFAULT 3,
  last_error TEXT,
  scheduled_at TEXT NOT NULL,
  started_at TEXT,
  completed_at TEXT,
  created_at TEXT NOT NULL
);
""",
    ),
    Migration(
        2,
        "episodic_gsw_schema",
        r"""
CREATE TABLE IF NOT EXISTS entity_roles (
  id TEXT PRIMARY KEY,
  entity_id TEXT NOT NULL,
  role_type TEXT NOT NULL,
  scope_entity_id TEXT,
  source_event_id TEXT,
  confidence REAL NOT NULL DEFAULT 0.0,
  valid_from TEXT NOT NULL,
  valid_to TEXT,
  status TEXT NOT NULL DEFAULT 'active',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  FOREIGN KEY(entity_id) REFERENCES entities(id) ON DELETE CASCADE,
  FOREIGN KEY(scope_entity_id) REFERENCES entities(id) ON DELETE SET NULL,
  FOREIGN KEY(source_event_id) REFERENCES events(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_entity_roles_entity_status
ON entity_roles(entity_id, status, role_type);

CREATE TABLE IF NOT EXISTS entity_states (
  id TEXT PRIMARY KEY,
  entity_id TEXT NOT NULL,
  role_id TEXT,
  state_type TEXT NOT NULL,
  value TEXT NOT NULL,
  value_json TEXT NOT NULL DEFAULT '{}',
  scope_entity_id TEXT,
  source_event_id TEXT,
  confidence REAL NOT NULL DEFAULT 0.0,
  valid_from TEXT NOT NULL,
  valid_to TEXT,
  status TEXT NOT NULL DEFAULT 'active',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  FOREIGN KEY(entity_id) REFERENCES entities(id) ON DELETE CASCADE,
  FOREIGN KEY(role_id) REFERENCES entity_roles(id) ON DELETE SET NULL,
  FOREIGN KEY(scope_entity_id) REFERENCES entities(id) ON DELETE SET NULL,
  FOREIGN KEY(source_event_id) REFERENCES events(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_entity_states_entity_status
ON entity_states(entity_id, status, state_type);

CREATE TABLE IF NOT EXISTS event_actions (
  id TEXT PRIMARY KEY,
  event_id TEXT NOT NULL,
  actor_entity_id TEXT,
  action_type TEXT NOT NULL,
  action_text TEXT NOT NULL,
  object_entity_id TEXT,
  valence TEXT NOT NULL DEFAULT 'neutral',
  confidence REAL NOT NULL DEFAULT 0.0,
  created_at TEXT NOT NULL,
  FOREIGN KEY(event_id) REFERENCES events(id) ON DELETE CASCADE,
  FOREIGN KEY(actor_entity_id) REFERENCES entities(id) ON DELETE SET NULL,
  FOREIGN KEY(object_entity_id) REFERENCES entities(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_event_actions_event
ON event_actions(event_id, action_type);

CREATE TABLE IF NOT EXISTS spatiotemporal_anchors (
  id TEXT PRIMARY KEY,
  anchor_type TEXT NOT NULL,
  label TEXT NOT NULL,
  normalized_value TEXT NOT NULL,
  granularity TEXT NOT NULL,
  source_observation_id TEXT,
  confidence REAL NOT NULL DEFAULT 0.0,
  status TEXT NOT NULL DEFAULT 'candidate',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  UNIQUE(anchor_type, normalized_value, granularity),
  FOREIGN KEY(source_observation_id) REFERENCES observations(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_spatiotemporal_anchors_type_value
ON spatiotemporal_anchors(anchor_type, normalized_value, granularity);

CREATE TABLE IF NOT EXISTS anchor_relations (
  id TEXT PRIMARY KEY,
  source_anchor_id TEXT NOT NULL,
  target_anchor_id TEXT NOT NULL,
  predicate TEXT NOT NULL,
  confidence REAL NOT NULL DEFAULT 0.0,
  created_at TEXT NOT NULL,
  FOREIGN KEY(source_anchor_id) REFERENCES spatiotemporal_anchors(id) ON DELETE CASCADE,
  FOREIGN KEY(target_anchor_id) REFERENCES spatiotemporal_anchors(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS event_anchors (
  event_id TEXT NOT NULL,
  anchor_id TEXT NOT NULL,
  relation TEXT NOT NULL DEFAULT 'occurred_at',
  confidence REAL NOT NULL DEFAULT 0.0,
  created_at TEXT NOT NULL,
  PRIMARY KEY(event_id, anchor_id, relation),
  FOREIGN KEY(event_id) REFERENCES events(id) ON DELETE CASCADE,
  FOREIGN KEY(anchor_id) REFERENCES spatiotemporal_anchors(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS entity_anchors (
  entity_id TEXT NOT NULL,
  anchor_id TEXT NOT NULL,
  relation TEXT NOT NULL DEFAULT 'coupled',
  confidence REAL NOT NULL DEFAULT 0.0,
  source_event_id TEXT,
  created_at TEXT NOT NULL,
  PRIMARY KEY(entity_id, anchor_id, relation),
  FOREIGN KEY(entity_id) REFERENCES entities(id) ON DELETE CASCADE,
  FOREIGN KEY(anchor_id) REFERENCES spatiotemporal_anchors(id) ON DELETE CASCADE,
  FOREIGN KEY(source_event_id) REFERENCES events(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS open_questions (
  id TEXT PRIMARY KEY,
  subject_entity_id TEXT,
  question_type TEXT NOT NULL,
  question_text TEXT NOT NULL,
  dedup_key TEXT NOT NULL UNIQUE,
  related_entity_ids_json TEXT NOT NULL DEFAULT '[]',
  status TEXT NOT NULL DEFAULT 'open',
  priority INTEGER NOT NULL DEFAULT 1,
  answer_event_id TEXT,
  answer_text TEXT,
  confidence REAL NOT NULL DEFAULT 0.0,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  answered_at TEXT,
  FOREIGN KEY(subject_entity_id) REFERENCES entities(id) ON DELETE SET NULL,
  FOREIGN KEY(answer_event_id) REFERENCES events(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_open_questions_status_type
ON open_questions(status, question_type, priority);
""",
    ),
    Migration(
        3,
        "entity_merge_audit",
        r"""
CREATE TABLE IF NOT EXISTS entity_merges (
  id TEXT PRIMARY KEY,
  source_entity_id TEXT NOT NULL,
  target_entity_id TEXT NOT NULL,
  source_canonical_name TEXT NOT NULL,
  source_entity_type TEXT NOT NULL,
  moved_counts_json TEXT NOT NULL DEFAULT '{}',
  notes TEXT,
  created_at TEXT NOT NULL,
  FOREIGN KEY(target_entity_id) REFERENCES entities(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_entity_merges_target
ON entity_merges(target_entity_id);
""",
    ),
    Migration(
        4,
        "memoryos_lifecycle_schema",
        r"""
CREATE TABLE IF NOT EXISTS memory_segments (
  id TEXT PRIMARY KEY,
  topic TEXT NOT NULL,
  summary TEXT NOT NULL DEFAULT '',
  entity_ids_json TEXT NOT NULL DEFAULT '[]',
  event_ids_json TEXT NOT NULL DEFAULT '[]',
  keywords_json TEXT NOT NULL DEFAULT '[]',
  heat REAL NOT NULL DEFAULT 0.0,
  visit_count INTEGER NOT NULL DEFAULT 0,
  interaction_length INTEGER NOT NULL DEFAULT 0,
  importance_sum REAL NOT NULL DEFAULT 0.0,
  last_accessed_at TEXT,
  lifecycle TEXT NOT NULL DEFAULT 'active',
  promoted_at TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_memory_segments_lifecycle_heat
ON memory_segments(lifecycle, heat DESC);

CREATE TABLE IF NOT EXISTS segment_events (
  segment_id TEXT NOT NULL,
  event_id TEXT NOT NULL,
  added_at TEXT NOT NULL,
  PRIMARY KEY(segment_id, event_id),
  FOREIGN KEY(segment_id) REFERENCES memory_segments(id) ON DELETE CASCADE,
  FOREIGN KEY(event_id) REFERENCES events(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_segment_events_event
ON segment_events(event_id);

CREATE TABLE IF NOT EXISTS persistent_memories (
  id TEXT PRIMARY KEY,
  memory_type TEXT NOT NULL,
  content TEXT NOT NULL,
  entity_ids_json TEXT NOT NULL DEFAULT '[]',
  supporting_event_ids_json TEXT NOT NULL DEFAULT '[]',
  source_segment_id TEXT,
  confidence REAL NOT NULL DEFAULT 0.0,
  stability REAL NOT NULL DEFAULT 0.0,
  status TEXT NOT NULL DEFAULT 'active',
  reinforcement_count INTEGER NOT NULL DEFAULT 1,
  last_reinforced_at TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  FOREIGN KEY(source_segment_id) REFERENCES memory_segments(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_persistent_memories_type_status
ON persistent_memories(memory_type, status);

CREATE TABLE IF NOT EXISTS entity_narratives (
  id TEXT PRIMARY KEY,
  entity_id TEXT NOT NULL,
  scope TEXT NOT NULL,
  query_hash TEXT,
  content TEXT NOT NULL,
  supporting_event_ids_json TEXT NOT NULL DEFAULT '[]',
  supporting_anchor_ids_json TEXT NOT NULL DEFAULT '[]',
  generated_at TEXT NOT NULL,
  expires_at TEXT,
  confidence REAL NOT NULL DEFAULT 0.0,
  status TEXT NOT NULL DEFAULT 'fresh',
  FOREIGN KEY(entity_id) REFERENCES entities(id) ON DELETE CASCADE,
  CHECK ((scope = 'query_specific' AND query_hash IS NOT NULL)
      OR (scope != 'query_specific' AND query_hash IS NULL))
);

CREATE INDEX IF NOT EXISTS idx_entity_narratives_entity_scope
ON entity_narratives(entity_id, scope);

CREATE INDEX IF NOT EXISTS idx_entity_narratives_query_hash
ON entity_narratives(query_hash);

CREATE TABLE IF NOT EXISTS memory_conflicts (
  id TEXT PRIMARY KEY,
  conflict_type TEXT NOT NULL,
  entity_id TEXT,
  new_record_type TEXT NOT NULL,
  new_record_id TEXT NOT NULL,
  old_record_type TEXT NOT NULL,
  old_record_id TEXT NOT NULL,
  description TEXT NOT NULL DEFAULT '',
  severity TEXT NOT NULL DEFAULT 'medium',
  status TEXT NOT NULL DEFAULT 'open',
  created_at TEXT NOT NULL,
  resolved_at TEXT,
  FOREIGN KEY(entity_id) REFERENCES entities(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_memory_conflicts_status_type
ON memory_conflicts(status, conflict_type);
""",
    ),
    Migration(
        5,
        "forget_and_status_columns",
        r"""
ALTER TABLE observations ADD COLUMN status TEXT NOT NULL DEFAULT 'active';
ALTER TABLE observations ADD COLUMN updated_at TEXT;

ALTER TABLE evidence ADD COLUMN status TEXT NOT NULL DEFAULT 'active';
ALTER TABLE evidence ADD COLUMN updated_at TEXT;

CREATE INDEX IF NOT EXISTS idx_observations_status
ON observations(status);

CREATE INDEX IF NOT EXISTS idx_evidence_status
ON evidence(status);

CREATE INDEX IF NOT EXISTS idx_events_status_observed_at
ON events(status, observed_at DESC);

CREATE INDEX IF NOT EXISTS idx_index_jobs_status_started_at
ON index_jobs(status, started_at);
""",
    ),
)


def apply_migrations(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        "version INTEGER PRIMARY KEY, name TEXT NOT NULL, applied_at TEXT NOT NULL, checksum TEXT NOT NULL)"
    )
    applied = {
        row["version"]: row
        for row in conn.execute("SELECT version, checksum FROM schema_migrations").fetchall()
    }
    for migration in MIGRATIONS:
        row = applied.get(migration.version)
        if row:
            if row["checksum"] != migration.checksum:
                raise RuntimeError(f"Migration checksum mismatch for version {migration.version}")
            continue
        with conn:
            execute_script(conn, migration.sql)
            conn.execute(
                "INSERT INTO schema_migrations(version, name, applied_at, checksum) VALUES (?, ?, ?, ?)",
                (migration.version, migration.name, utc_now(), migration.checksum),
            )
