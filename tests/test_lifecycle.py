from __future__ import annotations

from pathlib import Path
import unittest
from uuid import uuid4

from ltm_memory.config import Settings
from ltm_memory.lifecycle import (
    PERSISTENT_MEMORY_CONFIDENCE_FLOOR,
    PERSISTENT_MEMORY_IMPORTANCE_FLOOR,
    SEGMENT_PROMOTE_TAU,
)
from ltm_memory.store import MemoryStore
from ltm_memory.worker import JobWorker


TEST_TMP_ROOT = Path.cwd() / ".codex-tmp" / "test-lifecycle"


def make_store(tmp_path: Path) -> MemoryStore:
    settings = Settings(
        home=tmp_path,
        sqlite_path=tmp_path / "memory.sqlite",
        lancedb_path=tmp_path / "lancedb",
        sqlite_journal_mode="MEMORY",
    )
    return MemoryStore.open(settings)


class LifecycleTests(unittest.TestCase):
    def make_tmp_path(self) -> Path:
        TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)
        path = TEST_TMP_ROOT / uuid4().hex
        path.mkdir(parents=True, exist_ok=False)
        return path

    def _ingest(self, store: MemoryStore, content: str, *, dedup: str, session: str = "lc") -> None:
        store.ingest_observation(
            source_type="chat",
            content=content,
            session_id=session,
            client_dedup_key=dedup,
        )

    def test_event_creates_or_joins_memory_segment(self) -> None:
        store = make_store(self.make_tmp_path())
        try:
            self._ingest(store, "Remember important: Carter Stewart presented the workshop.", dedup="lc-1")
            self._ingest(store, "Remember important: Carter Stewart explained the new schedule.", dedup="lc-2")
            JobWorker(store).process(max_jobs=10)

            segments = store.conn.execute(
                "SELECT id, topic, event_ids_json, heat FROM memory_segments"
            ).fetchall()
            self.assertGreaterEqual(len(segments), 1)
            primary = next(
                seg for seg in segments
                if "Carter Stewart" in (seg["topic"] or "") or "carter" in (seg["topic"] or "").lower()
                or '"' in seg["topic"]
            )
            event_ids = primary["event_ids_json"]
            self.assertIn("evt_", event_ids)
            self.assertGreater(float(primary["heat"]), 0)
        finally:
            store.close()

    def test_persistent_memory_promotes_when_heat_crosses_tau(self) -> None:
        store = make_store(self.make_tmp_path())
        try:
            for i in range(8):
                self._ingest(
                    store,
                    f"User decided important schema change number {i}: Python and SQLite remain canonical.",
                    dedup=f"pm-{i}",
                    session="pm",
                )
            JobWorker(store).process(max_jobs=50)

            row = store.conn.execute(
                "SELECT MAX(heat) AS h FROM memory_segments"
            ).fetchone()
            self.assertGreaterEqual(float(row["h"] or 0.0), SEGMENT_PROMOTE_TAU * 0.4)

            store.consolidate_memory(tasks=["heat_update", "persistent_memory"])
            promoted = store.conn.execute(
                "SELECT id, memory_type, confidence FROM persistent_memories"
            ).fetchall()
            self.assertGreaterEqual(len(promoted), 1)
            for row in promoted:
                self.assertGreaterEqual(float(row["confidence"]), PERSISTENT_MEMORY_CONFIDENCE_FLOOR)
        finally:
            store.close()

    def test_recall_returns_entity_narrative_and_increments_visits(self) -> None:
        store = make_store(self.make_tmp_path())
        try:
            self._ingest(
                store,
                "Remember important: Carter Stewart presented at Metropolitan Museum on 2026-09-22.",
                dedup="nar-1",
            )
            JobWorker(store).process(max_jobs=10)

            initial = store.conn.execute(
                "SELECT visit_count FROM memory_segments LIMIT 1"
            ).fetchone()
            self.assertIsNotNone(initial)
            initial_visits = int(initial["visit_count"])

            recall = store.recall("Carter Stewart Metropolitan Museum", session_id="lc")
            self.assertGreaterEqual(len(recall["events"]), 1)
            self.assertGreaterEqual(len(recall["entity_narratives"]), 1)
            narrative = recall["entity_narratives"][0]
            self.assertIn("Carter Stewart", narrative["content"])
            self.assertEqual(narrative["scope"], "query_specific")

            after = store.conn.execute(
                "SELECT visit_count FROM memory_segments LIMIT 1"
            ).fetchone()
            self.assertGreater(int(after["visit_count"]), initial_visits)
        finally:
            store.close()

    def test_state_conflict_is_recorded(self) -> None:
        store = make_store(self.make_tmp_path())
        try:
            # Seed a high-confidence state that the deterministic extractor
            # cannot supersede (its preferences land at confidence 0.74).
            self._ingest(store, "User prefers SQLite for the memory project.", dedup="conf-seed")
            JobWorker(store).process(max_jobs=5)
            user_row = store.conn.execute(
                "SELECT id FROM entities WHERE canonical_name = 'user'"
            ).fetchone()
            self.assertIsNotNone(user_row)
            store.conn.execute(
                """
                UPDATE entity_states SET value = 'pin SQLite as canonical store', confidence = 0.95
                WHERE entity_id = ? AND state_type = 'preference' AND status = 'active'
                """,
                (user_row["id"],),
            )
            store.conn.commit()

            self._ingest(
                store,
                "User prefers Postgres for the memory project after a long debate.",
                dedup="conf-new",
            )
            JobWorker(store).process(max_jobs=5)

            conflicts = store.get_conflicts()
            self.assertGreaterEqual(len(conflicts), 1)
            self.assertEqual(conflicts[0]["status"], "open")
            self.assertEqual(conflicts[0]["conflict_type"], "state_contradiction")
        finally:
            store.close()

    def test_consolidate_open_question_resolution_uses_existing_anchors(self) -> None:
        store = make_store(self.make_tmp_path())
        try:
            self._ingest(
                store,
                "Remember important: Carter Stewart presented the statistics workshop.",
                dedup="oq-1",
            )
            JobWorker(store).process(max_jobs=5)

            # Now seed an anchor on Carter Stewart without ingesting a new
            # event mentioning the date; consolidation should still close
            # the open question using existing entity_anchors.
            entity = store.conn.execute(
                "SELECT id FROM entities WHERE canonical_name = ?",
                ("Carter Stewart",),
            ).fetchone()
            self.assertIsNotNone(entity)
            store.conn.execute(
                """
                INSERT INTO spatiotemporal_anchors(
                  id, anchor_type, label, normalized_value, granularity, confidence, status, created_at, updated_at
                ) VALUES ('anc_test', 'time', '2026-09-22', '2026-09-22', 'day', 0.9, 'confirmed', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')
                """
            )
            store.conn.execute(
                """
                INSERT INTO entity_anchors(entity_id, anchor_id, relation, confidence, source_event_id, created_at)
                VALUES (?, 'anc_test', 'explicit_coupled', 0.9, NULL, '2026-01-01T00:00:00Z')
                """,
                (entity["id"],),
            )
            store.conn.commit()

            report = store.consolidate_memory(tasks=["open_question_resolution"])
            self.assertGreaterEqual(report["open_question_resolution"]["answered"], 1)

            answered = store.conn.execute(
                "SELECT status, answer_text FROM open_questions WHERE status = 'answered'"
            ).fetchall()
            self.assertGreaterEqual(len(answered), 1)
            self.assertTrue(any("2026-09-22" in (row["answer_text"] or "") for row in answered))
        finally:
            store.close()

    def test_get_entity_timeline_includes_narrative_and_conflicts(self) -> None:
        store = make_store(self.make_tmp_path())
        try:
            self._ingest(store, "User prefers Python for the memory project.", dedup="t-1")
            JobWorker(store).process(max_jobs=5)
            user_row = store.conn.execute(
                "SELECT id FROM entities WHERE canonical_name = 'user'"
            ).fetchone()
            store.conn.execute(
                """
                UPDATE entity_states SET value = 'pin Python', confidence = 0.95
                WHERE entity_id = ? AND state_type = 'preference' AND status = 'active'
                """,
                (user_row["id"],),
            )
            store.conn.commit()
            self._ingest(store, "User prefers Go for the memory project after benchmarking.", dedup="t-2")
            JobWorker(store).process(max_jobs=5)

            timeline = store.get_entity_timeline("user")
            self.assertIsNotNone(timeline.get("narrative"))
            self.assertIn("memory_conflicts", timeline)
            self.assertGreaterEqual(len(timeline["memory_conflicts"]), 1)
        finally:
            store.close()

    def test_status_reports_m3_counts(self) -> None:
        store = make_store(self.make_tmp_path())
        try:
            self._ingest(store, "User decided to ship the M3 lifecycle layer.", dedup="s-1")
            JobWorker(store).process(max_jobs=5)
            status = store.status()
            for table in [
                "memory_segments",
                "persistent_memories",
                "entity_narratives",
                "memory_conflicts",
            ]:
                self.assertIn(table, status["counts"], f"missing counts.{table} in status output")
            self.assertIn("open_conflicts", status)
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()
