from __future__ import annotations

from pathlib import Path
import json
import unittest
import warnings
from uuid import uuid4

from ltm_memory.config import Settings
from ltm_memory.store import MemoryStore
from ltm_memory.worker import JobWorker


TEST_TMP_ROOT = Path.cwd() / ".codex-tmp" / "test-m35"


def make_settings(tmp_path: Path, *, embedding_model: str = "") -> Settings:
    return Settings(
        home=tmp_path,
        sqlite_path=tmp_path / "memory.sqlite",
        lancedb_path=tmp_path / "lancedb",
        sqlite_journal_mode="MEMORY",
        embedding_provider="test",
        embedding_model=embedding_model,
        job_running_timeout_seconds=1,
    )


def make_store(tmp_path: Path, *, embedding_model: str = "") -> MemoryStore:
    return MemoryStore.open(make_settings(tmp_path, embedding_model=embedding_model))


class M35LifecycleJobsTests(unittest.TestCase):
    def make_tmp_path(self) -> Path:
        TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)
        path = TEST_TMP_ROOT / uuid4().hex
        path.mkdir(parents=True, exist_ok=False)
        return path

    def _extract_one_event(self, store: MemoryStore, content: str, *, dedup: str, session: str = "m35") -> str:
        result = store.ingest_observation(
            source_type="chat",
            content=content,
            session_id=session,
            client_dedup_key=dedup,
        )
        self.assertEqual(result["status"], "queued")
        processed = JobWorker(store).process(max_jobs=10)
        self.assertEqual(processed["failed"], 0)
        row = store.conn.execute(
            """
            SELECT e.id
            FROM events e
            JOIN evidence ev ON ev.event_id = e.id
            WHERE ev.observation_id = ?
            """,
            (result["observation_id"],),
        ).fetchone()
        self.assertIsNotNone(row)
        return row["id"]

    def test_forget_event_soft_marks_narratives_stale_and_recomputes_persistent_memory(self) -> None:
        store = make_store(self.make_tmp_path())
        try:
            event_id = self._extract_one_event(
                store,
                "User decided important: Python and SQLite remain canonical for the memory server.",
                dedup="soft-1",
            )
            user = store.conn.execute("SELECT id FROM entities WHERE canonical_name = 'user'").fetchone()
            self.assertIsNotNone(user)

            timeline = store.get_entity_timeline("user")
            self.assertIsNotNone(timeline["narrative"])
            pm_id = "pm_test_soft"
            store.conn.execute(
                """
                INSERT INTO persistent_memories(
                  id, memory_type, content, entity_ids_json, supporting_event_ids_json,
                  confidence, stability, status, reinforcement_count,
                  created_at, updated_at
                ) VALUES (?, 'design_decision', 'Use Python and SQLite', ?, ?, 0.8, 0.6, 'active', 1, ?, ?)
                """,
                (
                    pm_id,
                    json.dumps([user["id"]]),
                    json.dumps([event_id]),
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T00:00:00Z",
                ),
            )
            store.conn.commit()

            report = store.forget(target_type="event", target_id=event_id, mode="soft", cascade=True)
            self.assertTrue(any(action["action"] == "event_marked_forgotten" for action in report["actions"]))
            event = store.conn.execute("SELECT status FROM events WHERE id = ?", (event_id,)).fetchone()
            self.assertEqual(event["status"], "forgotten")
            narrative = store.conn.execute(
                "SELECT status FROM entity_narratives WHERE entity_id = ?",
                (user["id"],),
            ).fetchone()
            self.assertEqual(narrative["status"], "stale")
            pm = store.conn.execute(
                "SELECT status, supporting_event_ids_json FROM persistent_memories WHERE id = ?",
                (pm_id,),
            ).fetchone()
            self.assertEqual(pm["status"], "forgotten")
            self.assertEqual(json.loads(pm["supporting_event_ids_json"]), [])
        finally:
            store.close()

    def test_forget_event_hard_queues_and_processes_delete_lancedb_job(self) -> None:
        store = make_store(self.make_tmp_path())
        try:
            event_id = self._extract_one_event(
                store,
                "User decided important: LanceDB index deletion should be asynchronous.",
                dedup="hard-1",
            )

            report = store.forget(target_type="event", target_id=event_id, mode="hard", cascade=True)
            self.assertTrue(any(action["action"] == "event_deleted" for action in report["actions"]))
            self.assertIsNone(store.conn.execute("SELECT id FROM events WHERE id = ?", (event_id,)).fetchone())
            job = store.conn.execute(
                "SELECT id, status FROM index_jobs WHERE job_type = 'delete_lancedb' AND payload_json LIKE ?",
                (f"%{event_id}%",),
            ).fetchone()
            self.assertIsNotNone(job)
            self.assertEqual(job["status"], "pending")

            processed = JobWorker(store).process(max_jobs=10)
            self.assertEqual(processed["failed"], 0)
            done = store.conn.execute("SELECT status FROM index_jobs WHERE id = ?", (job["id"],)).fetchone()
            self.assertEqual(done["status"], "completed")
        finally:
            store.close()

    def test_forget_observation_cascades_to_derived_event(self) -> None:
        store = make_store(self.make_tmp_path())
        try:
            result = store.ingest_observation(
                source_type="chat",
                content="User decided important: observation forget should cascade to its event.",
                session_id="obs-forget",
                client_dedup_key="obs-1",
            )
            JobWorker(store).process(max_jobs=5)
            event_id = store.conn.execute(
                "SELECT event_id FROM evidence WHERE observation_id = ?",
                (result["observation_id"],),
            ).fetchone()["event_id"]

            report = store.forget(
                target_type="observation",
                target_id=result["observation_id"],
                mode="soft",
                cascade=True,
            )
            self.assertTrue(any(action["action"] == "observation_marked_forgotten" for action in report["actions"]))
            evidence = store.conn.execute(
                "SELECT status FROM evidence WHERE observation_id = ?",
                (result["observation_id"],),
            ).fetchone()
            event = store.conn.execute("SELECT status FROM events WHERE id = ?", (event_id,)).fetchone()
            self.assertEqual(evidence["status"], "stale")
            self.assertEqual(event["status"], "forgotten")
        finally:
            store.close()

    def test_search_events_filters_by_text_entity_and_time_range(self) -> None:
        store = make_store(self.make_tmp_path())
        try:
            first = self._extract_one_event(
                store,
                "Remember important: Carter Stewart presented the search workshop.",
                dedup="search-1",
            )
            second = self._extract_one_event(
                store,
                "User decided important: SQLite remains canonical.",
                dedup="search-2",
            )
            store.conn.execute(
                "UPDATE events SET observed_at = '2026-01-01T00:00:00Z' WHERE id = ?",
                (first,),
            )
            store.conn.execute(
                "UPDATE events SET observed_at = '2026-02-01T00:00:00Z' WHERE id = ?",
                (second,),
            )
            store.conn.commit()
            carter = store.conn.execute(
                "SELECT id FROM entities WHERE canonical_name = ?",
                ("Carter Stewart",),
            ).fetchone()
            self.assertIsNotNone(carter)

            text_results = store.search_events(query="SQLite")
            self.assertEqual([event["id"] for event in text_results], [second])

            entity_results = store.search_events(entity_id=carter["id"])
            self.assertEqual([event["id"] for event in entity_results], [first])

            ranged = store.search_events(since="2026-01-15T00:00:00Z", until="2026-02-15T00:00:00Z")
            self.assertEqual([event["id"] for event in ranged], [second])
        finally:
            store.close()

    def test_running_job_timeout_is_recovered_before_processing(self) -> None:
        store = make_store(self.make_tmp_path())
        try:
            store.init()
            store.conn.execute(
                """
                INSERT INTO index_jobs(
                  id, idempotency_key, job_type, payload_json, status,
                  retry_count, max_retries, scheduled_at, started_at, created_at
                ) VALUES (
                  'job_stale', 'job_stale', 'rebuild_index', '{}', 'running',
                  0, 3, '2000-01-01T00:00:00Z', '2000-01-01T00:00:00Z', '2000-01-01T00:00:00Z'
                )
                """
            )
            store.conn.commit()

            processed = JobWorker(store).process(max_jobs=1)
            self.assertEqual(processed["recovered_running_jobs"], 1)
            self.assertEqual(processed["failed"], 0)
            row = store.conn.execute("SELECT status FROM index_jobs WHERE id = 'job_stale'").fetchone()
            self.assertEqual(row["status"], "completed")
        finally:
            store.close()

    def test_embedding_signature_change_is_reported_until_rebuild_index(self) -> None:
        root = self.make_tmp_path()
        first = make_store(root, embedding_model="small")
        try:
            first.init()
            self.assertFalse(first.status()["embedding_signature_change_detected"])
        finally:
            first.close()

        second = make_store(root, embedding_model="large")
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                status = second.status()
            self.assertTrue(status["embedding_signature_change_detected"])
            self.assertTrue(any("Embedding model signature changed" in str(w.message) for w in caught))

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                JobWorker(second).rebuild_index()
            self.assertFalse(second.status()["embedding_signature_change_detected"])
        finally:
            second.close()


if __name__ == "__main__":
    unittest.main()
