from __future__ import annotations

from pathlib import Path
import unittest
from uuid import uuid4

from ltm_memory.config import Settings
from ltm_memory.store import MemoryStore
from ltm_memory.worker import JobWorker


TEST_TMP_ROOT = Path.cwd() / ".codex-tmp" / "test-runs"


def make_store(tmp_path: Path) -> MemoryStore:
    settings = Settings(
        home=tmp_path,
        sqlite_path=tmp_path / "memory.sqlite",
        lancedb_path=tmp_path / "lancedb",
        sqlite_journal_mode="MEMORY",
    )
    return MemoryStore.open(settings)


class MemoryStoreTests(unittest.TestCase):
    def make_tmp_path(self) -> Path:
        TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)
        path = TEST_TMP_ROOT / uuid4().hex
        path.mkdir(parents=True, exist_ok=False)
        return path

    def test_ingest_observation_and_recall(self) -> None:
        store = make_store(self.make_tmp_path())
        try:
            result = store.ingest_observation(
                source_type="chat",
                content="User decided to build the memory server with SQLite and LanceDB.",
                session_id="s1",
                client_dedup_key="turn-1",
            )
            self.assertTrue(result["observation_id"].startswith("obs_"))
            self.assertTrue(result["session_page_id"].startswith("page_"))

            recall = store.recall("SQLite LanceDB", session_id="s1")
            self.assertEqual(len(recall["observations"]), 1)
            self.assertTrue(recall["session_context"][0]["content"].startswith("User decided"))
        finally:
            store.close()

    def test_client_dedup_key_is_idempotent(self) -> None:
        store = make_store(self.make_tmp_path())
        try:
            first = store.ingest_observation(
                source_type="chat",
                content="Remember this once.",
                session_id="s1",
                client_dedup_key="same",
            )
            second = store.ingest_observation(
                source_type="chat",
                content="Remember this once.",
                session_id="s1",
                client_dedup_key="same",
            )
            self.assertEqual(first["observation_id"], second["observation_id"])
            self.assertEqual(second["status"], "deduplicated")
            self.assertEqual(store.status()["counts"]["observations"], 1)
        finally:
            store.close()

    def test_process_jobs_extracts_m2_event_graph(self) -> None:
        store = make_store(self.make_tmp_path())
        try:
            result = store.ingest_observation(
                source_type="chat",
                content="User decided to implement MCP memory with Python, SQLite FTS5, and LanceDB.",
                session_id="s1",
                client_dedup_key="important",
            )
            self.assertEqual(result["status"], "queued")

            processed = JobWorker(store).process(max_jobs=5)
            self.assertEqual(processed["failed"], 0)

            status = store.status()
            self.assertEqual(status["counts"]["events"], 1)
            self.assertGreaterEqual(status["counts"]["entities"], 4)
            self.assertGreaterEqual(status["counts"]["entity_roles"], 1)
            self.assertGreaterEqual(status["counts"]["entity_states"], 1)
            self.assertEqual(status["counts"]["event_actions"], 1)

            recall = store.recall("Python SQLite LanceDB", session_id="s1")
            self.assertEqual(len(recall["events"]), 1)
            self.assertEqual(recall["events"][0]["actions"][0]["action_type"], "decide")
            self.assertIn("SQLite", recall["events"][0]["summary"])
        finally:
            store.close()

    def test_open_question_is_answered_by_later_anchor_evidence(self) -> None:
        store = make_store(self.make_tmp_path())
        try:
            first = store.ingest_observation(
                source_type="chat",
                content="Remember important: Carter Stewart presented the statistics workshop.",
                session_id="s1",
                client_dedup_key="oq-1",
            )
            self.assertEqual(first["status"], "queued")
            JobWorker(store).process(max_jobs=5)
            self.assertGreaterEqual(store.status()["counts"]["open_questions"], 1)

            second = store.ingest_observation(
                source_type="chat",
                content="Remember important: Carter Stewart presented at Metropolitan Museum on 2026-09-22.",
                session_id="s1",
                client_dedup_key="oq-2",
            )
            self.assertEqual(second["status"], "queued")
            JobWorker(store).process(max_jobs=10)

            questions = store.get_entity_timeline("Carter Stewart")["open_questions"]
            self.assertTrue(any(question["status"] == "answered" for question in questions))
            self.assertTrue(any("Metropolitan Museum" in (question["answer_text"] or "") for question in questions))
        finally:
            store.close()

    def test_merge_entities_rewrites_references_and_records_audit(self) -> None:
        store = make_store(self.make_tmp_path())
        try:
            store.ingest_observation(
                source_type="chat",
                content="Remember important: Carter Stewart presented at Metropolitan Museum on 2026-09-22.",
                session_id="s1",
                client_dedup_key="m1",
            )
            JobWorker(store).process(max_jobs=5)

            row = store.conn.execute(
                "SELECT id FROM entities WHERE canonical_name = ?",
                ("Carter Stewart",),
            ).fetchone()
            self.assertIsNotNone(row, "Carter Stewart should have been created")
            source_id = row["id"]

            store.conn.execute(
                """
                INSERT INTO entities(id, entity_type, canonical_name, normalized_name, aliases_json, description, created_at, last_seen_at)
                VALUES ('ent_carter_alias', 'person', 'C. Stewart', 'c. stewart', '[]', '', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')
                """
            )
            store.conn.commit()

            result = store.merge_entities(
                source_id="ent_carter_alias",
                target_id=source_id,
                notes="manual alias",
            )
            self.assertEqual(result["status"], "merged")
            self.assertIn("C. Stewart", result["aliases"])

            self.assertIsNone(
                store.conn.execute(
                    "SELECT id FROM entities WHERE id = ?",
                    ("ent_carter_alias",),
                ).fetchone()
            )

            audit = store.conn.execute(
                "SELECT moved_counts_json FROM entity_merges WHERE source_entity_id = ?",
                ("ent_carter_alias",),
            ).fetchone()
            self.assertIsNotNone(audit)

            timeline = store.get_entity_timeline("Carter Stewart")
            self.assertEqual(timeline["entity"]["id"], source_id)
            self.assertGreaterEqual(len(timeline["events"]), 1)
        finally:
            store.close()

    def test_merge_entities_resolves_event_entity_pk_conflicts(self) -> None:
        store = make_store(self.make_tmp_path())
        try:
            store.ingest_observation(
                source_type="chat",
                content="Remember important: Carter Stewart presented the workshop at Metropolitan Museum.",
                session_id="s1",
                client_dedup_key="m2-1",
            )
            JobWorker(store).process(max_jobs=5)

            target = store.conn.execute(
                "SELECT id FROM entities WHERE canonical_name = ?",
                ("Carter Stewart",),
            ).fetchone()
            self.assertIsNotNone(target)
            target_id = target["id"]

            event_id = store.conn.execute(
                "SELECT event_id FROM event_entities WHERE entity_id = ? LIMIT 1",
                (target_id,),
            ).fetchone()["event_id"]

            store.conn.execute(
                """
                INSERT INTO entities(id, entity_type, canonical_name, normalized_name, aliases_json, description, created_at, last_seen_at)
                VALUES ('ent_dup', 'person', 'Carter S.', 'carter s.', '[]', '', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')
                """
            )
            store.conn.execute(
                """
                INSERT INTO event_entities(event_id, entity_id, role_in_event, confidence, created_at)
                VALUES (?, 'ent_dup', 'mentioned', 0.5, '2026-01-01T00:00:00Z')
                """,
                (event_id,),
            )
            store.conn.commit()

            store.merge_entities(source_id="ent_dup", target_id=target_id)

            target_rows = store.conn.execute(
                """
                SELECT COUNT(*) AS n FROM event_entities
                WHERE event_id = ? AND entity_id = ? AND role_in_event = 'mentioned'
                """,
                (event_id, target_id),
            ).fetchone()
            self.assertEqual(target_rows["n"], 1, "target should keep exactly one mentioned edge")
            source_rows = store.conn.execute(
                "SELECT COUNT(*) AS n FROM event_entities WHERE entity_id = 'ent_dup'"
            ).fetchone()
            self.assertEqual(source_rows["n"], 0, "source rows should be removed")
        finally:
            store.close()

    def test_anchor_propagates_through_shared_entity(self) -> None:
        store = make_store(self.make_tmp_path())
        try:
            store.ingest_observation(
                source_type="chat",
                content="Remember important: The MCP workshop happened on 2026-09-22 at Metropolitan Museum.",
                session_id="s1",
                client_dedup_key="anchor-1",
            )
            JobWorker(store).process(max_jobs=5)

            store.ingest_observation(
                source_type="chat",
                content="Remember important: Carter Stewart presented the MCP workshop.",
                session_id="s1",
                client_dedup_key="anchor-2",
            )
            JobWorker(store).process(max_jobs=10)

            timeline = store.get_entity_timeline("Carter Stewart")
            labels = {anchor["label"] for anchor in timeline["anchors"]}
            self.assertIn("2026-09-22", labels)
            self.assertIn("Metropolitan Museum", labels)
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()
