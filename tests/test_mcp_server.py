from __future__ import annotations

import asyncio
import os
from pathlib import Path
from uuid import uuid4
import unittest

from ltm_memory.mcp_server import create_server


TEST_TMP_ROOT = Path.cwd() / ".codex-tmp" / "mcp-test-runs"


class EnvPatch:
    def __init__(self, values: dict[str, str]) -> None:
        self.values = values
        self.old: dict[str, str | None] = {}

    def __enter__(self) -> None:
        for key, value in self.values.items():
            self.old[key] = os.environ.get(key)
            os.environ[key] = value

    def __exit__(self, *args: object) -> None:
        for key, value in self.old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class MCPServerTests(unittest.TestCase):
    def test_mcp_tools_are_registered_and_callable(self) -> None:
        TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)
        root = TEST_TMP_ROOT / uuid4().hex
        root.mkdir(parents=True, exist_ok=False)
        with EnvPatch(
            {
                "LTM_SQLITE_PATH": str(root / "memory.sqlite"),
                "LTM_LANCEDB_PATH": str(root / "lancedb"),
                "LTM_SQLITE_JOURNAL_MODE": "MEMORY",
            }
        ):
            result = asyncio.run(self._call_tools())
        self.assertEqual(result["ingest"]["status"], "queued")
        self.assertEqual(len(result["recall"]["observations"]), 1)
        self.assertEqual(result["jobs"]["failed"], 0)
        self.assertEqual(result["timeline"]["entity"]["canonical_name"], "user")

    async def _call_tools(self) -> dict:
        server = create_server()
        tools = {tool.name for tool in await server.list_tools()}
        self.assertIn("ingest_observation", tools)
        self.assertIn("recall", tools)
        self.assertIn("process_background_jobs", tools)
        self.assertIn("get_entity_timeline", tools)
        self.assertIn("forget", tools)
        self.assertIn("search_events", tools)

        _, ingest = await server.call_tool(
            "ingest_observation",
            {
                "content": "User decided to implement MCP memory with Python, SQLite FTS5, and LanceDB.",
                "session_id": "mcp-test",
                "client_dedup_key": "turn-1",
            },
        )
        _, jobs = await server.call_tool("process_background_jobs", {"max_jobs": 5})
        _, recall = await server.call_tool(
            "recall",
            {
                "query": "Python SQLite LanceDB",
                "session_id": "mcp-test",
                "limits": {"events": 3, "observations": 3},
            },
        )
        _, timeline = await server.call_tool("get_entity_timeline", {"entity_query": "user"})
        _, events = await server.call_tool("search_events", {"query": "SQLite", "limit": 5})
        events = events["result"]
        self.assertGreaterEqual(len(events), 1)
        _, forget = await server.call_tool(
            "forget",
            {
                "target_type": "event",
                "target_id": events[0]["id"],
                "mode": "soft",
            },
        )
        self.assertEqual(forget["target_type"], "event")
        return {"ingest": ingest, "jobs": jobs, "recall": recall, "timeline": timeline}


if __name__ == "__main__":
    unittest.main()
