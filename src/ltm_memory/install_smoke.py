from __future__ import annotations

from pathlib import Path
from uuid import uuid4
import json
import os
import shutil

from .config import Settings
from .store import MemoryStore
from .worker import JobWorker


def run_smoke(*, keep_data: bool = False) -> dict:
    """Run a zero-cost local smoke test against the installed package."""

    root = Path.cwd() / ".codex-tmp" / "install-smoke" / uuid4().hex
    root.mkdir(parents=True, exist_ok=False)
    old_provider = os.environ.get("LTM_LLM_PROVIDER")
    os.environ["LTM_LLM_PROVIDER"] = "none"

    settings = Settings(
        home=root,
        sqlite_path=root / "memory.sqlite",
        lancedb_path=root / "lancedb",
        sqlite_journal_mode=os.environ.get("LTM_SQLITE_JOURNAL_MODE", "MEMORY"),
        embedding_provider="none",
    )
    store = MemoryStore.open(settings)
    try:
        ingest = store.ingest_observation(
            source_type="chat",
            content="User decided important: install smoke should verify SQLite, jobs, events, and recall.",
            session_id="install-smoke",
            client_dedup_key="install-smoke-1",
        )
        jobs = JobWorker(store).process(max_jobs=5)
        recall = store.recall("SQLite jobs events recall", session_id="install-smoke")
        status = store.status()
        ok = (
            ingest["status"] == "queued"
            and jobs["failed"] == 0
            and status["counts"]["events"] >= 1
            and len(recall["events"]) >= 1
        )
        return {
            "ok": ok,
            "data_dir": str(root),
            "kept_data": keep_data,
            "ingest_status": ingest["status"],
            "jobs_failed": jobs["failed"],
            "event_count": status["counts"]["events"],
            "recall_events": len(recall["events"]),
        }
    finally:
        store.close()
        if old_provider is None:
            os.environ.pop("LTM_LLM_PROVIDER", None)
        else:
            os.environ["LTM_LLM_PROVIDER"] = old_provider
        if not keep_data:
            shutil.rmtree(root, ignore_errors=True)


def main() -> int:
    keep_data = os.environ.get("LTM_SMOKE_KEEP_DATA", "").lower() in {"1", "true", "yes"}
    result = run_smoke(keep_data=keep_data)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
