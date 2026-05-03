from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import Settings


@dataclass(frozen=True)
class LanceStatus:
    available: bool
    message: str


class LanceMemoryIndex:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        try:
            import lancedb  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on optional package
            self._lancedb = None
            self.status = LanceStatus(False, f"lancedb unavailable: {exc}")
        else:  # pragma: no cover - optional dependency not installed in CI sandbox
            self._lancedb = lancedb
            self.status = LanceStatus(True, "available")

    def upsert(self, records: list[dict[str, Any]]) -> LanceStatus:
        if not records:
            return LanceStatus(self.status.available, "no records")
        if not self._lancedb:
            return self.status

        # LanceDB integration is intentionally kept behind the optional import for M1.
        # Embedding-backed vector schema lands with the embedding provider adapter.
        try:  # pragma: no cover - exercised only when optional package is installed
            db = self._lancedb.connect(str(self.settings.lancedb_path))
            table_name = "memory_index"
            try:
                table = db.open_table(table_name)
                table.add(records)
            except Exception:
                db.create_table(table_name, records)
        except Exception as exc:
            return LanceStatus(False, f"lancedb upsert failed: {exc}")
        return LanceStatus(True, f"upserted {len(records)} records")

    def delete(self, *, record_type: str, sqlite_id: str) -> LanceStatus:
        if not self._lancedb:
            return self.status

        try:  # pragma: no cover - exercised only when optional package is installed
            db = self._lancedb.connect(str(self.settings.lancedb_path))
            try:
                table = db.open_table("memory_index")
            except Exception as exc:
                return LanceStatus(True, f"memory_index table missing; delete skipped: {exc}")
            table.delete(f"record_type = '{record_type}' AND sqlite_id = '{sqlite_id}'")
        except Exception as exc:
            return LanceStatus(False, f"lancedb delete failed: {exc}")
        return LanceStatus(True, f"deleted {record_type}:{sqlite_id}")
