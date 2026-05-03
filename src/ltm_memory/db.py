from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import sqlite3


def connect(path: Path, journal_mode: str = "WAL") -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    requested = journal_mode.upper()
    if requested != "WAL":
        conn.execute(f"PRAGMA journal_mode={requested}").fetchone()
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn
    try:
        conn.execute("PRAGMA journal_mode=WAL").fetchone()
    except sqlite3.OperationalError:
        # Some sandboxed filesystems do not support SQLite WAL/rollback journals.
        # Keep WAL as the normal path, but allow local smoke tests to run.
        conn.close()
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=MEMORY").fetchone()
        except sqlite3.OperationalError:
            conn.close()
            conn = sqlite3.connect(path)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=OFF").fetchone()
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def execute_script(conn: sqlite3.Connection, statements: str) -> None:
    conn.executescript(statements)


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def rows_to_dicts(rows: Iterable[sqlite3.Row]) -> list[dict]:
    return [dict(row) for row in rows]
