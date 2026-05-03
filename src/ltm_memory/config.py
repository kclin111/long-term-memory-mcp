from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


APP_DIR_NAME = "long-term-memory-mcp"


def _default_home() -> Path:
    if os.name == "nt":
        base = os.environ.get("APPDATA")
        if base:
            return Path(base) / APP_DIR_NAME
    return Path.home() / ".local" / "share" / APP_DIR_NAME


def _expand(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def load_dotenv(path: str | Path = ".env") -> None:
    """Load simple KEY=VALUE pairs from a local .env file.

    Existing environment variables win. This intentionally supports the
    common dotenv subset we need for local MCP development without adding a
    runtime dependency.
    """

    dotenv = Path(path)
    if not dotenv.exists():
        return
    for raw_line in dotenv.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass(frozen=True)
class Settings:
    home: Path
    sqlite_path: Path
    lancedb_path: Path
    session_idle_flush_seconds: int = 900
    max_observation_chars: int = 4000
    importance_promote_threshold: float = 0.65
    novelty_promote_threshold: float = 0.35
    confidence_min_threshold: float = 0.50
    recall_max_response_tokens: int = 8000
    sqlite_journal_mode: str = "WAL"
    job_running_timeout_seconds: int = 300
    embedding_provider: str = "none"
    embedding_model: str = ""

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        home = _expand(os.environ.get("LTM_HOME", _default_home()))
        sqlite_path = _expand(os.environ.get("LTM_SQLITE_PATH", home / "memory.sqlite"))
        lancedb_path = _expand(os.environ.get("LTM_LANCEDB_PATH", home / "lancedb"))
        return cls(
            home=home,
            sqlite_path=sqlite_path,
            lancedb_path=lancedb_path,
            session_idle_flush_seconds=int(os.environ.get("LTM_SESSION_IDLE_FLUSH_SECONDS", "900")),
            max_observation_chars=int(os.environ.get("LTM_MAX_OBSERVATION_CHARS", "4000")),
            importance_promote_threshold=float(os.environ.get("LTM_IMPORTANCE_PROMOTE_THRESHOLD", "0.65")),
            novelty_promote_threshold=float(os.environ.get("LTM_NOVELTY_PROMOTE_THRESHOLD", "0.35")),
            confidence_min_threshold=float(os.environ.get("LTM_CONFIDENCE_MIN_THRESHOLD", "0.50")),
            recall_max_response_tokens=int(os.environ.get("LTM_RECALL_MAX_RESPONSE_TOKENS", "8000")),
            sqlite_journal_mode=os.environ.get("LTM_SQLITE_JOURNAL_MODE", "WAL").upper(),
            job_running_timeout_seconds=int(os.environ.get("LTM_JOB_RUNNING_TIMEOUT_SECONDS", "300")),
            embedding_provider=os.environ.get("LTM_EMBEDDING_PROVIDER", "none").lower(),
            embedding_model=os.environ.get("LTM_EMBEDDING_MODEL", ""),
        )

    @property
    def embedding_signature(self) -> str:
        return f"{self.embedding_provider}:{self.embedding_model}".strip(":")

    def ensure_dirs(self) -> None:
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self.lancedb_path.mkdir(parents=True, exist_ok=True)
