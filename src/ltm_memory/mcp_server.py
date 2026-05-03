from __future__ import annotations

from typing import Any, Literal

from mcp.server.fastmcp import FastMCP

from .store import MemoryStore
from .worker import JobWorker


ServerMode = Literal["agent", "admin"]


def _with_store(fn):
    store = MemoryStore.open()
    try:
        return fn(store)
    finally:
        store.close()


def create_server(mode: ServerMode = "agent") -> FastMCP:
    """Create an MCP server.

    ``agent`` is the default LLM-facing surface. It intentionally keeps the
    action space small and excludes ingestion/maintenance tools. Automatic
    memory writes should be triggered by the MCP host/client outside model
    tool selection.

    ``admin`` exposes the full developer/maintenance surface, including
    ingestion, background jobs, entity merge, and index management.
    """

    if mode not in {"agent", "admin"}:
        raise ValueError(f"unknown MCP server mode: {mode}")

    mcp = FastMCP(
        "long-term-memory" if mode == "agent" else "long-term-memory-admin",
        instructions=(
            "Local-first event-centric long-term memory server. "
            "SQLite is canonical; LanceDB is an optional retrieval index. "
            f"Mode: {mode}."
        ),
    )

    _register_agent_tools(mcp, include_soft_forget=(mode == "agent"))
    if mode == "admin":
        _register_admin_tools(mcp)
    return mcp


def _register_agent_tools(mcp: FastMCP, *, include_soft_forget: bool) -> None:
    @mcp.tool()
    def recall(
        query: str,
        session_id: str | None = None,
        limit: int = 10,
        limits: dict[str, int] | None = None,
        scope: str = "auto",
        include_session_buffer: bool = True,
        max_response_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Recall observations, events, narratives, memories, and relevant conflicts."""

        return _with_store(
            lambda store: store.recall(
                query,
                session_id=session_id,
                limit=limit,
                limits=limits,
                scope=scope,
                include_session_buffer=include_session_buffer,
                max_response_tokens=max_response_tokens,
            )
        )

    @mcp.tool()
    def get_entity_timeline(
        entity_query: str,
        since: str | None = None,
        until: str | None = None,
        event_types: list[str] | None = None,
        state_types: list[str] | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Return events, roles, states, anchors, narrative, memories, and conflicts for one entity."""

        return _with_store(
            lambda store: store.get_entity_timeline(
                entity_query,
                since=since,
                until=until,
                event_types=event_types,
                state_types=state_types,
                limit=limit,
            )
        )

    if include_soft_forget:
        @mcp.tool()
        def forget(
            target_type: str,
            target_id: str,
            cascade: bool = True,
        ) -> dict[str, Any]:
            """Soft-forget a memory record after the user explicitly asks to forget it."""

            return _with_store(
                lambda store: store.forget(
                    target_type=target_type,
                    target_id=target_id,
                    mode="soft",
                    cascade=cascade,
                )
            )


def _register_admin_tools(mcp: FastMCP) -> None:
    @mcp.tool()
    def ingest_observation(
        content: str,
        source_type: str = "chat",
        session_id: str | None = None,
        source_uri: str | None = None,
        client_dedup_key: str | None = None,
        metadata: dict[str, Any] | None = None,
        role: str = "user",
    ) -> dict[str, Any]:
        """Host-side automatic ingestion endpoint. Prefer client hooks over LLM tool selection."""

        return _with_store(
            lambda store: store.ingest_observation(
                source_type=source_type,
                content=content,
                session_id=session_id,
                source_uri=source_uri,
                client_dedup_key=client_dedup_key,
                metadata=metadata,
                role=role,
            )
        )

    @mcp.tool()
    def search_events(
        query: str | None = None,
        entity_id: str | None = None,
        event_type: str | None = None,
        since: str | None = None,
        until: str | None = None,
        min_importance: float | None = None,
        include_forgotten: bool = False,
        limit: int = 25,
    ) -> list[dict[str, Any]]:
        """Search extracted events using text, entity, type, time, and importance filters."""

        return _with_store(
            lambda store: store.search_events(
                query=query,
                entity_id=entity_id,
                event_type=event_type,
                since=since,
                until=until,
                min_importance=min_importance,
                include_forgotten=include_forgotten,
                limit=limit,
            )
        )

    @mcp.tool()
    def process_background_jobs(max_jobs: int = 50) -> dict[str, Any]:
        """Process pending extraction/indexing jobs."""

        return _with_store(lambda store: JobWorker(store).process(max_jobs=max_jobs))

    @mcp.tool()
    def memory_status() -> dict[str, Any]:
        """Return SQLite/LanceDB paths and record counts."""

        return _with_store(lambda store: store.status())

    @mcp.tool()
    def forget(
        target_type: str,
        target_id: str,
        mode: str = "soft",
        cascade: bool = True,
    ) -> dict[str, Any]:
        """Forget an event, observation, memory segment, persistent memory, or open question."""

        return _with_store(
            lambda store: store.forget(
                target_type=target_type,
                target_id=target_id,
                mode=mode,
                cascade=cascade,
            )
        )

    @mcp.tool()
    def rebuild_index() -> dict[str, Any]:
        """Rebuild the optional LanceDB index from canonical SQLite records."""

        return _with_store(lambda store: JobWorker(store).rebuild_index())

    @mcp.tool()
    def merge_entities(
        source_id: str,
        target_id: str,
        notes: str | None = None,
    ) -> dict[str, Any]:
        """Merge ``source_id`` into ``target_id`` and rewrite all references."""

        return _with_store(
            lambda store: store.merge_entities(
                source_id=source_id,
                target_id=target_id,
                notes=notes,
            )
        )

    @mcp.tool()
    def consolidate_memory(
        tasks: list[str] | None = None,
        scope: str = "auto",
        entity_id: str | None = None,
        segment_id: str | None = None,
    ) -> dict[str, Any]:
        """Run MemoryOS-style consolidation tasks."""

        return _with_store(
            lambda store: store.consolidate_memory(
                tasks=tasks,
                scope=scope,
                entity_id=entity_id,
                segment_id=segment_id,
            )
        )

    @mcp.tool()
    def get_open_questions(
        entity_id: str | None = None,
        status: str | None = None,
        priority: int | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Return forward-falling open questions, filtered by entity/status/priority."""

        return _with_store(
            lambda store: store.get_open_questions(
                entity_id=entity_id,
                status=status,
                priority=priority,
                limit=limit,
            )
        )

    @mcp.tool()
    def get_conflicts(
        entity_id: str | None = None,
        status: str | None = "open",
        conflict_type: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Return memory_conflicts rows. Defaults to open conflicts only."""

        return _with_store(
            lambda store: store.get_conflicts(
                entity_id=entity_id,
                status=status,
                conflict_type=conflict_type,
                limit=limit,
            )
        )

    @mcp.tool()
    def flush_session_buffer(session_id: str) -> dict[str, Any]:
        """Force a SessionBuffer flush checkpoint for the given session_id."""

        return _with_store(lambda store: store.flush_session_buffer(session_id))


def main() -> None:
    create_server("agent").run("stdio")


def admin_main() -> None:
    create_server("admin").run("stdio")


if __name__ == "__main__":
    main()
