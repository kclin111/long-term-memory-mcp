from __future__ import annotations

import argparse
import json
import sys

from .config import Settings
from .store import MemoryStore
from .worker import JobWorker


def _print_json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ltm-memory")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="Initialize the local SQLite store.")
    sub.add_parser("status", help="Show memory store status.")

    ingest = sub.add_parser("ingest", help="Store an observation.")
    ingest.add_argument("content")
    ingest.add_argument("--source-type", default="chat")
    ingest.add_argument("--session-id")
    ingest.add_argument("--source-uri")
    ingest.add_argument("--client-dedup-key")
    ingest.add_argument("--role", default="user")
    ingest.add_argument("--metadata-json", default="{}")

    recall = sub.add_parser("recall", help="Lexically recall observations.")
    recall.add_argument("query")
    recall.add_argument("--session-id")
    recall.add_argument("--limit", type=int, default=10)
    recall.add_argument("--limits-json", default="{}")
    recall.add_argument("--scope", default="auto")
    recall.add_argument("--max-response-tokens", type=int)
    recall.add_argument("--no-session-buffer", action="store_true")

    timeline = sub.add_parser("entity-timeline", help="Show an entity's event/state timeline.")
    timeline.add_argument("entity_query")
    timeline.add_argument("--since")
    timeline.add_argument("--until")
    timeline.add_argument("--event-types", nargs="*")
    timeline.add_argument("--state-types", nargs="*")
    timeline.add_argument("--limit", type=int, default=50)

    jobs = sub.add_parser("process-jobs", help="Process pending background jobs.")
    jobs.add_argument("--max-jobs", type=int, default=50)

    sub.add_parser("rebuild-index", help="Rebuild the optional LanceDB retrieval index from SQLite.")

    merge = sub.add_parser("merge-entities", help="Merge two entities and rewrite references.")
    merge.add_argument("source_id")
    merge.add_argument("target_id")
    merge.add_argument("--notes")

    benchmark = sub.add_parser("benchmark", help="Run the synthetic memory-quality benchmark.")
    benchmark.add_argument(
        "--scenario-file",
        help="Path to a benchmark scenario JSON. Defaults to the bundled M2 baseline.",
    )

    episodic = sub.add_parser(
        "benchmark-episodic",
        help="Run the stricter GSW-style episodic synthetic benchmark.",
    )
    episodic.add_argument(
        "--scenario-file",
        help="Path to a benchmark scenario JSON. Defaults to the bundled episodic hard set.",
    )

    locomo = sub.add_parser(
        "benchmark-locomo",
        help="Run a LoCoMo-style retrieval/evidence benchmark against a local dataset JSON.",
    )
    locomo.add_argument("--dataset", required=True, help="Path to locomo10.json or a compatible sample JSON.")
    locomo.add_argument("--sample-limit", type=int)
    locomo.add_argument("--max-questions", type=int)
    locomo.add_argument("--categories", nargs="*", default=["1", "2", "3", "4"])
    locomo.add_argument(
        "--source-mode",
        default="dialogs",
        help="dialogs, observations, session_summaries, or a + combination such as dialogs+observations.",
    )
    locomo.add_argument("--recall-limit", type=int, default=10)
    locomo.add_argument("--max-jobs", type=int, default=1000)

    consolidate = sub.add_parser("consolidate", help="Run MemoryOS consolidation tasks.")
    consolidate.add_argument(
        "--tasks",
        nargs="*",
        choices=["heat_update", "persistent_memory", "narrative_refresh", "open_question_resolution", "archive"],
    )
    consolidate.add_argument("--scope", default="auto")
    consolidate.add_argument("--entity-id")
    consolidate.add_argument("--segment-id")

    open_q = sub.add_parser("open-questions", help="List forward-falling open questions.")
    open_q.add_argument("--entity-id")
    open_q.add_argument("--status")
    open_q.add_argument("--priority", type=int)
    open_q.add_argument("--limit", type=int, default=50)

    conflicts = sub.add_parser("conflicts", help="List memory conflicts.")
    conflicts.add_argument("--entity-id")
    conflicts.add_argument("--status", default="open")
    conflicts.add_argument("--conflict-type")
    conflicts.add_argument("--limit", type=int, default=50)

    flush = sub.add_parser("flush-session", help="Force a SessionBuffer flush checkpoint.")
    flush.add_argument("session_id")

    forget = sub.add_parser("forget", help="Forget or delete a memory record.")
    forget.add_argument("--type", required=True, dest="target_type")
    forget.add_argument("--id", required=True, dest="target_id")
    forget.add_argument("--mode", choices=["soft", "hard"], default="soft")
    forget.add_argument("--no-cascade", action="store_true")

    search = sub.add_parser("search-events", help="Search extracted events with structured filters.")
    search.add_argument("--query")
    search.add_argument("--entity-id")
    search.add_argument("--event-type")
    search.add_argument("--since")
    search.add_argument("--until")
    search.add_argument("--min-importance", type=float)
    search.add_argument("--include-forgotten", action="store_true")
    search.add_argument("--limit", type=int, default=25)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "benchmark":
        from .benchmark import run_benchmark

        _print_json(run_benchmark(scenario_file=args.scenario_file))
        return 0
    if args.command == "benchmark-episodic":
        from .benchmark import run_episodic_benchmark

        _print_json(run_episodic_benchmark(scenario_file=args.scenario_file))
        return 0
    if args.command == "benchmark-locomo":
        from .benchmark_locomo import run_locomo_benchmark

        _print_json(
            run_locomo_benchmark(
                dataset_file=args.dataset,
                sample_limit=args.sample_limit,
                max_questions=args.max_questions,
                categories=args.categories,
                source_mode=args.source_mode,
                recall_limit=args.recall_limit,
                max_jobs=args.max_jobs,
            )
        )
        return 0

    settings = Settings.from_env()
    store = MemoryStore.open(settings)
    try:
        if args.command == "init":
            store.init()
            _print_json({"status": "ok", "sqlite_path": str(settings.sqlite_path)})
            return 0
        if args.command == "status":
            _print_json(store.status())
            return 0
        if args.command == "ingest":
            try:
                metadata = json.loads(args.metadata_json)
            except json.JSONDecodeError as exc:
                print(f"Invalid --metadata-json: {exc}", file=sys.stderr)
                return 2
            result = store.ingest_observation(
                source_type=args.source_type,
                content=args.content,
                session_id=args.session_id,
                source_uri=args.source_uri,
                client_dedup_key=args.client_dedup_key,
                metadata=metadata,
                role=args.role,
            )
            _print_json(result)
            return 0
        if args.command == "recall":
            try:
                limits = json.loads(args.limits_json)
            except json.JSONDecodeError as exc:
                print(f"Invalid --limits-json: {exc}", file=sys.stderr)
                return 2
            _print_json(
                store.recall(
                    args.query,
                    session_id=args.session_id,
                    limit=args.limit,
                    limits=limits,
                    scope=args.scope,
                    include_session_buffer=not args.no_session_buffer,
                    max_response_tokens=args.max_response_tokens,
                )
            )
            return 0
        if args.command == "entity-timeline":
            _print_json(
                store.get_entity_timeline(
                    args.entity_query,
                    since=args.since,
                    until=args.until,
                    event_types=args.event_types,
                    state_types=args.state_types,
                    limit=args.limit,
                )
            )
            return 0
        if args.command == "process-jobs":
            _print_json(JobWorker(store).process(max_jobs=args.max_jobs))
            return 0
        if args.command == "rebuild-index":
            _print_json(JobWorker(store).rebuild_index())
            return 0
        if args.command == "merge-entities":
            _print_json(
                store.merge_entities(
                    source_id=args.source_id,
                    target_id=args.target_id,
                    notes=args.notes,
                )
            )
            return 0
        if args.command == "consolidate":
            _print_json(
                store.consolidate_memory(
                    tasks=args.tasks,
                    scope=args.scope,
                    entity_id=args.entity_id,
                    segment_id=args.segment_id,
                )
            )
            return 0
        if args.command == "open-questions":
            _print_json(
                store.get_open_questions(
                    entity_id=args.entity_id,
                    status=args.status,
                    priority=args.priority,
                    limit=args.limit,
                )
            )
            return 0
        if args.command == "conflicts":
            _print_json(
                store.get_conflicts(
                    entity_id=args.entity_id,
                    status=args.status,
                    conflict_type=args.conflict_type,
                    limit=args.limit,
                )
            )
            return 0
        if args.command == "flush-session":
            _print_json(store.flush_session_buffer(args.session_id))
            return 0
        if args.command == "forget":
            _print_json(
                store.forget(
                    target_type=args.target_type,
                    target_id=args.target_id,
                    mode=args.mode,
                    cascade=not args.no_cascade,
                )
            )
            return 0
        if args.command == "search-events":
            _print_json(
                store.search_events(
                    query=args.query,
                    entity_id=args.entity_id,
                    event_type=args.event_type,
                    since=args.since,
                    until=args.until,
                    min_importance=args.min_importance,
                    include_forgotten=args.include_forgotten,
                    limit=args.limit,
                )
            )
            return 0
    finally:
        store.close()
    return 1
