# Long-Term Memory MCP

Local-first, event-centric long-term memory for MCP-compatible agents.

## Quickstart

```powershell
git clone <repo-url>
cd long-term-memory-mcp
python -m venv .venv
.\.venv\Scripts\activate
pip install -e .
Copy-Item .env.example .env
```

Edit `.env`:

```env
LTM_HOME=.ltm-data
LTM_LLM_PROVIDER=openrouter
LTM_LLM_MODEL=openai/gpt-4o-mini
OPENROUTER_API_KEY=sk-or-v1-your-key-here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

Run a zero-cost local smoke test:

```powershell
ltm-memory-smoke
```

For a fuller guide, see `docs/quickstart.md`.

## Install Options

Development checkout:

```powershell
pip install -e .
```

Build and install a wheel locally:

```powershell
python -m pip install build
python -m build
pip install .\dist\long_term_memory_mcp-0.1.0-py3-none-any.whl
```

Install directly from a Git repository:

```powershell
pip install "long-term-memory-mcp @ git+https://github.com/<owner>/<repo>.git"
```

When published to PyPI, the target install command will be:

```powershell
pip install long-term-memory-mcp
```

The implementation starts with a Python/SQLite core:

- SQLite canonical store
- SQLite FTS5 lexical recall
- SessionBuffer / STM
- Observation ingestion with idempotency
- Schema migrations

M1 added:

- Background job processing
- Basic deterministic event extraction
- Conservative entity linking
- Evidence records
- Optional LanceDB adapter

M2 adds the first GSW-style episodic layer:

- `EntityRole` and `EntityState`
- `EventAction`
- `SpatiotemporalAnchor`
- `OpenQuestion` / forward-falling question records
- Anchor coupling across entities in the same event
- Conservative anchor propagation through shared entities
- `get_entity_timeline`

M2.5 sharpens the M2 layer toward "trustworthy":

- `merge_entities` MCP/CLI tool with full reference rewrite + audit log (`entity_merges`)
- Pluggable Operator with an explicit prompt template and deterministic fallback (`LTM_LLM_PROVIDER`)
- `LLMProvider` protocol + `CallableLLMProvider` for testable LLM adapters
- Reproducible synthetic benchmark (PRD §14.3) at `benchmarks/synthetic/m2_baseline.json`
- `ltm-memory benchmark` CLI for regression checks on event recall F1, evidence coverage, and the hallucination proxy

M3 adds the MemoryOS-style lifecycle layer:

- Simplified MTM `MemorySegment` with topic/keywords/entity overlap segmentation,
  visit counting, and the PRD §7.15 heat formula
- `PersistentMemory` extraction triggered when segment heat crosses `tau`
- `EntityNarrative` cache with scope-aware TTL (`global` / `project` / `session` /
  `query_specific`) plus rule-based generation
- `MemoryConflict` detection on contradictory `EntityState` rows that cannot
  pass the §7.6 supersede checklist
- `recall` now returns `entity_narratives`, `persistent_memories`, and active
  `memory_conflicts`, and bumps segment visit counts
- `get_entity_timeline` returns the entity's `global` narrative,
  persistent memories, and active conflicts
- New tools/CLI: `consolidate_memory`, `get_open_questions`, `get_conflicts`,
  `flush_session_buffer`

M3.5 adds lifecycle safety and operational hygiene:

- `forget` for soft/hard forgetting with cascade rules for observations,
  events, persistent memories, memory segments, and open questions
- `search_events` with text/entity/type/time/importance filters
- Job queue crash recovery for stale `running` jobs
- Background job handlers for `delete_lancedb`, narrative invalidation/refresh,
  persistent-memory extraction, and `rebuild_index`
- Embedding model signature tracking; `memory_status` reports when a rebuild is needed

## Local Smoke Test

```powershell
$env:PYTHONPATH='src'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:LTM_SQLITE_JOURNAL_MODE='MEMORY'
$env:LTM_SQLITE_PATH='.codex-tmp\demo\memory.sqlite'
$env:LTM_LANCEDB_PATH='.codex-tmp\demo\lancedb'

python -m ltm_memory init
python -m ltm_memory ingest "User decided to build the memory server with Python, SQLite FTS5, and LanceDB." --session-id demo --client-dedup-key turn-1
python -m ltm_memory process-jobs --max-jobs 5
python -m ltm_memory recall "Python SQLite LanceDB" --session-id demo
python -m ltm_memory entity-timeline user
python -m ltm_memory search-events --query SQLite
python -m ltm_memory status
python -m ltm_memory benchmark
python -m ltm_memory benchmark-episodic
```

After package installation, the console scripts are available too:

```powershell
ltm-memory init
ltm-memory ingest "User decided to build the memory server with Python, SQLite FTS5, and LanceDB." --session-id demo --client-dedup-key turn-1
ltm-memory process-jobs --max-jobs 5
ltm-memory recall "Python SQLite LanceDB" --session-id demo
ltm-memory-smoke
```

Merging an alias into a canonical entity:

```powershell
python -m ltm_memory merge-entities ent_source_id ent_target_id --notes "manual alias"
```

Running consolidation tasks (heat update, persistent-memory promotion,
narrative refresh, open question resolution, narrative archival):

```powershell
python -m ltm_memory consolidate
python -m ltm_memory consolidate --tasks heat_update persistent_memory
python -m ltm_memory open-questions --status open
python -m ltm_memory conflicts --status open
python -m ltm_memory flush-session demo
```

Forgetting or deleting records:

```powershell
python -m ltm_memory forget --type event --id evt_example --mode soft
python -m ltm_memory forget --type observation --id obs_example --mode hard
```

Use `LTM_SQLITE_JOURNAL_MODE=MEMORY` only for constrained sandboxes that cannot use SQLite WAL. Normal local installs should use the default `WAL`.

## MCP Server

There are two MCP entrypoints:

- `ltm-memory-mcp`: agent-facing, small tool surface for normal LLM clients.
- `ltm-memory-admin-mcp`: admin/host-facing, full maintenance and ingestion surface.

Automatic memory ingestion should be a host/client behavior, not an LLM choice.
The MCP server cannot see conversation turns by itself; your host should call
`ingest_observation` after each user/assistant turn, or use the CLI/admin MCP
server from a trusted automation layer. The default agent-facing server does
not expose `ingest_observation`.

For package installs, expose the agent-facing server to the LLM:

```json
{
  "mcpServers": {
    "long-term-memory": {
      "command": "ltm-memory-mcp",
      "args": [],
      "env": {
        "LTM_HOME": ".ltm-data",
        "LTM_LLM_PROVIDER": "openrouter",
        "LTM_LLM_MODEL": "openai/gpt-4o-mini",
        "OPENROUTER_API_KEY": "sk-or-v1-your-key-here",
        "OPENROUTER_BASE_URL": "https://openrouter.ai/api/v1"
      }
    }
  }
}
```

For development from this checkout:

```json
{
  "mcpServers": {
    "long-term-memory": {
      "command": "D:\\nycu lab\\long-term-memory-mcp\\.venv\\Scripts\\python.exe",
      "args": ["-m", "ltm_memory.mcp_server"],
      "env": {
        "PYTHONPATH": "D:\\nycu lab\\long-term-memory-mcp\\src",
        "LTM_SQLITE_PATH": "D:\\nycu lab\\long-term-memory-mcp\\.codex-tmp\\memory.sqlite",
        "LTM_LANCEDB_PATH": "D:\\nycu lab\\long-term-memory-mcp\\.codex-tmp\\lancedb"
      }
    }
  }
}
```

Tools exposed:

- `recall`
- `get_entity_timeline`
- `forget` (soft-forget only)

Admin tools exposed by `ltm-memory-admin-mcp`:

- `ingest_observation`
- `process_background_jobs`
- `memory_status`
- `search_events`
- `rebuild_index`
- `merge_entities`
- `consolidate_memory`
- `get_open_questions`
- `get_conflicts`
- `flush_session_buffer`
- `forget` (soft/hard)

Agent config lives at `docs/mcp-client-config.example.json`.
Trusted admin config lives at `docs/mcp-admin-config.example.json`.

## Operator / LLM Provider

The Operator (ExtractWorker) is pluggable. Defaults to deterministic
extraction so the server runs in fully degraded mode without any LLM
configured. An LLM adapter can be wired in by injecting an
`LLMProvider` (see `src/ltm_memory/operator.py`). A local `.env` file is
loaded automatically when present. Environment knobs:

```text
LTM_LLM_PROVIDER=none | openrouter
LTM_LLM_MODEL=openai/gpt-4o-mini
OPENROUTER_API_KEY=...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
LTM_LLM_TIMEOUT_SECONDS=60
```

`LTM_LLM_PROVIDER=none` is the default. `openrouter` uses OpenRouter's
OpenAI-compatible chat-completions endpoint and asks the model for a JSON
object Operator payload. Malformed or failed model responses fall back to
deterministic extraction and record the fallback reason in
`event.structured_json`.

## Synthetic Benchmark (PRD §14.3)

The bundled `benchmarks/synthetic/m2_baseline.json` exercises:
decision recall with evidence, anchor coupling within an observation,
anchor propagation through a shared entity, forward-falling open
question resolution, multi-event preference recall, MemoryOS segment
heat/narrative caching, and forget-cascade behavior.

```powershell
python -m ltm_memory benchmark
python -m ltm_memory benchmark --scenario-file benchmarks/synthetic/m2_baseline.json
```

The stricter GSW-style synthetic suite adds forward-falling questions,
role/state/action separation, anchor coupling, and narrative reuse:

```powershell
python -m ltm_memory benchmark-episodic
python -m ltm_memory benchmark-episodic --scenario-file benchmarks/synthetic/episodic_hard.json
```

The harness reports per-scenario `event_recall_f1`,
`evidence_coverage`, `hallucination_rate`, and pass/fail counts.

## LoCoMo Benchmark

LoCoMo is the best first external benchmark for this project because it
contains long multi-session conversations, QA annotations, event
summaries, timestamps, and dialog-id evidence. Download the official
`locomo10.json` from `snap-research/locomo`, then run the local retrieval
gate:

```powershell
python -m ltm_memory benchmark-locomo --dataset data\locomo10.json --sample-limit 1
python -m ltm_memory benchmark-locomo --dataset data\locomo10.json --sample-limit 10 --recall-limit 15
```

This harness is intentionally zero-cost: it does not call OpenRouter or
any LLM judge. It measures whether `recall` retrieves the annotated
dialog evidence and answer-bearing text:

- `evidence_recall_at_k`
- `any_evidence_hit_rate`
- `answer_string_hit_rate`
- per-category breakdown for LoCoMo categories 1-4

Use `benchmark-episodic` for GSW-style mechanism regression, and
`benchmark-locomo` for MemoryOS-style long multi-session retrieval. A
future explicit LLM-as-judge mode can be added when you want to spend API
credits on final answer quality.

More details live in `docs/benchmarking.md`.

## Tests

```powershell
$env:PYTHONPATH='src'
$env:PYTHONDONTWRITEBYTECODE='1'
.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```
