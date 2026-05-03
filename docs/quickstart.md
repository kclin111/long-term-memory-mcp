# Long-Term Memory MCP Quickstart

This server is local-first. SQLite is the canonical memory store; LanceDB is a rebuildable local index.

## Install From Source

```powershell
git clone <repo-url>
cd long-term-memory-mcp
python -m venv .venv
.\.venv\Scripts\activate
pip install -e .
```

## Configure

```powershell
Copy-Item .env.example .env
```

Edit `.env`:

```env
LTM_HOME=.ltm-data
LTM_LLM_PROVIDER=openrouter
LTM_LLM_MODEL=your-llm-model
OPENROUTER_API_KEY=sk-or-v1-your-key-here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

Use `LTM_LLM_PROVIDER=none` for zero-cost deterministic extraction.

## Smoke Test

```powershell
ltm-memory-smoke
```

Or run the steps manually:

```powershell
ltm-memory init
ltm-memory ingest "User decided to use SQLite as canonical memory and LanceDB as rebuildable index." --session-id demo --client-dedup-key turn-1
ltm-memory process-jobs --max-jobs 5
ltm-memory recall "SQLite LanceDB" --session-id demo
ltm-memory search-events --query SQLite
ltm-memory benchmark
ltm-memory benchmark-episodic
```

For external benchmark setup, see `docs/benchmarking.md`. The first
recommended dataset is LoCoMo:

```powershell
ltm-memory benchmark-locomo --dataset data\locomo10.json --sample-limit 1
```

## MCP Client Config

After package installation, expose the small agent-facing server to the LLM:

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

This server exposes only `recall`, `get_entity_timeline`, and soft `forget`.

Automatic memory ingestion is host-side. The MCP server cannot observe chat
turns by itself, and `ingest_observation` is intentionally not exposed to the
normal LLM tool list. Configure your host/client wrapper to call
`ingest_observation` after each turn through the admin server or CLI.

Trusted automation/admin config is in `docs/mcp-admin-config.example.json`:

```json
{
  "mcpServers": {
    "long-term-memory-admin": {
      "command": "ltm-memory-admin-mcp",
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

For editable checkout development, use:

```json
{
  "command": "D:\\nycu lab\\long-term-memory-mcp\\.venv\\Scripts\\python.exe",
  "args": ["-m", "ltm_memory.mcp_server"],
  "env": {
    "PYTHONPATH": "D:\\nycu lab\\long-term-memory-mcp\\src",
    "LTM_HOME": "D:\\nycu lab\\long-term-memory-mcp\\.ltm-data"
  }
}
```

## Common Problems

- `OPENROUTER_API_KEY is required`: copy `.env.example` to `.env` and fill the key.
- `not a valid model ID`: set `LTM_LLM_MODEL` to a real OpenRouter model ID, for example `openai/gpt-4o-mini`.
- The Operator says `operator_fallback_used=true`: the server kept working, but LLM extraction failed and deterministic extraction was used. Check `operator_fallback_reason` in event `structured_json`.
- SQLite WAL fails in a restricted sandbox: set `LTM_SQLITE_JOURNAL_MODE=MEMORY` only for that environment.
