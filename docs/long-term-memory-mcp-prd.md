# Long-Term Memory MCP Server PRD

Version: v0.3

Revision focus:

- Incorporate third-party review feedback.
- Promote GSW mechanisms from future ideas into core architecture.
- Make MemoryOS STM/MTM/LPM lifecycle explicit.
- Keep SQLite + LanceDB as the default local-first storage architecture.
- Clarify MVP concurrency, anchor coupling, recall contract, and release milestones.
- Clarify transaction boundaries, lazy idle flush, timestamp policy, chunking, rebuild semantics, and M0 implementation defaults.

## 1. 背景與目標

本專案要實作一個開源的 Long-Term Memory MCP Server，提供 AI Agent 跨 session、跨文件、跨任務的長期記憶能力。

系統設計參考兩篇論文：

- `Memory OS of AI Agent`：提供記憶生命週期管理概念，包含 STM、MTM、LPM、segment/page、heat-based promotion/eviction。
- `Beyond Fact Retrieval: Episodic Memory for RAG with Generative Semantic Workspaces`：提供事件中心的 episodic memory 表示法，包含 Operator、Reconciler、entity timeline、role/state/time/space grounding。

本系統的核心定位是：

> Event-Centric Long-Term Memory MCP Server

也就是以事件作為主要記憶單位，persona、facts、preferences、project knowledge 都是從事件時間線中萃取出的衍生記憶。

更精確地說，本系統不是單純的 `text + embedding` RAG，也不是只做 event extraction。它應該實作一個簡化版 episodic workspace：

- MemoryOS 負責記憶生命週期：STM、MTM、LPM、heat-based promotion/eviction。
- GSW 負責 episodic representation：Operator、Reconciler、role/state/action、spatiotemporal coupling、forward-falling questions、entity-level narrative。

## 2. Product Goals

1. 提供 MCP-compatible 的長期記憶能力，讓支援 MCP 的 agent/client 可以自動寫入與查詢記憶。
2. 支援事件中心記憶，而不只是傳統 `text + embedding` 的 RAG 記憶。
3. 支援 personal assistant memory 與 project/document memory，第一版偏重事件與專案脈絡。
4. 預設 local-first、zero-config，不要求使用者部署資料庫服務。
5. 所有重要記憶都可追溯 evidence，避免不可驗證的幻覺式記憶。
6. SQLite 作為 canonical store，LanceDB 作為可重建的 retrieval index。
7. 支援 SessionBuffer/STM，讓 MCP recall 能快速取得最近對話脈絡。
8. 支援 unresolved questions / forward-falling questions，使後續 observation 能補完早期未解的 episodic memory。
9. 支援 entity-level narrative 作為 recall 的主要 payload，而不只返回 raw chunks。

## 3. Non-Goals

第一版不做：

- 不預設支援 Neo4j、Postgres、Qdrant 等外部服務。
- 不要求使用者部署 Docker 或資料庫 server。
- 不追求完整 GSW 論文等級的角色/狀態/時間/空間推理。
- 不把所有對話永久保存成長期記憶。
- 不綁定特定 LLM provider。
- 不做雲端同步或多使用者協作權限系統。
- 不在 MVP 自動解決所有 memory conflict；MVP 至少要偵測並標記 conflict。

## 4. Core Design Principles

### 4.1 Event Is the Source of Truth

系統不應以 persona profile 或 vector chunks 作為主要記憶單位，而應以事件為中心。

資料流：

```text
SessionBuffer / STM
-> Observation
-> Event
-> Entity Role/State Timeline
-> Entity Narrative
-> Persistent Memory
```

### 4.2 Structured Envelope + Progressive Enrichment

Event 不強制一開始就完整具備 actor/action/object/time/space。

初始寫入只要求：

- summary
- event_type
- entities
- evidence_ids
- observed_at
- importance
- confidence

後續 enrichment 再補：

- actors
- actions
- objects
- spatiotemporal_anchors
- entity_roles
- entity_states
- relations
- causal_links
- unresolved_questions

### 4.3 Automatic Ingestion + Selective Promotion

系統應支援自動記憶，但不是所有輸入都永久保存。

流程：

```text
Conversation / document / tool output
-> SessionBuffer 更新
-> Observation 寫入 SQLite
-> Importance + novelty 判斷
-> Index/extraction jobs 入隊
-> Async Operator extraction
-> Async Reconciler update
-> Entity narrative / persistent memory consolidation
```

### 4.4 SQLite Is Truth, LanceDB Is Index

SQLite 保存所有不可丟失的 canonical data。

LanceDB 保存 semantic/hybrid retrieval index。LanceDB 可刪除、可重建，不應成為唯一資料來源。

### 4.5 Forward-Falling Questions Are First-Class

GSW 的核心能力之一是記住「目前還無法回答，但未來可能被後續 context 解答」的問題。

因此 unresolved questions 不應只存在 `event.structured_json`，而應是一級資料表。每次新 observation/event 進來時，系統應嘗試用新 evidence 解答既有 open questions。

### 4.6 Role and State Are Separate

Role 與 State 不應混在同一個 `state_type`。

Role 描述 actor 在情境中的功能與行動分布，例如 `presenter`、`researcher`、`project_owner`。

State 描述 actor 在某個 role 下的條件或狀態，例如 `nervous`、`captured`、`in_design_phase`。

在 schema 上：

```text
EntityRole 1 -> many EntityState
EntityState.role_id nullable
```

`role_id` 可為 null，以支援全域偏好、專案狀態等不依附特定 role 的 state。

### 4.7 Spatiotemporal Anchors Are Shared Nodes

時間與空間不應只塞在 event JSON 裡。系統應把 time/space 建模為可共享的 anchor node，讓同一時空中的多個 entity/event 可以共同指向同一個 spatiotemporal anchor。

這是為了支援 GSW 的 space/time coupling：當某段 context 推斷出地點或時間時，所有被判定共享同一情境的 entities 都能繼承該 anchor。

Initial coupling rule:

- Entities extracted from the same observation or same Operator chunk share a candidate anchor when the chunk contains an explicit or strongly implied time/place.
- Candidate anchors become confirmed anchors only after explicit textual evidence or Reconciler-level corroboration from related events/entities.
- Same dialogue chain alone is not enough to auto-confirm shared anchors; it may only raise candidate confidence.
- Anchor propagation should happen in the Reconciler / IntegrateWorker, not in the local Operator proposal.

### 4.8 MVP Concurrency Model

MVP uses a single-writer integration model.

```text
ExtractWorker jobs may run concurrently.
IntegrateWorker jobs must run serially.
All read-modify-write updates to entities, states, anchors, open questions, segments, and conflicts happen inside one SQLite transaction.
```

Rationale:

- SQLite WAL allows concurrent readers, but only one writer.
- Entity timeline updates require application-level consistency across multiple statements.
- Single-writer IntegrateWorker is simpler and safer than per-entity advisory locks for MVP.

Future versions may introduce per-entity locks or partitioned workers after correctness tests exist.

## 5. System Architecture

```text
MCP Client / Agent
        |
        v
Long-Term Memory MCP Server
        |
        +-- SessionBuffer / STM
        |     +-- FIFO dialogue pages
        |     +-- dialogue chain metadata
        |     +-- background context for Operator
        |
        +-- Ingestion Pipeline
        |     +-- Observation writer
        |     +-- Importance scorer
        |     +-- Novelty scorer
        |     +-- Job enqueue
        |
        +-- Async Memory Worker
        |     +-- Operator / ExtractWorker
        |     |     +-- Local event extraction
        |     |     +-- Local role/state/action extraction
        |     |     +-- Local spatiotemporal anchor proposals
        |     |     +-- Local open question proposals
        |     |
        |     +-- Reconciler / IntegrateWorker
        |     |     +-- Entity resolution
        |     |     +-- Anchor propagation
        |     |     +-- OpenQuestion resolution
        |     |     +-- Timeline update
        |     |
        |     +-- Conflict detector
        |     +-- MTM segment updater
        |     +-- LanceDB indexer
        |
        +-- Recall Pipeline
        |     +-- Query planner
        |     +-- Entity extraction
        |     +-- LanceDB candidate retrieval
        |     +-- SQLite graph/timeline expansion
        |     +-- Entity narrative generation/cache
        |     +-- Rerank / compression
        |
        +-- Consolidation Pipeline
        |     +-- MTM segment assignment
        |     +-- Segment summarization
        |     +-- Entity narrative refresh
        |     +-- Persistent memory extraction
        |     +-- Heat update
        |     +-- Index jobs
        |
        +-- SQLite canonical store
        |
        +-- LanceDB retrieval index
```

## 6. Storage Decision

### 6.1 Default Local Storage

Linux/macOS:

```text
~/.local/share/long-term-memory-mcp/
  memory.sqlite
  lancedb/
  attachments/
  exports/
```

Windows:

```text
%APPDATA%/long-term-memory-mcp/
  memory.sqlite
  lancedb/
  attachments/
  exports/
```

可透過環境變數覆寫：

```text
LTM_HOME
LTM_SQLITE_PATH
LTM_LANCEDB_PATH
```

Path precedence:

```text
LTM_SQLITE_PATH overrides LTM_HOME for SQLite.
LTM_LANCEDB_PATH overrides LTM_HOME for LanceDB.
If neither override is set, both are derived from LTM_HOME.
```

Timestamp policy:

```text
All timestamps must be stored as ISO 8601 UTC strings.
Local time conversion is the client's responsibility.
```

### 6.2 SQLite Responsibilities

SQLite 存：

- observations
- session_pages
- dialogue_chains
- events
- entities
- event_entities
- entity_roles
- entity_states
- event_actions
- relations
- evidence
- spatiotemporal_anchors
- event_anchors
- entity_anchors
- anchor_relations
- open_questions
- question_answers
- entity_narratives
- memory_segments
- persistent_memories
- memory_conflicts
- index_jobs
- settings / metadata

Settings table:

```text
settings(
  key text primary key,
  value_json text not null,
  updated_at datetime not null
)
```

Reserved settings keys:

```text
schema_version
last_consolidation_at
daily_llm_cost_usd
last_index_rebuild_at
```

### 6.3 LanceDB Responsibilities

LanceDB 存 retrieval index：

- record_id
- record_type
- text
- vector
- metadata
- entity_ids
- event_type
- importance
- confidence
- timestamps

第一版可先使用單一 table：

```text
memory_index
```

`record_type` 初始值：

```text
event
entity
entity_narrative
persistent_memory
observation
open_question
memory_segment
```

未來可拆：

```text
event_index
entity_index
entity_narrative_index
persistent_memory_index
observation_index
memory_segment_index
```

LanceDB entry 必須包含 SQLite canonical id：

```text
sqlite_table
sqlite_id
record_type
```

SQLite 不應依賴 LanceDB id。LanceDB 損毀或刪除時，必須可由 SQLite 重新建立。

For `record_type = memory_segment`, `metadata` should include at least:

```text
topic
lifecycle
heat
visit_count
entity_ids
event_ids
last_accessed_at
```

## 7. Memory Layers

### 7.1 SessionBuffer / STM

SessionBuffer 是 MemoryOS STM 在 MCP server 裡的實作。它是 in-memory FIFO buffer，保存最近 N 個 dialogue pages，供低延遲 recall 與 event extraction background context 使用。

Defaults:

```text
max_pages = 7 to 10
scope = per session_id
storage = in-memory, mirrored to SQLite session_pages on write
```

Dialogue page:

```text
id
session_id
role
content
timestamp
turn_index
metadata_json
```

Dialogue chain metadata:

```text
page_id
chain_id
previous_page_id
continuity_score
chain_summary
topic_shift_detected
```

Usage:

- `recall` should always include recent SessionBuffer context before hitting long-term memory.
- Operator extraction should receive SessionBuffer chain summary as background context.
- SessionBuffer is not LPM. It can expire without deleting canonical observations.
- Every dialogue page should be mirrored to SQLite `session_pages` when written, so process crashes do not erase the evidence trail.
- In-memory SessionBuffer may evict old pages after FIFO overflow; SQLite mirror should remain until normal retention/forget policies apply.
- Automatic flush should run on buffer overflow, explicit `flush_session_buffer`, and session idle timeout.
- Idle timeout flush is lazy in event-driven/manual worker mode: before `ingest_observation`, `recall`, or `process_background_jobs` proceeds, sessions whose `last_activity` exceeds `LTM_SESSION_IDLE_FLUSH_SECONDS` should be flushed.

Suggested setting:

```text
LTM_SESSION_IDLE_FLUSH_SECONDS=900
```

### 7.2 Observation

不可變的原始證據層。

Examples:

- 對話訊息
- 文件段落
- tool result
- manual note
- 使用者修正

Suggested fields:

```text
id
client_dedup_key
source_type
source_uri
content
metadata_json
observed_at
session_id
created_at
```

Idempotency:

- `client_dedup_key` is optional.
- If provided, `(source_type, session_id, client_dedup_key)` should be unique.
- Replayed observations with the same dedup key should return the existing observation id rather than creating duplicates.

Long observation policy:

- `ingest_observation` should accept long content, but ExtractWorker should chunk before Operator extraction.
- MVP chunking may use sentence-window or paragraph-window chunking.
- Suggested default:

```text
LTM_MAX_OBSERVATION_CHARS=4000
```

### 7.3 Event

主要記憶單位。代表有意義的狀態變化、決策、偏好、任務進展、重要事實、文件主張或可被未來查詢引用的經驗。

Event 採 partial structured schema。初始 event 不必完整具備 actor/action/object/time/space，但必須保留未來 enrichment 的欄位。

`structured_json` is a staging area for raw extraction output. After reconciliation, actions should be reified into `event_actions`, anchors into `spatiotemporal_anchors`, roles into `entity_roles`, and states into `entity_states`. Long-lived canonical data should not live only inside `structured_json`.

Suggested fields:

```text
id
event_type
summary
structured_json
observed_at
occurred_at
importance
novelty
confidence
status
created_at
updated_at
```

Initial event types:

```text
decision
preference
correction
task_progress
project_state
document_claim
fact
interaction
error_workaround
commitment
state_change
```

Status:

```text
observed
extracted
reconciled
conflicting
archived
forgotten
```

### 7.4 Entity

記憶中可被追蹤的主體。

Examples:

- user
- assistant
- project
- document
- concept
- task
- tool
- organization
- place
- event
- time

Suggested fields:

```text
id
entity_type
canonical_name
aliases_json
description
created_at
last_seen_at
```

MVP entity resolution should be conservative:

- Match by `canonical_name + entity_type`.
- Match by explicit aliases.
- Support reserved aliases such as `user`, `assistant`, `current_project`.
- Use SessionBuffer for simple pronoun/self references.
- Low-confidence matches must create candidate links instead of auto-merging entities.
- Normalize entity names before comparison: trim whitespace, Unicode NFC normalization, and lowercase for matching while preserving display casing.

### 7.5 EntityRole

Role 描述 actor 在情境中的功能與行動分布，應與 state 分開。

Suggested fields:

```text
id
entity_id
role_label
description
event_id
valid_from
valid_to
confidence
created_at
```

Examples:

```text
Carter Stewart -> presenter
user -> project_owner
long-term-memory-mcp -> software_project
Operator -> semantic_extractor
```

### 7.6 EntityState

State 描述 entity 在某個 role 下的狀態、限制、偏好、目標或事實。不得只覆寫最新值，應保留 temporal versioning。

`role_id` 可為 null。若 state 依附特定 role，應填入 `role_id`。

Suggested fields:

```text
id
entity_id
role_id
state_type
value
valid_from
valid_to
confidence
event_id
status
created_at
```

Initial state types:

```text
status
preference
trait
fact
goal
constraint
decision
emotion
capability
```

Status:

```text
active
superseded
conflicting
deprecated
forgotten
```

State update rule:

- If new state clearly supersedes an old active state, set old `valid_to` and mark old state as `superseded`.
- If new state contradicts an old active state and timing/scope does not explain the change, create `memory_conflicts`.
- MVP may mark conflicts without resolving them.

Supersede checklist:

- same `entity_id`
- same `state_type`
- same scope, where scope means same `role_id`, same project/entity context, or same spatiotemporal anchor when applicable
- new state is newer by `valid_from` or `observed_at`
- new confidence >= old confidence, or the new state has explicit user correction evidence

If these are not all true, create a conflict instead of destructive overwrite.

### 7.7 EventAction / Verb

GSW treats verbs/actions as causal certificates connecting role/state transitions. MVP should model actions as first-class records, even if causal valence is initially lightweight.

Suggested fields:

```text
id
event_id
actor_entity_id
verb
object_entity_id
object_value
role_id
state_before_id
state_after_id
valence_json
confidence
evidence_id
created_at
```

Examples:

```text
user prefers automatic memory
project adopts SQLite + LanceDB
Operator extracts actors/roles/states/actions/time/space
Reconciler resolves open question
```

### 7.8 Relation

Entity 與 entity/value 之間的關係。

Suggested fields:

```text
id
subject_entity_id
predicate
object_entity_id
object_value
event_id
confidence
valid_from
valid_to
created_at
```

Examples:

```text
long-term-memory-mcp uses MemoryOS lifecycle
long-term-memory-mcp uses GSW episodic representation
Operator extracts roles/states/actions/time/space
user prefers event-centric memory
```

### 7.9 SpatiotemporalAnchor

時間與空間是 GSW 的共享節點，不應只存在 event JSON。

Suggested fields:

```text
id
anchor_type
label
normalized_value
start_time
end_time
location_name
geo_json
granularity
confidence
created_at
```

Granularity examples:

```text
temporal: year | month | day | hour | minute | second | range | unknown
spatial: country | region | city | venue | room | coordinate | unknown
```

Anchor types:

```text
temporal
spatial
spatiotemporal
```

Link tables:

```text
event_anchors(event_id, anchor_id, confidence, evidence_id)
entity_anchors(entity_id, anchor_id, event_id, confidence, evidence_id)
anchor_relations(subject_anchor_id, predicate, object_anchor_id, confidence)
```

Initial anchor relations:

```text
same_as
subsumes
near
before
after
overlaps
```

Suggested confidence policy:

- `same_as` should generally require confidence >= 0.90.
- `subsumes` should generally require confidence >= 0.75.
- `near`, `before`, `after`, and `overlaps` may be stored with lower confidence but should not trigger automatic anchor merge.

### 7.10 OpenQuestion / Forward-Falling Question

OpenQuestion 是 GSW forward-falling questions 的一級實體。它表示「根據目前 memory，系統知道有一個未解問題，後續 context 可能解答它」。

Suggested fields:

```text
id
question_text
question_type
origin_event_id
subject_entity_id
related_entity_ids_json
dedup_key
status
priority
created_at
answered_at
```

`priority` is an integer enum:

```text
high = 3
medium = 2
low = 1
```

Status:

```text
open
answered
obsolete
conflicting
forgotten
```

QuestionAnswer:

```text
id
question_id
answer_text
answer_entity_id
answer_event_id
evidence_id
confidence
created_at
```

During async ingestion, new events should be checked against open questions with matching entities, roles, actions, or anchors.

MVP dedup rule:

```text
dedup_key = hash(subject_entity_id, question_type, sorted(related_entity_ids), normalized_anchor_scope)
```

If two generated questions have the same dedup key, keep the older question and append the new event as supporting context.

MVP priority rule:

- High: question blocks event grounding, such as missing actor, location, time, or object.
- Medium: question improves narrative detail but does not block grounding.
- Low: speculative future/prototypical question.

Answer matching:

- First filter by subject/related entities and anchor overlap.
- Then use lexical/embedding similarity.
- If configured, use LLM judgment to validate the answer with evidence.

### 7.11 EntityNarrative

EntityNarrative 是 GSW entity-level chronological summary 的對應物。它是 recall 的主要 payload，可 cache、可重建。

Suggested fields:

```text
id
entity_id
scope
query_hash
content
supporting_event_ids_json
supporting_anchor_ids_json
generated_at
expires_at
confidence
```

Scopes:

```text
global
project
session
query_specific
```

Notes:

- `global/project/session` narrative 可背景定期 refresh。
- `query_specific` narrative 可在 recall 時動態生成或短期 cache。
- EntityNarrative 不應取代 Event；它是由 canonical memory 生成的 view。
- MVP supports single-entity narratives only. Multi-entity joint narratives, such as `user x current_project`, are out of scope for MVP.

Constraint:

```text
query_hash IS NOT NULL iff scope = query_specific
query_hash IS NULL iff scope != query_specific
```

Default TTL:

```text
global: 7 days
project: 3 days
session: until session idle flush or 24 hours
query_specific: 1 hour
```

### 7.12 PersistentMemory

PersistentMemory 對應 MemoryOS LPM。它保存穩定的 persona、preference、project knowledge、workflow pattern 等長期記憶。

它不是 GSW entity summary。GSW-style summary 應使用 EntityNarrative。

Suggested fields:

```text
id
memory_type
content
entity_ids_json
supporting_event_ids_json
confidence
stability
last_reinforced_at
created_at
updated_at
```

Initial memory types:

```text
user_preference
project_knowledge
stable_fact
habit
design_decision
workflow_pattern
```

Stability:

```text
stability is a float in [0, 1].
It increases when supporting events are reinforced over time.
It decreases when conflicting evidence emerges or supporting evidence is forgotten.
MVP may compute stability as a normalized reinforcement count.
```

### 7.13 Evidence

Evidence links canonical memory back to observations and source spans.

Suggested fields:

```text
id
observation_id
event_id
span_start
span_end
content_excerpt
created_at
```

### 7.14 EventEntity

EventEntity links events to participating entities.

Suggested fields:

```text
event_id
entity_id
role_in_event
confidence
created_at
```

### 7.15 MemorySegment / MTM

MemoryOS-style MTM 管理單位。Segment 不是真相來源，而是聚合與 lifecycle 管理用。

Suggested fields:

```text
id
topic
summary
entity_ids_json
event_ids_json
keywords_json
heat
visit_count
interaction_length
last_accessed_at
lifecycle
created_at
updated_at
```

Lifecycle:

```text
active
archived
promoted
forgotten
```

Base MemoryOS heat:

```text
heat = alpha * N_visit + beta * L_interaction + gamma * R_recency
R_recency = exp(-delta_t / mu)
default_mu = 1e7
default_tau = 5
```

Optional extended heat:

```text
extended_heat = heat + delta * importance
```

`importance` is an implementation extension, not part of the original MemoryOS formula.

Visit count update rule:

- Increment `visit_count` when a recall result includes the segment or one of its events/narratives.
- Increment only once per segment per recall call.
- Update `last_accessed_at` on every increment.

MVP MTM scope:

- MVP includes simplified MTM.
- Segment assignment may use topic keywords, entity overlap, and embedding similarity.
- MVP does not need full segmented paging implementation, but must maintain segment membership, heat, lifecycle, and promotion decisions.

PersistentMemory trigger:

```text
promote when:
  segment.heat >= default_tau
  and segment contains stable/repeated high-importance events
  and confidence >= confidence_min_threshold
```

Immediate event-level promotion is allowed for explicit user corrections, durable preferences, and project design decisions with high confidence.

### 7.16 MemoryConflict

MVP 不必自動解決所有衝突，但必須偵測並標記。

Suggested fields:

```text
id
conflict_type
entity_id
new_record_type
new_record_id
old_record_type
old_record_id
description
severity
status
created_at
resolved_at
```

Status:

```text
open
ignored
resolved
```

Initial conflict types:

```text
state_contradiction
role_contradiction
anchor_contradiction
action_contradiction
relation_contradiction
evidence_invalidated
```

## 8. MCP Tools

### 8.1 `ingest_observation`

自動記憶入口。Client/agent 可在每輪對話後或 session 結束時呼叫。

此 tool 必須走 fast path：同步寫入 Observation、更新 SessionBuffer、建立 jobs，然後立即回傳。Operator/Reconciler/embedding/indexing 預設由 background worker 執行。

Input:

```json
{
  "source_type": "chat",
  "content": "...",
  "session_id": "...",
  "source_uri": "...",
  "client_dedup_key": "...",
  "metadata": {}
}
```

Output:

```json
{
  "observation_id": "...",
  "session_page_id": "...",
  "importance": 0.82,
  "novelty": 0.71,
  "queued_jobs": ["..."],
  "status": "queued"
}
```

### 8.2 `recall`

根據 query 回傳 memory pack。

Input:

```json
{
  "query": "...",
  "scope": "auto",
  "session_id": "...",
  "include_session_buffer": true,
  "limits": {
    "events": 10,
    "narratives": 5,
    "persistent_memories": 5,
    "open_questions": 5,
    "evidence": 10
  },
  "max_response_tokens": 8000,
  "include_evidence": true
}
```

Scope enum:

```text
auto
current_session
project
user
global
```

`auto` strategy:

- If `session_id` is provided, include current session context and related project/user memory.
- If project metadata is provided, prioritize project scope.
- Otherwise fall back to user/global memory.

SessionBuffer rule:

- SessionBuffer is included only when `session_id` is provided and `include_session_buffer = true`.
- If no `session_id` is provided, recall must skip SessionBuffer and use persistent memory/indexed memory only.

Output:

```json
{
  "relevant_events": [],
  "entity_timelines": [],
  "entity_narratives": [],
  "persistent_memories": [],
  "spatiotemporal_anchors": [],
  "open_questions": [],
  "evidence": [],
  "confidence": 0.0
}
```

### 8.3 `search_events`

搜尋事件。

Input filters:

- query
- entity_id
- event_type
- time range
- importance threshold

### 8.4 `get_entity_timeline`

取得某個 entity 的角色、狀態、事件、actions、anchors 與 narrative。

Input filters:

```json
{
  "entity_id": "...",
  "since": "...",
  "until": "...",
  "event_types": ["decision", "preference"],
  "state_types": ["preference", "goal"],
  "limit": 50
}
```

Defaults:

```text
limit = 50
since = last 30 days, unless entity has fewer than 50 recent records
```

Output should include:

- entity
- roles
- states
- actions
- related events
- spatiotemporal anchors
- persistent memories
- cached narratives
- open conflicts

### 8.5 `consolidate_memory`

手動或排程觸發 consolidation：

- event -> persistent memory
- entity narrative refresh
- segment summary refresh
- heat update
- stale memory archive
- open question resolution pass

Input:

```json
{
  "tasks": ["persistent_memory", "narrative_refresh", "segment_summary", "heat_update", "archive", "open_question_resolution"],
  "scope": "auto",
  "entity_id": "...",
  "segment_id": "..."
}
```

If `tasks` is omitted, run all safe consolidation tasks for the selected scope.

### 8.6 `forget`

刪除或降低某些記憶權重。

Modes:

- soft forget: lifecycle 設為 forgotten，預設 recall 不返回
- hard delete: 刪除 SQLite canonical data 與 LanceDB index

Forget must cascade safely:

- If forgetting an observation, dependent evidence should be invalidated.
- If forgetting an event, dependent narratives and persistent memories must be marked stale.
- If forgetting supporting events for a persistent memory, confidence/stability must be recomputed.
- Hard delete should enqueue `rebuild_index` or targeted LanceDB delete jobs.

### 8.7 `memory_status`

回傳目前記憶庫狀態：

- observation count
- event count
- entity count
- open question count
- conflict count
- persistent memory count
- entity narrative count
- pending index jobs
- failed jobs
- LanceDB index health

### 8.8 `rebuild_index`

從 SQLite 重建 LanceDB。

`rebuild_index` must recompute embeddings from canonical SQLite text. It must not assume existing LanceDB vectors are still valid.

The server should store an `embedding_model_signature` setting. If the configured embedding provider/model changes, startup should warn that `rebuild_index` is required.

### 8.9 `get_open_questions`

查詢目前未解的 forward-falling questions。

Input filters:

- entity_id
- event_id
- status
- priority
- created_at range

### 8.10 `get_conflicts`

查詢目前 memory conflicts。MVP 只需支援查詢與標記，不必自動解決。

### 8.11 `flush_session_buffer`

將某個 session 的 SessionBuffer 強制寫入 SQLite mirror，並觸發 consolidation jobs。適合 client 在 session end 時呼叫。

### 8.12 `process_background_jobs`

手動處理 background jobs。若部署環境不啟用常駐 worker，可由 client/agent 定期呼叫。

Input:

```json
{
  "max_jobs": 50,
  "max_seconds": 30
}
```

## 9. Ingestion Flow

`ingest_observation` uses a two-stage design.

Fast path:

```text
1. Receive observation
2. Update SessionBuffer / STM
3. Write observation to SQLite
4. Run cheap importance and novelty heuristics
5. Create index/extraction jobs if above threshold
6. Return observation_id and queued_jobs
```

Background worker:

The implementation should preserve the GSW conceptual split but may name modules in engineering terms:

```text
Operator = ExtractWorker = local observation/session context -> local workspace proposal W_n
Reconciler = IntegrateWorker = prior memory M_{n-1} + W_n -> updated memory M_n
```

ExtractWorker / Operator:

```text
1. Load pending jobs
2. Run Operator extraction with SessionBuffer chain summary as background context
3. Extract partial event structure
4. Produce local role/state/action proposals
5. Produce local spatiotemporal anchor proposals
6. Produce local open question proposals
```

IntegrateWorker / Reconciler:

```text
1. Resolve or create entities conservatively
2. Reify entity_roles, entity_states, event_actions, relations
3. Create, link, or propagate spatiotemporal anchors
4. Deduplicate and update open questions
5. Try to answer existing open questions using new evidence
6. Update MTM segment membership and heat
7. Run conflict detection
8. Refresh affected entity narratives
9. Extract or refresh persistent memories when triggers fire
10. Upsert searchable records into LanceDB
11. Mark jobs completed or failed with retry metadata
```

Concurrency:

- MVP must run only one IntegrateWorker at a time.
- The IntegrateWorker must use SQLite transactions for each integration job.
- Long LLM calls should happen before the integration transaction begins.
- Optional LLM judgment for OpenQuestion answer validation must happen before the SQLite integration transaction begins; the validated result should be passed into the transaction payload.
- Integration transaction should contain only deterministic read/write operations needed to apply the local workspace proposal.
- LanceDB upsert happens after the SQLite transaction commits. LanceDB failure should retry independently and must never roll back SQLite state.

Important rule:

> SQLite transaction success means memory is saved. LanceDB indexing failure must not lose memory.

### 9.1 Job Queue Requirements

`index_jobs` / `memory_jobs` should support:

- idempotency key
- job_type
- payload_json
- status
- retry_count
- max_retries
- last_error
- scheduled_at
- started_at
- completed_at

Initial job types:

```text
extract_event
resolve_entities
reconcile_workspace
resolve_open_questions
detect_conflicts
refresh_entity_narrative
invalidate_entity_narrative
extract_persistent_memory
update_memory_segment
upsert_lancedb
delete_lancedb
rebuild_index
```

Crash recovery:

- Jobs stuck in `running` beyond timeout should return to `pending`.
- Every job must be safe to retry.
- LanceDB upsert jobs must use SQLite ids as stable keys.
- Event/state/anchor update or forget operations must enqueue `invalidate_entity_narrative` for dependent narratives before recall can reuse them.

### 9.2 Score Definitions

`importance`, `novelty`, and `confidence` must be explicit because they drive selective promotion.

Importance:

- Measures long-term value.
- Signals: decision, preference, correction, commitment, project state change, document claim, repeated topic, explicit user instruction.
- MVP can use rules plus optional LLM scorer.

Novelty:

- Measures whether the observation/event adds new information beyond existing memory.
- MVP can compare against recent SessionBuffer, LanceDB nearest neighbors, and active entity states.

Confidence:

- Measures extraction reliability.
- Should consider evidence clarity, LLM structured output validation, number of supporting observations, and conflict status.
- MVP may use max confidence among supporting evidence, then downgrade if conflicts are open.

Default thresholds:

```text
importance_promote_threshold = 0.65
novelty_promote_threshold = 0.35
confidence_min_threshold = 0.50
```

Low-score observations remain in Observation/STM but are not promoted to Event by default.

## 10. Recall Flow

```text
1. Receive query
2. Include recent SessionBuffer context
3. Extract query entities, anchors, and intent
4. Retrieve candidates from LanceDB
5. Load canonical records from SQLite by sqlite_id
6. Expand related entities, roles, states, actions, anchors, relations, open questions, and evidence
7. Generate or load entity-level chronological narratives
8. Rerank and compress
9. Return memory pack
```

The output should not be raw chunks only. It should include structured memory:

- relevant events
- entity timelines
- entity narratives
- persistent memories
- spatiotemporal anchors
- open questions if relevant
- supporting evidence
- confidence

### 10.1 Recall Defaults

Initial recall settings:

```text
lancedb_candidate_limit = 50
sqlite_expansion_depth = 1
max_returned_events = 10
max_returned_narratives = 5
max_returned_persistent_memories = 5
max_returned_open_questions = 5
max_response_tokens = 8000
include_session_buffer = true
include_evidence = true
```

Rerank may be implemented by:

- lexical/entity overlap
- embedding similarity
- recency
- importance
- confidence
- optional LLM reranker

Compression may be rule-based for MVP. LLM compression should be optional because MCP stdio calls should remain responsive.

## 11. Automatic Memory Policy

Automatically promote these categories:

- user decisions
- user preferences
- corrections
- project state changes
- task progress
- commitments
- important facts
- recurring patterns
- errors and workarounds
- document claims
- role/state changes
- spatiotemporal facts needed to ground events
- answers to open questions

Do not promote by default:

- greetings
- transient small talk
- duplicate statements
- low-value operational chatter
- ungrounded model guesses

### 11.1 Conflict Policy

When a new state conflicts with an active state:

- If the new state is clearly newer and temporally explains the change, close the old state's `valid_to`.
- If both states appear simultaneously valid, create `memory_conflicts`.
- Recall should prefer active, high-confidence, recent states, but may expose conflicts when `include_conflicts = true`.
- MVP does not need automatic conflict resolution UI.

### 11.2 LLM Cost and Latency Policy

MCP tool calls should be fast. Expensive LLM operations should default to async jobs.

Synchronous path may use:

- cheap rules
- local heuristics
- optional lightweight classifier

Async path may use:

- Operator extraction
- Reconciler
- open question resolution
- entity narrative generation
- persistent memory extraction
- optional LLM reranking/compression

LLM provider should be configurable. The server should not hardcode GPT-4o or any closed provider.

Suggested modes:

```text
LTM_EXTRACTION_MODE=async
LTM_BACKGROUND_WORKER=inline | process | manual
LTM_LLM_PROVIDER=openai | anthropic | ollama | lmstudio | none
LTM_LLM_MODEL=...
LTM_EMBEDDING_PROVIDER=openai | voyage | ollama | local | none
LTM_EMBEDDING_MODEL=...
LTM_MAX_LLM_CALLS_PER_OBSERVATION=2
LTM_MAX_DAILY_LLM_COST_USD=...
LTM_SESSION_IDLE_FLUSH_SECONDS=900
LTM_MAX_OBSERVATION_CHARS=4000
LTM_IMPORTANCE_PROMOTE_THRESHOLD=0.65
LTM_NOVELTY_PROMOTE_THRESHOLD=0.35
LTM_CONFIDENCE_MIN_THRESHOLD=0.50
LTM_RECALL_MAX_RESPONSE_TOKENS=8000
LTM_SQLITE_JOURNAL_MODE=WAL
```

Background worker modes:

```text
inline: run background jobs in the MCP server process after fast-path response, usually via an internal async task or thread.
process: run a separate worker process that shares the same SQLite/LanceDB paths.
manual: do not run automatically; jobs are processed only when `process_background_jobs` is called.
```

MVP default should be `inline` for simplest deployment. Production-like local installs may use `process`.

Latency targets:

```text
ingest_observation fast path: p95 < 100 ms without LLM calls
recall without LLM compression: p95 < 500 ms for local DB under 100k events
process_background_jobs: best effort, allowed to exceed interactive latency
```

## 12. Deployment UX

Target MCP config example:

The example below is a production-like local install. MVP defaults may use `LTM_BACKGROUND_WORKER=inline`.

```json
{
  "mcpServers": {
    "long-term-memory": {
      "command": "uvx",
      "args": ["long-term-memory-mcp"],
      "env": {
        "LTM_HOME": "~/.local/share/long-term-memory-mcp",
        "LTM_EXTRACTION_MODE": "async",
        "LTM_BACKGROUND_WORKER": "process",
        "LTM_LLM_PROVIDER": "openai",
        "LTM_LLM_MODEL": "gpt-4o-mini",
        "LTM_EMBEDDING_PROVIDER": "openai",
        "LTM_EMBEDDING_MODEL": "text-embedding-3-small",
        "LTM_SESSION_IDLE_FLUSH_SECONDS": "900",
        "LTM_MAX_OBSERVATION_CHARS": "4000",
        "LTM_IMPORTANCE_PROMOTE_THRESHOLD": "0.65",
        "LTM_NOVELTY_PROMOTE_THRESHOLD": "0.35",
        "LTM_CONFIDENCE_MIN_THRESHOLD": "0.50",
        "LTM_RECALL_MAX_RESPONSE_TOKENS": "8000",
        "LTM_SQLITE_JOURNAL_MODE": "WAL"
      }
    }
  }
}
```

The user should not need to manually create SQLite or LanceDB. Server startup should create required folders, SQLite schema, and LanceDB tables automatically.

If no LLM provider is configured, server should still run in degraded mode:

- store observations
- maintain SessionBuffer
- support lexical/FTS recall
- queue extraction jobs as blocked or skipped
- allow later reprocessing after LLM config is added

### 12.1 Schema Migrations

MVP should use explicit SQLite schema versioning.

Recommended approach:

```text
schema_migrations(version, name, applied_at, checksum)
```

Migration files should be plain SQL or a small in-repo migration runner. Avoid requiring a separate database service or heavyweight migration framework for the local-first MVP.

Startup behavior:

- create database if missing
- apply pending migrations in order
- refuse to start on checksum mismatch unless an explicit repair flag is provided
- LanceDB schema/indexes should be validated separately and rebuilt from SQLite if needed

## 13. MVP Scope

The full MVP below is the first complete event-centric episodic memory release. To avoid a too-large first useful release, implementation should be sliced.

Suggested release slices:

```text
M0 skeleton:
  MCP stdio server
  SQLite migrations
  SessionBuffer
  Observation write
  SQLite FTS5 lexical recall
  memory_status

M1 retrieval:
  LanceDB index
  async job queue
  rebuild_index
  basic event extraction
  conservative entity linking

M2 episodic core:
  EntityRole / EntityState
  EventAction
  SpatiotemporalAnchor
  OpenQuestion
  basic Reconciler / IntegrateWorker

M3 MemoryOS lifecycle:
  simplified MTM MemorySegment
  heat update
  PersistentMemory trigger
  EntityNarrative cache
  conflict detection
```

User-facing milestone meaning:

```text
M0 + M1: structured RAG experience; the server remembers observations and can retrieve them lexically/semantically.
M2: first GSW-style episodic memory experience; the server can connect roles, states, actions, anchors, and open questions.
M3: first MemoryOS-style lifecycle experience; the server can segment, heat-rank, and promote stable memories.
```

MVP should include:

1. MCP stdio server.
2. SQLite schema and migrations.
3. LanceDB local index.
4. `ingest_observation`.
5. SessionBuffer / STM with FIFO dialogue pages and dialogue chain metadata.
6. Basic importance, novelty, and confidence scoring.
7. Async job queue with idempotency, retry, and crash recovery.
8. Event extraction with partial structured schema.
9. Conservative entity creation and linking.
10. EntityRole and EntityState tables.
11. EventAction table.
12. SpatiotemporalAnchor tables and basic coupling.
13. OpenQuestion / forward-falling question table.
14. Basic open question resolution pass.
15. Conflict detection and conflict marking.
16. Simplified MemorySegment / MTM with heat and lifecycle.
17. EntityNarrative cache/generation.
18. PersistentMemory extraction.
19. `recall`.
20. `get_entity_timeline`.
21. `get_open_questions`.
22. `get_conflicts`.
23. `memory_status`.
24. `rebuild_index`.

MVP can defer:

- advanced Reconciler reasoning
- conflict resolution UI
- complex graph traversal
- multi-backend adapters
- full automatic background scheduler if `process_background_jobs` exists
- cloud sync
- multimodal ingestion
- probabilistic role/state transition modeling

## 14. Evaluation Plan

MVP should include automated tests and a small evaluation suite.

### 14.1 Unit / Integration Tests

Required tests:

- observation write is durable even if LanceDB indexing fails
- LanceDB index can be rebuilt from SQLite
- SessionBuffer keeps FIFO order and dialogue chain metadata
- entity resolution does not auto-merge low-confidence aliases
- EntityRole and EntityState are stored separately
- spatiotemporal anchors can be shared across multiple entities/events
- open questions can be created and later answered
- conflicting states are marked without destructive overwrite
- forget invalidates dependent narratives/persistent memories

### 14.2 Memory Quality Tests

Minimum synthetic scenarios:

- User preference changes over time.
- Project decision is recalled with evidence.
- Same entity appears across multiple sessions with different roles.
- Date/location appears in one observation and actor appears in another, requiring anchor coupling.
- An open question from an earlier observation is answered by a later observation.

### 14.3 Benchmark Direction

Full EpBench/LoCoMo evaluation can be deferred, but the design should allow it.

Initial target:

```text
small_epbench_style_subset:
  event recall F1 >= baseline vector search
  lower hallucination rate than raw chunk recall
  evidence coverage >= 90% for returned memories
```

Hallucination measurement:

- Baseline should be raw chunk recall using the same underlying observations.
- Evaluation questions should have known ground-truth entities/times/locations/events.
- A returned answer is hallucinated if it includes entities, times, locations, or claims not supported by returned evidence.
- MVP can use deterministic LLM-as-judge with stored prompts and temperature 0, plus spot-checkable JSON outputs.
- For release gating, keep a fixed synthetic benchmark file in the repo so results are reproducible.

## 15. Future Extensions

- Stronger GSW-style Operator/Reconciler.
- Cross-document episodic workspace.
- Advanced entity resolution and coreference resolution.
- Conflict resolution UI and user-mediated memory correction.
- Postgres / Neo4j / Qdrant adapters.
- Import/export memory packs.
- Optional encryption at rest.
- Memory inspection dashboard.
- Full EpBench and LoCoMo evaluation.
- Local model fine-tuning for Operator/Reconciler.
- Multi-agent shared memory policies.

## 16. Key Engineering Decisions

| Decision | Source | Evidence / Rationale |
| --- | --- | --- |
| Use SQLite + LanceDB | Engineering constraint | Local-first MCP should not require deployed DB services; SQLite is canonical and LanceDB is rebuildable retrieval index. |
| Event is the primary memory unit | GSW Introduction + episodic memory framing | Episodic memory should track actors, roles, states, actions, and spatiotemporal context rather than isolated chunks. |
| SessionBuffer implements STM | MemoryOS §3.2 | STM stores recent dialogue pages and dialogue chain context. |
| Simplified MemorySegment implements MTM | MemoryOS §3.2/§3.3 | MTM uses topic-like segments/pages and heat-based lifecycle. MVP implements minimal segment membership, heat, and promotion. |
| PersistentMemory implements LPM | MemoryOS §3.2/§3.3 | LPM stores stable user/agent persona, knowledge, and traits after promotion. |
| EntityNarrative implements entity-level summary | GSW Question Answering + Appendix C | GSW improves token efficiency by returning reconciled entity-level narratives instead of raw chunks. |
| OpenQuestion implements forward-falling questions | GSW Operator task design + Reconciler QA resolution | Questions generated from partial context can be answered by later chunks/events. |
| SpatiotemporalAnchor implements space/time coupling | GSW Time and Space Continuity + Reconciler | Entities sharing a situation should share temporal/spatial nodes. |
| EntityRole and EntityState are separate | GSW "Actors, Roles and States" formal model | Role defines action distribution; state constrains behavior under a role. |
| EventAction is first-class | GSW "Verbs and Valences" formal model | Verbs/actions serve as causal certificates for role/state transitions. |
| Event schema is progressively enriched | Engineering constraint + GSW | MCP ingestion must be fast; extraction may be partial and improved asynchronously. |
| Automatic memory uses selective promotion | MemoryOS | Not all observations become long-term memory; promotion depends on lifecycle policy and heat. |
| Expensive extraction/reconciliation/indexing is async | MCP stdio UX | Tool calls should return quickly; LLM work belongs in background jobs. |
| Every retrieved memory should be traceable to evidence | Both papers | Interpretability and grounded recall are required to avoid hallucinated memory. |
