# Benchmarking

This project should be measured in two layers:

1. Mechanism regression: can the memory graph perform the behaviors the
   papers motivated?
2. Long-conversation retrieval: can the MCP server retrieve the right
   evidence from a realistic multi-session chat history?

The current benchmark commands are intentionally local and zero-cost.
They do not call OpenRouter unless a future explicit judge mode is added.

## 1. Synthetic Baseline

Use this as the fast CI gate for M2/M3 regressions:

```powershell
python -m ltm_memory benchmark
```

It covers:

- decision recall with evidence
- spatiotemporal anchor coupling
- anchor propagation through shared entities
- forward-falling open-question resolution
- preference recall across events
- MemoryOS segment heat and narrative caching
- forget cascade behavior

## 2. Episodic Hard Synthetic

Use this when changing Operator/Reconciler, entity linking, anchors,
roles, states, actions, or EntityNarrative:

```powershell
python -m ltm_memory benchmark-episodic
```

The default file is:

```text
benchmarks/synthetic/episodic_hard.json
```

It is GSW-shaped: the scenarios are designed around forward-falling
questions, space/time coupling, role/state/action separation, and
entity-level narrative payloads.

## 3. LoCoMo Retrieval Gate

LoCoMo is the recommended first external benchmark because the official
dataset has long multi-session conversations, timestamps, QA labels,
dialog-id evidence, generated observations, session summaries, and event
summaries.

Download `locomo10.json` from the official repository:

```text
https://github.com/snap-research/locomo
```

Then run:

```powershell
python -m ltm_memory benchmark-locomo --dataset data\locomo10.json --sample-limit 1
python -m ltm_memory benchmark-locomo --dataset data\locomo10.json --sample-limit 10 --recall-limit 15
```

Default behavior:

- ingests dialog turns with session timestamps
- processes deterministic background jobs
- evaluates LoCoMo categories 1-4
- skips category 5 adversarial questions by default
- reports retrieval/evidence metrics, not final answer accuracy

Useful options:

```powershell
python -m ltm_memory benchmark-locomo --dataset data\locomo10.json --categories 1 2 3 4 5
python -m ltm_memory benchmark-locomo --dataset data\locomo10.json --source-mode dialogs+observations
python -m ltm_memory benchmark-locomo --dataset data\locomo10.json --max-questions 25 --recall-limit 20
```

Metrics:

- `evidence_recall_at_k`: fraction of gold dialog-id evidence retrieved
- `any_evidence_hit_rate`: fraction of evidence-bearing questions with at
  least one gold evidence dialog retrieved
- `answer_string_hit_rate`: cheap proxy for whether the retrieved context
  contains answer-bearing text
- `category_metrics`: the same metrics grouped by LoCoMo category

Interpretation:

- If `any_evidence_hit_rate` is low, improve recall/search before touching
  answer generation.
- If evidence hit rate is good but answer quality is bad in a future
  LLM-judge benchmark, improve response synthesis and context packing.
- If temporal category scores are weak, inspect timestamp ingestion,
  anchors, and event timeline retrieval.

## 4. Later: LLM-as-Judge Answer Quality

Do this only after retrieval is stable, because it costs API credits and
adds model variance. The future shape should be explicit, for example:

```powershell
python -m ltm_memory benchmark-locomo --dataset data\locomo10.json --judge-provider openrouter
```

That mode should:

- retrieve memory payloads with `recall`
- ask an answer model to answer from retrieved context only
- judge generated answers against LoCoMo gold answers
- report accuracy by category, cost, latency, and judge model version

## References

- LoCoMo official repository: https://github.com/snap-research/locomo
- MemoryOS paper page: https://huggingface.co/papers/2506.06326
- GSW paper page: https://huggingface.co/papers/2511.07587
- LongMemEval paper page: https://huggingface.co/papers/2410.10813
