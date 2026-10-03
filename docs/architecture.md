# Ledger Architecture

This document explains how Ledger is put together: the component boundaries,
the data that flows between them, and the extension points you use to adapt
it to your own LLM application.

## 1. System overview

```text
                    ┌─────────────────────┐
                    │ Golden Dataset      │  datasets/golden.jsonl
                    │ versioned cases     │  (schema-validated JSONL)
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Evaluation Runner   │  src/ledger/evaluation/runner.py
                    │ concurrency+retries │
                    └──────────┬──────────┘
                               │
               ┌───────────────┴────────────────┐
               ▼                                ▼
      ┌─────────────────┐              ┌─────────────────┐
      │ Baseline SUT    │              │ Candidate SUT   │   src/ledger/systems/
      │ Version A       │              │ Version B       │   (rag | mock | yours)
      └────────┬────────┘              └────────┬────────┘
               │                                │
               └───────────────┬────────────────┘
                               ▼
                    ┌─────────────────────┐
                    │ Evaluation Engine   │  src/ledger/evaluation/evaluator.py
                    ├─────────────────────┤
                    │ Deterministic       │  src/ledger/metrics/deterministic.py
                    │ LLM Judge (rubric)  │  src/ledger/judges/rubric.py
                    │ Per-case results    │
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │ Result Store        │  src/ledger/tracking/artifacts.py
                    │ results.jsonl       │  results/<run_id>/...
                    │ summary.json        │  experiments/index.jsonl
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │ Regression Engine   │  src/ledger/regression/
                    ├─────────────────────┤
                    │ Paired comparator   │  comparator.py
                    │ Bootstrap CI + p    │  bootstrap.py
                    │ Policy rules        │  policy.py + detector.py
                    └──────────┬──────────┘
                    ┌──────────┴───────────┐
                    ▼                      ▼
          ┌─────────────────┐     ┌─────────────────┐
          │ Report Generator│     │ CI Gate         │  src/ledger/cli/main.py
          │ md / json / tty │     │ exit 0/1/2      │  scripts/ci/gate.sh
          └─────────────────┘     └─────────────────┘
```

Two evaluation runs (baseline and candidate) are produced by the *same*
pipeline with different configs, then compared. Nothing about the comparison
depends on what the systems actually are.

## 2. Layers and responsibilities

| Layer | Package | Responsibility |
|---|---|---|
| Contracts | `core/` | Pydantic models (`TestCase` inputs live in `dataset/schema.py`, outputs/results/summaries in `core/models.py`) and the four ABCs: `SystemUnderTest`, `LLMProvider`, `Judge`, `DeterministicMetric` |
| Configuration | `config/` | YAML loading with `extends` layering, `${ENV}` interpolation, secret hygiene (`api_key_env` only), pydantic validation |
| Data | `dataset/` | JSONL loading + dataset-level health validation |
| Providers | `providers/` | `mock` (deterministic offline) and `openai_compat` (any OpenAI-style HTTP API) chat providers + embedder client |
| Systems under test | `systems/` | `mock` SUT (harness self-test) and the sample `rag` system (chunk → embed → retrieve → prompt → generate) |
| Evaluation | `evaluation/` | Runner (execute SUT), Evaluator (score per case), Pipeline (orchestrate + persist), Compare service (load two runs, gate) |
| Metrics | `metrics/` | Deterministic metric implementations, judge-score bridge, aggregation math |
| Judges | `judges/` | Rubric prompts, hardened `RubricJudge`, human-label calibration |
| Regression | `regression/` | Paired bootstrap statistics, comparator, YAML policy, rule-based detector |
| Reporting | `reporting/` | `Report` model + Markdown / JSON / rich-console renderers |
| Tracking | `tracking/` | Run ids, artifact layout, experiment index, run resolution |
| CLI | `cli/` | `ledger` Typer app: evaluate / compare / gate / report / calibrate / validate-dataset / list-runs / demo |

### Dependency direction

Dependencies point inward: `cli → evaluation → (systems, metrics, judges,
regression, reporting, tracking) → core/config/dataset/utils`. `core`
depends on nothing but pydantic and the dataset schema. No module reaches
"up" the stack, which keeps every piece unit-testable in isolation.

## 3. Key interfaces

### SystemUnderTest — the extension point that matters most

```python
class SystemUnderTest(ABC):
    version: str = "unknown"

    def generate(self, test_case: TestCase) -> SystemOutput: ...
    def describe(self) -> dict[str, Any]: ...
```

The framework never inspects a SUT's internals. A RAG pipeline, an agent, a
raw prompt template, or a remote HTTP service are all valid SUTs as long as
they turn a `TestCase` into a `SystemOutput` (answer + optional retrieved
context + latency/model/prompt metadata). Returned `retrieved_context`
unlocks retrieval-specific metrics; returning nothing there simply marks
those metrics non-applicable.

### LLMProvider — generation and judging behind one port

```python
class LLMProvider(ABC):
    name: str
    def complete(self, request: CompletionRequest) -> CompletionResponse: ...
```

`CompletionRequest.purpose` (`"generation"` / `"judge"`) lets test doubles
route behavior and lets you point generation and judging at different
models/endpoints (recommended: judge ≠ SUT model to limit self-preference
bias).

### Judge — structured verdicts, failures as data

```python
class Judge(ABC):
    def score(self, case: TestCase, output: SystemOutput) -> JudgeScore: ...
```

`RubricJudge` guarantees `score()` never raises: malformed JSON triggers
repair retries; exhausted retries produce `JudgeScore(failed=True)`. The
evaluator turns judge failures into policy-visible signals
(`judge.on_failure`: `exclude` / `fail_case` / `abort`, plus the gate's
`max_judge_failure_rate` rule), so one broken verdict can neither crash a
suite nor silently weaken the gate.

### DeterministicMetric — [0,1] or None

```python
class DeterministicMetric(ABC):
    name: str
    def compute(self, ctx: MetricContext) -> float | None: ...
```

`None` means *not applicable* (e.g. `retrieval_hit` for a SUT without
retrieval) and is excluded from aggregation - never counted as zero.

## 4. Data and artifact model

### Per-case result (`EvaluationResult`)

One flat, self-contained JSON row per (case, run): inputs (question,
reference, expected behavior, category, critical), system output (answer,
retrieved context, model, prompt version, latency, error), scores
(deterministic map, judge dimensions/overall/reason), and the derived
`overall_score` / `passed`. Flatness is deliberate: `results.jsonl` loads
directly into pandas/polars/BI tools.

### Run artifacts

```text
results/<run_id>/
    results.jsonl          # every EvaluationResult
    summary.json           # RunSummary: aggregates, counts, latency, runtime fingerprint
    config.snapshot.json   # effective config, secrets redacted
experiments/index.jsonl    # append-only run registry (run_id, version, dataset hash, headline metrics)
reports/                   # comparison_*.md/.json, run_*.md/.json, calibration_*.md/.json
```

Run ids are `<UTC stamp>-<version-slug>-<nonce>`, so listing/sorting by name
is chronological. `ledger compare/gate/report` accept a run id, a unique id
prefix, a directory path, or `latest[:system_version]`.

### Reproducibility levers

* **Seed** (`run.seed`): drives mock provider behavior and bootstrap
  resampling; recorded in `summary.json`.
* **Dataset fingerprint**: SHA-256 of the dataset file, stored per run; the
  comparator warns and intersects when fingerprints differ.
* **Config snapshot**: the exact effective configuration, redacted.
* **Runtime fingerprint**: ledger version, python version, platform, git
  commit + dirty flag.
* **Deterministic offline mode**: mock provider + hashing embedder make the
  whole loop bit-reproducible without network (real providers can't promise
  this; see methodology doc).

## 5. The regression engine

```text
two runs ──▶ align by test_id ──▶ paired per-case values per metric
                                      │
                 ┌────────────────────┼──────────────────────┐
                 ▼                    ▼                      ▼
        paired bootstrap CI      pass/fail flip           per-case
        + p-value per metric     sign test                movement
        (overall, judge dims,    (newly failing /         (regressed /
         deterministic metrics,   newly passing)           improved /
         categories, critical                             stable)
         slice)
                 └────────────────────┴──────────────────────┘
                                      ▼
                     policy rules (YAML, not code)
                                      ▼
                        PASS / WARNING / FAIL + violations
```

* **Pairing**: baseline and candidate run on the same dataset, so every
  metric comparison is *paired* per test case. Pairing removes between-case
  variance and dramatically increases power at n≈40.
* **Paired percentile bootstrap** (`regression/bootstrap.py`): resample case
  indices with replacement (2000×, seeded), recompute the mean difference,
  take percentile CI and a two-sided bootstrap p-value (floored at
  1/n_samples). No normality assumptions; correct for skewed per-case
  scores.
* **Sign test**: exact binomial test on pass/fail flips - assumption-free
  corroboration of the count rules.
* **Detector rules** (`regression/detector.py`): R1 overall, R2 categories,
  R3 critical slice + newly-failing criticals, R4 per-metric thresholds, R5
  count budgets, R6 judge-failure integrity, plus structural warnings
  (dataset mismatch, `min_paired_cases`). With `require_significance: true`,
  beyond-threshold degradations that lack statistical support downgrade to
  WARNING instead of blocking merges.
* Every violation carries a machine-readable `rule` id and a human message
  rendered into reports and the CI step summary.

## 6. Sample RAG system (and why it is small)

`systems/rag/` implements documents → chunking → embeddings → retrieval →
versioned prompt → generation with injected parts:

* `chunking.py` - sentence-aware, heading-tagged chunks; markdown table rows
  kept atomic; configurable size/overlap.
* `embeddings.py` - `HashingTfIdfEmbedder`: offline, deterministic hashed
  word/char-n-gram TF-IDF (no model download). An OpenAI-compatible HTTP
  embedder is available via config for semantic retrieval.
* `retrieval.py` - exact cosine top-k over numpy with deterministic tie
  breaking.
* `prompts.py` - versioned prompt templates (`v1_grounding`,
  `v2_concise_cited`, `v0_sloppy` for the degraded demo). The prompt version
  is recorded on every result.
* `system.py` - `RAGSystem` composing the above behind `SystemUnderTest`.

The RAG app is a *demonstration target*, deliberately minimal: the project's
value is that Ledger can evaluate **any** two versions of **any** system.

## 7. Extension guide

**Evaluate your own application.** Implement `SystemUnderTest` (an HTTP
call to your service is usually enough), register it with
`@register_system("myapp")` or construct it directly, and point a config at
it:

```yaml
system:
  kind: myapp
  # your config fields
```

**Add a metric.** Subclass `DeterministicMetric`, add it to
`METRIC_REGISTRY`, list it under `metrics.deterministic` in config. Return
`None` when inapplicable.

**Change the judge rubric.** Edit `judges/prompts.py` (anchors, weights,
tags), then re-run `ledger calibrate` - a rubric change invalidates previous
calibration evidence.

**Tune the gate.** Everything lives in `configs/policy.yaml`; no Python
changes needed.

**Swap statistics.** `paired_bootstrap` is one function; the comparator only
needs `point/ci/p` - a permutation test or Bayesian alternative slots in
behind the same `MetricComparison` fields.
