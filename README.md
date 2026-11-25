# Ledger

**Regression testing & evaluation infrastructure for LLM applications.**

Ledger answers one question with statistical rigor, every time a prompt,
model, retrieval setting, or system config changes:

> *Did this change make our LLM application measurably worse — and should CI
> block it?*

It runs a versioned golden dataset against two versions of your system
(baseline vs candidate), scores every answer with deterministic metrics and a
calibrated LLM-as-a-judge, compares the paired results with bootstrap
statistics, applies a YAML-configured regression policy, and turns the whole
thing into a CI quality gate with human-readable reports.

```text
AI application changes → automated evaluation → statistical comparison
       → regression detection → CI quality gate → bad change blocked
```

> **Status / honesty note.** Everything in this repository runs offline out
> of the box via a deterministic mock provider. All example numbers shown
> below were produced by that mock mode and demonstrate *plumbing*, not the
> quality of any real model. No benchmark here should be interpreted as a
> real-world performance claim.

---

## Table of contents

1. [Why Ledger exists](#why-ledger-exists)
2. [Architecture](#architecture)
3. [Core concepts](#core-concepts)
4. [Repository structure](#repository-structure)
5. [Installation](#installation)
6. [Configuration](#configuration)
7. [Local execution & CLI](#local-execution--cli)
8. [Running an evaluation](#running-an-evaluation)
9. [Comparing versions & the gate](#comparing-versions--the-gate)
10. [Judge calibration](#judge-calibration)
11. [Regression detection](#regression-detection)
12. [CI/CD](#cicd)
13. [Docker](#docker)
14. [Testing](#testing)
15. [Example generated reports](#example-generated-reports-mock-mode)
16. [Engineering decisions](#engineering-decisions)
17. [Limitations](#limitations)
18. [Future improvements](#future-improvements)

---

## Why Ledger exists

LLM applications regress silently. A "harmless" prompt tweak, a model minor
version bump, a `top_k` change, a new system instruction — any of them can
degrade answer quality on a slice of traffic you never look at. Traditional
unit tests can't catch this: outputs are open-ended, non-deterministic, and
*partially* correct.

Teams respond with one of two failure modes:

* **Vibe-checking** a handful of prompts before shipping (unmeasured,
  unrepeatable), or
* **Dashboard theater**: aggregate score lines without pairing, without
  significance, without thresholds — noise gets shipped as "regression" and
  real regressions hide inside variance.

Ledger is the engineering answer: treat model behavior as a **testable system
property**. Golden dataset in, per-case scores out, paired statistics between
versions, an explicit policy for what "too much degradation" means, and a
gate that blocks merges when the evidence says so — with reports that explain
*which rule fired, on which cases, with which confidence interval*.

## Architecture

```text
                    ┌─────────────────────┐
                    │ Golden Dataset      │  versioned, schema-validated cases
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │ Evaluation Runner   │  concurrency, retries, error capture
                    └──────────┬──────────┘
               ┌───────────────┴────────────────┐
               ▼                                ▼
      ┌─────────────────┐              ┌─────────────────┐
      │ Baseline SUT    │              │ Candidate SUT   │  SystemUnderTest
      │ Version A       │              │ Version B       │  interface (RAG,
      └────────┬────────┘              └────────┬────────┘  agent, API, ...)
               └───────────────┬────────────────┘
                               ▼
                    ┌─────────────────────┐
                    │ Evaluation Engine   │  deterministic metrics
                    │                     │  + rubric LLM judge (hardened)
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │ Result Store        │  results.jsonl / summary.json
                    │                     │  config snapshots, experiment index
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │ Regression Engine   │  paired bootstrap CI + p-values,
                    │                     │  sign test, policy rules
                    └──────────┬──────────┘
                    ┌──────────┴───────────┐
                    ▼                      ▼
          ┌─────────────────┐     ┌─────────────────┐
          │ Report Generator│     │ CI Gate         │  exit 0/1/2
          │ md / json / tty │     │ GitHub Actions  │  PASS / WARNING / FAIL
          └─────────────────┘     └─────────────────┘
```

Full component walkthrough, data model and extension guide:
**[docs/architecture.md](docs/architecture.md)**.

## Core concepts

| Concept | Meaning in Ledger |
|---|---|
| **System Under Test (SUT)** | Your application behind one interface: `generate(test_case) -> SystemOutput`. RAG, agents, raw prompts, remote services — the framework doesn't care. |
| **Golden dataset** | Versioned JSONL of structured cases: question, reference and/or expected behavior, category, `critical` flag, per-case rubric, retrieval expectations. |
| **Evaluation run** | Dataset × one system config → per-case results + aggregate summary, persisted with fingerprints (dataset hash, config snapshot, seed, git commit). |
| **Deterministic metrics** | Mechanical checks in [0,1]: token F1, key-term coverage, citation presence, retrieval hit, refusal match, JSON validity, length compliance. `None` = not applicable. |
| **LLM-as-a-judge** | Rubric judge scoring correctness / relevance / groundedness / instruction-following (1-5), strict JSON contract, repair retries, recomputed aggregates. Failures are data, never crashes. |
| **Judge calibration** | Agreement between judge and human labels: MAE, Pearson/Spearman, within-±1 agreement, pass agreement, Cohen's κ. |
| **Paired comparison** | Same cases on both sides ⇒ per-case deltas ⇒ paired percentile bootstrap CI + p-value per metric/category/critical slice, plus an exact sign test on pass/fail flips. |
| **Regression policy** | YAML thresholds & rules (never code): overall/categories/critical bands, per-metric thresholds, newly-failing budgets, judge-failure-rate integrity rule, significance requirement. |
| **Gate decision** | `PASS` / `WARNING` / `FAIL` with machine-readable rule ids and human messages; mapped to CI exit codes 0/2/1. |

## Repository structure

```text
.
├── src/ledger/
│   ├── config/          # YAML config system (extends, env interpolation, secret hygiene)
│   ├── core/            # domain models + the four interfaces (SUT/Provider/Judge/Metric)
│   ├── dataset/         # golden-dataset schema, loader, health validator
│   ├── providers/       # mock (offline deterministic) + OpenAI-compatible HTTP provider
│   ├── systems/         # SystemUnderTest impls: mock SUT + sample RAG (chunk/embed/retrieve/prompt)
│   ├── evaluation/      # runner, evaluator, pipeline orchestration, compare service
│   ├── metrics/         # deterministic metrics, judge bridge, aggregation
│   ├── judges/          # rubric prompts, hardened judge, human-label calibration
│   ├── regression/      # bootstrap stats, paired comparator, policy, detector
│   ├── reporting/       # Report model + markdown / json / console renderers
│   ├── tracking/        # run ids, artifact store, experiment index
│   ├── cli/             # `ledger` Typer app
│   └── utils/           # logging, text, JSON repair, env/secrets, provenance
├── datasets/            # golden.jsonl, human_labels.jsonl (+ generator, README)
├── data/corpus/         # sample RAG knowledge base (fictional "Helios" docs)
├── configs/             # default / baseline / candidate / candidate_degraded / policy
├── scripts/             # evaluate.py, compare.py, calibrate_judge.py, generate_report.py, ci/gate.sh
├── tests/               # 250 tests: unit + integration (offline), fixtures
├── docs/                # architecture, evaluation-methodology, ci-cd
├── .github/workflows/   # ci.yml (lint/types/tests) + llm-evaluation.yml (the gate)
├── experiments/         # append-only run registry (index.example.jsonl committed)
├── reports/             # generated reports (reports/example/ committed)
├── results/             # run artifacts (git-ignored)
├── Dockerfile, docker-compose.yml, Makefile, pyproject.toml, requirements*.txt
└── .env.example
```

## Installation

Requires Python **3.11+**. No GPU, no model downloads, no API keys needed for
the default (mock) mode.

```bash
git clone https://github.com/your-org/ledger.git
cd ledger

make install-dev      # venv + runtime & dev deps + editable install
# or manually:
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt && .venv/bin/pip install -e ".[dev]"

.venv/bin/ledger --help
```

Verify the installation (offline, deterministic):

```bash
make test             # 250 unit + integration tests
make demo             # full pipeline: evaluate ×2 → compare → gate → reports
```

## Configuration

Everything behavior-relevant is configuration; secrets are environment-only.

| File | Role |
|---|---|
| `configs/default.yaml` | shared base: dataset, providers, judge rubric weights/thresholds, metrics, outputs |
| `configs/baseline.yaml` | the shipped system (RAG: chunk 900/120, top_k 4, prompt `v1_grounding`) |
| `configs/candidate.yaml` | the change under test (chunk 600/100, top_k 6, prompt `v2_concise_cited`) |
| `configs/candidate_degraded.yaml` | intentionally bad config that demos the gate blocking a change |
| `configs/policy.yaml` | the regression policy (thresholds, significance, count budgets) |

Mechanics (details in [docs/architecture.md](docs/architecture.md)):

* `extends:` layering — child configs override only what differs.
* `${ENV_VAR:-default}` interpolation inside YAML strings.
* **Secrets never live in YAML.** Providers reference an env var by name
  (`api_key_env: LEDGER_PROVIDER_API_KEY`); a literal `api_key` in a config
  file is rejected at load time. Copy `.env.example` → `.env` for local keys;
  it is git-ignored, and artifact config snapshots are secret-redacted.
* `LEDGER_FORCE_MOCK=1` (or `--mock`) forces the deterministic offline stack
  regardless of config — the default path in CI.

Switching to a real provider (any OpenAI-compatible API — OpenAI, Azure,
vLLM, Ollama, LiteLLM, TGI, ...):

```bash
cp .env.example .env      # fill in keys / base URLs / models
LEDGER_PROVIDER_NAME=openai_compat \
LEDGER_JUDGE_NAME=openai_compat \
ledger evaluate --config configs/baseline.yaml
```

## Local execution & CLI

```text
ledger evaluate          run one system config over the dataset → artifacts
ledger compare           compare two runs, apply policy, write reports
ledger gate              compare as a CI gate (exit 0=PASS, 1=FAIL, 2=WARNING)
ledger report            single-run aggregate report (md/json)
ledger calibrate         judge vs human labels → agreement report
ledger validate-dataset  schema + health checks for a golden dataset
ledger list-runs         stored runs, newest first
ledger demo              one-command offline end-to-end (evaluate ×2 + gate)
```

Makefile shortcuts: `make install[-dev] lint format typecheck test coverage
validate evaluate-baseline evaluate-candidate compare gate demo demo-fail
calibrate report list-runs docker-build docker-demo clean`.

## Running an evaluation

```bash
ledger evaluate --config configs/baseline.yaml
```

Pipeline: load config → load & validate dataset → build provider + SUT → run
all cases (bounded concurrency, per-case retries, errors captured not thrown)
→ deterministic metrics + judge per case → aggregate → persist:

```text
results/<run_id>/
  results.jsonl          # one self-contained row per case (see schema below)
  summary.json           # aggregates: metrics, categories, counts, latency, runtime fingerprint
  config.snapshot.json   # effective config, secrets redacted
experiments/index.jsonl  # append-only registry entry
```

Each `results.jsonl` row carries: `test_id, run_id, system_version, category,
critical, question, reference_answer, expected_behavior, actual_answer,
retrieved_context, deterministic_scores, judge_scores, judge_overall,
judge_reason, judge_failed, overall_score, passed, model_name,
prompt_version, latency_ms, error, timestamp`.

Useful flags: `--limit N` (smoke runs), `--mock` (force offline),
`--run-id X` (stable ids for CI caching), `--dataset path`, `--quiet`.

## Comparing versions & the gate

```bash
ledger evaluate --config configs/baseline.yaml
ledger evaluate --config configs/candidate.yaml
ledger gate --latest --policy configs/policy.yaml     # newest two runs
# or one-shot (evaluates both, then gates):
ledger gate --config-baseline configs/baseline.yaml \
            --config-candidate configs/candidate.yaml
```

Run references accept ids, unique prefixes, paths, or `latest[:version]`.
Reports land in `reports/` as Markdown + JSON; the console prints the summary.

**Exit-code contract:** `0` PASS · `1` FAIL · `2` WARNING (becomes 1 with
`--fail-on-warning`) · `3` usage/config error.

## Judge calibration

The judge converts text into the numbers everything else trusts — so its
agreement with humans must be measured, not assumed:

```bash
ledger calibrate --labels datasets/human_labels.jsonl --config configs/baseline.yaml
```

Produces (per dimension and overall): MAE, RMSE, bias (judge−human),
Pearson r, Spearman ρ, agreement-within-±1; plus pass-decision agreement,
Cohen's κ with interpretation, and the confusion matrix. Saved as
`reports/calibration_<ts>.{json,md}`.

Why it matters, what judges get wrong (verbosity/self-preference/position
bias, non-determinism, drift), and how to use labels:
**[docs/evaluation-methodology.md §3-4](docs/evaluation-methodology.md)**.
The bundled labels are example annotations for demonstrating the pipeline —
replace them with real ones for real decisions.

## Regression detection

No `candidate_mean < baseline_mean` hand-waving. Per metric, category and the
critical slice, Ledger computes **paired** per-case deltas, then:

* **Paired percentile bootstrap** (default 2000 resamples, seeded): mean
  delta, 95% CI, two-sided bootstrap p-value (floored at 1/n_samples).
* **Exact sign test** on pass/fail flips (assumption-free corroboration).
* **Policy rules** (`configs/policy.yaml`, all values configurable):

```yaml
require_significance: true      # beyond-threshold drops FAIL only when statistically supported
significance_alpha: 0.05
min_paired_cases: 10            # low-power warning floor
overall:    { absolute: 0.03, warning_fraction: 0.5 }
categories: { absolute: 0.05 }
critical:   { absolute: 0.02 }  # tightest band for release-blocker cases
metrics:
  retrieval_hit: { absolute: 0.05 }
counts:
  max_newly_failing_tests: 3
  max_critical_failures: 0      # any newly-failing critical test blocks
  max_judge_failure_rate: 0.1   # untrustworthy evaluation => FAIL
bootstrap: { n_samples: 2000, seed: 1337, confidence: 0.95 }
```

Decision semantics: **FAIL** on statistically-supported beyond-threshold
degradation, critical regressions, count-budget breaches, or judge-failure
integrity breaches; **WARNING** for the warning band, non-significant
degradations, dataset mismatches, small samples; **PASS** otherwise. Every
decision ships with rule ids + deltas + CIs in the report. The statistical
method is swappable behind `regression/bootstrap.py`.

## CI/CD

Two workflows (full guide: **[docs/ci-cd.md](docs/ci-cd.md)**):

* **`ci.yml`** — ruff lint + format check, mypy, pytest with coverage,
  strict golden-dataset validation. Runs on every PR.
* **`llm-evaluation.yml`** — the gate:
  1. checkout → setup Python → install deps
  2. lint + unit tests (fast feedback)
  3. resolve provider mode: **mock by default**; real provider via
     `workflow_dispatch` input + repository secrets (`LEDGER_PROVIDER_API_KEY`,
     `LEDGER_JUDGE_API_KEY`, base URLs, models — never hardcoded)
  4. evaluate baseline (cached via `actions/cache` in mock mode, keyed on
     configs+dataset+code hashes) and candidate
  5. `scripts/ci/gate.sh` → exit-code contract decides the check
  6. Markdown report appended to `$GITHUB_STEP_SUMMARY` (even on FAIL)
  7. reports + summaries + experiment index uploaded as artifacts (30 days)

Local parity: `make ci-lint ci-test ci-eval`.

## Docker

```bash
make docker-build                     # build image (non-root, slim, pinned deps)
docker compose run --rm ledger demo   # offline end-to-end inside the container
docker compose run --rm ledger demo --candidate-config configs/candidate_degraded.yaml
```

`docker-compose.yml` defines a mock-mode `ledger` service and an opt-in
`eval-real` profile that passes provider env vars through at runtime (keys
never enter the image). Results/reports are bind-mounted to the host. The
image's `HEALTHCHECK` verifies the package imports and the CLI responds.

## Testing

```bash
make test          # 250 tests, ~2s, fully offline
make coverage      # ~90% statement coverage of src/ledger
make lint && make typecheck
```

* **Unit** (`tests/unit/`): dataset schema & validation, every deterministic
  metric, aggregation math, JSON-repair & judge validation/retries/failure
  modes, score recomputation & critical thresholds, bootstrap (shift
  recovery, seed reproducibility, CI monotonicity, input validation), sign
  test (hand-computed exact p-values), policy loading, comparator pairing &
  slices, detector rules (each FAIL/WARNING path), config layering &
  interpolation & secret rejection, calibration statistics (hand-computed),
  RAG components (chunking atomicity/overlap, embedder determinism, top-k
  ordering, prompt versions), artifact store (round-trips, redaction,
  resolution), report renderers.
* **Integration** (`tests/integration/`): full pipeline (artifacts,
  determinism across runs, judge-disabled fallback, SUT-crash resilience),
  reference-vs-degraded comparison → FAIL with report files, CLI behaviors
  and the **gate exit-code contract** (PASS=0, FAIL=1, one-shot mode,
  `--latest`), shipped-config smoke run.
* External LLMs are never called: tests use the mock provider and a
  `ScriptedProvider` double that replays malformed/hostile judge responses.

## Example generated reports (mock mode)

> ⚠️ **EXAMPLE OUTPUT** — produced by the deterministic **mock** provider on
> the bundled dataset. Real, unedited tool output; it characterizes the mock
> heuristics, **not** any real model. Committed copies:
> [`reports/example/`](reports/example/).

`ledger demo` (baseline-v1 vs candidate-v2 — a prompt+retrieval change):

```text
LEDGER EVALUATION REPORT
Baseline:     baseline-v1
Candidate:    candidate-v2
Paired cases: 39

Overall:
  0.899 -> 0.896   delta: -0.002 (-0.2%)   CI: [-0.006, 0.000]   p=0.735

Categories (overall score)
  critical (all)         0.869 -> 0.869   +0.000  (+0.0%)
  answer_relevance       0.938 -> 0.938   +0.000  (+0.0%)
  factual_correctness    0.938 -> 0.938   +0.000  (+0.0%)
  groundedness           0.948 -> 0.948   +0.000  (+0.0%)
  retrieval_correctness  0.958 -> 0.938   -0.021  (-2.2%)
  ...

Tests:
  Passed (candidate):   36     Regressed:            1
  Failed (candidate):   3      Newly failing:        0
  Critical regressions: 0      Improved:             0

Decision:
  PASS
```

`make demo-fail` (the intentionally degraded config — tiny context, no
grounding instructions):

```text
Overall:
  0.899 -> 0.348   delta: -0.550 (-61.2%)   CI: [-0.680, -0.403]   p=0.001

Tests:
  Passed (candidate):   11     Regressed:            28
  Failed (candidate):   28     Newly failing:        28
  Critical regressions: 3      Improved:             3

Policy violations (excerpt):
  FAIL     overall_regression    overall quality regressed -0.5502
                                 (0.8985 -> 0.3483, CI [-0.6795, -0.4028], p=0.0005)
  FAIL     critical_failures     3 critical test(s) newly failing (max allowed: 0):
                                 fc-001, fc-004, rc-004
  FAIL     newly_failing_tests   28 test(s) newly failing (max allowed: 3)
  WARNING  metric_nonsignificant metric 'retrieval_hit' degraded -0.0556 beyond
                                 threshold but is not statistically supported

Decision:
  FAIL          (process exit code: 1)
```

Markdown/JSON equivalents: [`reports/example/comparison_fail_mock.md`](reports/example/comparison_fail_mock.md).

## Engineering decisions

* **Interfaces over frameworks.** Four small ABCs (SUT, Provider, Judge,
  Metric) define every seam; dependency injection everywhere; no plugin
  machinery, no service mesh, no databases. The smallest architecture that
  demonstrates the concepts convincingly.
* **Pydantic at every boundary.** Dataset rows, provider I/O, results,
  summaries, comparisons, policies, reports — validated on the way in,
  serializable on the way out, `extra="forbid"` to catch config typos.
* **Failures are data.** SUT crashes, judge garbage, API timeouts: captured
  per case (`error`, `judge_failed`), aggregated (`counts`), and made
  gate-relevant (integrity rules) — one bad verdict can neither crash a
  suite nor silently weaken the gate.
* **Never trust the judge's arithmetic.** Overall scores and pass decisions
  are recomputed from dimension scores + configured weights/thresholds;
  self-reported aggregates are diagnostics only.
* **Statistics before thresholds.** Paired bootstrap + sign test;
  `require_significance` downgrades noise-driven "regressions" to warnings.
  Thresholds live in YAML, so policy debates happen in config review, not in
  code.
* **Reproducibility as a first-class artifact.** Dataset fingerprints, config
  snapshots (redacted), seeds, git commit/dirty state, versions — recorded
  per run; the offline stack is bit-deterministic (asserted in tests).
* **Mock-first.** The deterministic mock provider + hashing embedder make the
  entire loop runnable in CI for free; real providers are a config flip away.
* **Deliberate omissions.** No Kubernetes/Kafka/Airflow/Redis/vector DB: a
  local-first tool with a file-based artifact store is the right scope for an
  evaluation harness. Brute-force cosine retrieval is exact at golden-dataset
  scale; an ANN index would be premature complexity.
* **Secret hygiene by construction.** `api_key_env` indirection, literal keys
  rejected at config load, `SecretStr` excluded from serialization, artifact
  redaction pass, `.env` git-ignored, CI secrets only via `secrets.*`.

## Limitations

* The bundled sample RAG and corpus are deliberately minimal; the mock
  provider is a lexical heuristic, not a model — mock-mode scores validate
  plumbing only.
* The bundled human labels are example annotations, and calibration numbers
  from the mock judge are illustrative of the *pipeline*, not of any real
  judge's quality.
* Lexical deterministic metrics (token F1, similarity) are weak proxies for
  meaning by design; they localize regressions, they don't judge quality.
* Bootstrap CIs at n≈40 are wide; `require_significance` prevents false
  blocks but also means small real regressions may only WARN. Grow the
  dataset for power.
* Real-provider determinism cannot be guaranteed (hosted APIs); Ledger
  mitigates via temperature 0, seeds, and statistics — not via promises.
* Single-process artifact store (JSONL/JSON files). Fine for CI-scale
  volumes; not a multi-tenant database.
* Judge bias is measured (calibration), not eliminated; per-case rubrics and
  judge≠SUT model reduce but do not remove it.

## Future improvements

* Permutation-test / Bayesian (BEST-style) alternatives behind the existing
  statistics seam; sequential testing for always-valid CI gating.
* Drift detection across the experiment index (metric trend lines over run
  history, alerting on slow slides).
* Slice discovery: automatic tagging/clustering of regressed cases to propose
  new golden-set entries from production failures.
* Real annotation workflow for calibration labels (multi-annotator agreement,
  adjudication) replacing the bundled examples.
* Position-swap and verbosity-controlled judge prompts to actively cancel
  known biases; judge ensembles with disagreement flags.
* Cost/latency budget rules in the policy layer (token spend per run as a
  gated metric).
* Pluggable result-store backends (SQLite/Postgres/S3) behind `ArtifactStore`
  for teams with high run volume.
* Semantic-embedder defaults with local models (e.g. sentence-transformers)
  as an opt-in extra for the sample RAG.

---


