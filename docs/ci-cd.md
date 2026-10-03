# CI/CD for LLM Evaluation

How Ledger plugs into GitHub Actions (or any CI), how the gate decides, and
how to operate it day-to-day.

## 1. Workflows shipped with the repository

| Workflow | Trigger | What it does |
|---|---|---|
| `.github/workflows/ci.yml` | push to `main`, PRs | ruff lint + format check, mypy, full pytest with coverage, golden-dataset validation (strict) |
| `.github/workflows/llm-evaluation.yml` | PRs touching `configs/`, `datasets/`, `data/`, SUT/judge-prompt code; manual dispatch; nightly | evaluates baseline & candidate configs, applies the regression policy, publishes the report to the job summary, uploads artifacts, **fails the check when the gate says FAIL** |

Both workflows run **fully offline by default** using the deterministic mock
provider - no secrets, no cost, no flakiness. Real-provider mode is opt-in
(see §4).

## 2. The gate contract

`ledger gate` (wrapped by `scripts/ci/gate.sh`) implements the decision as
process exit codes:

| Exit code | Meaning | CI effect |
|---|---|---|
| `0` | PASS | green |
| `1` | FAIL | **blocks the merge** |
| `2` | WARNING | green by default; with `--fail-on-warning` becomes `1` |
| `3` | usage/config/dataset error | blocks (infrastructure problem) |

FAIL is produced only by policy rules (see
`docs/evaluation-methodology.md` §5): statistically-supported degradation
beyond configured thresholds, critical-case regressions, count-budget
violations, or an untrustworthy evaluation (judge failure rate too high).
Every rule violation is rendered into the Markdown report and the GitHub job
summary, so a blocked PR explains itself.

### Typical PR flow

```text
PR changes configs/candidate.yaml (new prompt version)
        │
        ▼
llm-evaluation.yml runs
  ├─ lint + unit tests            (fast feedback)
  ├─ ledger evaluate baseline     (cached in mock mode*)
  ├─ ledger evaluate candidate
  ├─ scripts/ci/gate.sh           → exit 0/1/2
  ├─ report → $GITHUB_STEP_SUMMARY
  └─ artifacts → reports/ + results/*/ (30-day retention)
        │
        ▼
FAIL  → PR blocked, violations listed in the job summary
WARN  → mergeable, degradations visible for human review
PASS  → mergeable
```

\* In mock mode the baseline run is cached with `actions/cache` keyed on the
hash of configs + dataset + corpus + `src/ledger/**`; any change invalidates
it. Real-provider mode always re-evaluates the baseline (server-side drift
makes caching unsafe).

## 3. Setting it up on your repository

1. Push this repository (or copy `.github/workflows/`, `scripts/ci/`,
   `configs/`, `datasets/`, `data/`, `src/`).
2. For mock-mode gating: nothing else. It runs on every PR touching the
   listed paths.
3. Adjust the `paths:` filter of `llm-evaluation.yml` to include *your*
   application code (anything that can change model behavior: prompts,
   retrieval settings, model pins, vendored system code).
4. Point `configs/baseline.yaml` at what production runs, and make
   `configs/candidate.yaml` the config your PRs modify.
5. Tune `configs/policy.yaml` thresholds (start with defaults; see the
   methodology doc for a calibration process).

## 4. Real-provider mode (optional, secret-driven)

Create repository secrets (Settings → Secrets and variables → Actions):

| Secret | Purpose |
|---|---|
| `LEDGER_PROVIDER_API_KEY` | SUT generation API key |
| `LEDGER_PROVIDER_BASE_URL` | OpenAI-compatible endpoint (if not default) |
| `LEDGER_PROVIDER_MODEL` | model id for the SUT |
| `LEDGER_JUDGE_API_KEY` | judge API key (a *different*, ideally stronger model recommended) |
| `LEDGER_JUDGE_BASE_URL` / `LEDGER_JUDGE_MODEL` | judge endpoint/model |

Then either:

* run the workflow manually: **Actions → LLM Evaluation Gate → Run
  workflow → use_real_provider = true**, or
* rely on the nightly schedule (it auto-detects whether secrets exist).

Cost/latency controls for real runs:

* `limit` workflow input (e.g. `12`) for smoke gates;
* keep `temperature: 0` and pin exact model versions;
* judge input truncation is bounded by `judge.max_input_chars`;
* the nightly schedule is the right place for full-dataset real runs; PR
  gates can stay on mock mode for plumbing + a small real smoke.

**Security notes**

* Secrets are only injected as environment variables of specific steps; the
  code resolves keys via `api_key_env` names and never writes them to
  artifacts (config snapshots run through `redact_secrets`, verified by
  tests).
* Do not print env in workflows; uploaded artifacts contain evaluations of
  your dataset - treat them with the same confidentiality as the dataset.

## 5. Baseline management strategies

| Strategy | How | When to use |
|---|---|---|
| **Evaluate both every time** (default here) | CI evaluates baseline + candidate per run | mock mode (free), or small datasets on real providers |
| **Cache the baseline** | `actions/cache` keyed on config+dataset+code hashes (already wired for mock mode) | deterministic offline runs |
| **Stored baseline artifact** | run baseline on `main` (nightly), upload `results/` as an artifact/release; PR jobs download it and `ledger gate --baseline <run-id>` | expensive real-provider suites; compare against a *pinned* reference run |

Whichever you choose, the comparator detects dataset-fingerprint mismatches
between the two runs and warns - never compare across golden-dataset edits
without re-baselining.

## 6. Operating the gate day-to-day

* **A FAIL you believe is wrong**: check the violations list first - each
  names the rule, the delta, the CI and p-value. If the delta is real but
  acceptable (a deliberate trade-off), either adjust the policy in the same
  PR (visible, reviewable) or mark affected cases non-critical with a
  comment. Do not silently delete cases.
* **A WARNING**: review the degraded slices in the report; warnings are the
  early band (default: half the fail threshold) and non-significant
  degradations. Treat recurring warnings on the same category as a signal to
  fix or re-scope cases.
* **Flapping decisions on unchanged code**: measure your noise floor
  (baseline-vs-baseline runs). If flips are noise, raise `min_paired_cases`,
  raise thresholds, or move to a larger dataset - in that order.
* **Dataset changes**: PRs touching `datasets/golden.jsonl` re-run
  validation (strict) in `ci.yml`; consider re-running calibration if cases
  feed the label set.

## 7. Running CI locally (parity check)

```bash
make ci-lint          # ruff check + format --check
make ci-test          # pytest with coverage
make ci-eval          # offline gate: baseline vs candidate, mock mode
make demo-fail        # same, with the intentionally degraded candidate → exit 1
```

`scripts/ci/gate.sh` is the exact command CI runs, so local results match
the pipeline (same exit-code contract).
