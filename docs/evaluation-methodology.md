# Evaluation Methodology

How Ledger measures quality, why each piece is designed the way it is, and
where the sharp edges are. Read this before trusting (or changing) a gate
decision.

## 1. The golden dataset

### Design principles

1. **Every case carries an evaluation signal.** A case must define at least
   one of: `reference_answer`, `expected_behavior`, `must_contain`, `rubric`,
   or `expected_sources`. The loader rejects cases without any signal -
   unevaluatable cases rot silently otherwise.
2. **Categories map to failure modes, not topics.** The bundled taxonomy
   covers the ways LLM applications actually break:

   | Category | What it detects |
   |---|---|
   | `factual_correctness` | wrong facts after prompt/model changes |
   | `groundedness` | answers drifting away from retrieved context |
   | `answer_relevance` | answering a different question than asked |
   | `retrieval_correctness` | retrieval regressions, isolated from generation |
   | `instruction_following` | format contracts (JSON, bullets, length, citations) |
   | `refusal` | hallucination on unanswerable questions (all `critical`) |
   | `ambiguity` | clarifying vs. guessing behavior |
   | `edge_case` | unicode, long inputs, empty input, degenerate repetition |
   | `adversarial` | prompt injection, social engineering, fabrication pressure |

3. **Critical cases are release blockers.** `critical: true` cases get a
   stricter judge threshold (`critical_pass_threshold`) and their own policy
   band (`policy.critical`, default 2 points) plus a zero-tolerance
   newly-failing rule. Mark a case critical when a wrong answer is
   materially worse than a mediocre one (safety, compliance, security,
   money).
4. **Facts are checkable.** Reference answers contain specific, corpus-
   verifiable values (numbers, names, headers). Vague references make
   lexical metrics and judge calibration meaningless.
5. **The dataset is code.** It lives in git, has a schema
   (`dataset/schema.py`), a validator (`dataset/validator.py`), a CI check
   (`ledger validate-dataset --strict`), and a fingerprint recorded on every
   run. Changing the dataset is a reviewable diff; comparisons across
   dataset versions are flagged automatically.

### Extending it

Append JSONL rows (or edit `datasets/build_golden_dataset.py` and
regenerate). Guidelines that keep the suite healthy:

* One behavior under test per case; give it a stable id (`fc-007`, ...).
* Prefer `expected_behavior` over `reference_answer` for open-ended cases -
  over-specified references punish valid paraphrases.
* Add `expected_sources` whenever retrieval quality matters for the case.
* Use `must_not_contain` for fabrication traps (invented certifications,
  credential-shaped strings).
* Re-run `ledger validate-dataset` and a mock-mode evaluation after edits.

## 2. Scoring model

### The primary signal: judge overall (normalized)

Each case receives four rubric dimension scores, integers 1-5:
**correctness**, **relevance**, **groundedness**, **instruction_following**.
Ledger recomputes a weighted overall (default weights 2/1/2/1) and
normalizes it to 0-1:

```text
overall_norm = (weighted_mean(dimensions) - 1) / 4
```

This normalized overall is the primary regression signal: it is per-case
(hence pairable), bounded, and sensitive to partial degradation. The judge's
*self-reported* `overall`/`pass` fields are parsed for diagnostics only -
Ledger never trusts a model to do its own arithmetic or apply its own
thresholds.

### Per-case pass/fail

* Judge healthy → `passed = weighted_overall >= threshold` (stricter
  threshold for critical cases).
* Judge failed → per `judge.on_failure`: `exclude` falls back to the mean of
  applicable deterministic metrics vs `fallback_pass_threshold`; `fail_case`
  fails the case; `abort` stops the run.
* SUT error → automatic fail (score 0.0), counted as `errored`.

### Deterministic metrics: mechanical checks, honestly scoped

| Metric | Measures | Explicitly does NOT measure |
|---|---|---|
| `exact_match` | normalized string identity vs reference | semantic equivalence |
| `token_f1` | token overlap precision/recall vs reference | correctness of novel phrasing |
| `normalized_similarity` | char-sequence similarity | meaning |
| `key_term_coverage` | presence of `must_contain` terms; hard-fail on `must_not_contain` | whether terms are used correctly |
| `citation_presence` | citation markers when `expect_citations` | whether citations are *true* |
| `retrieval_hit` | recall of `expected_sources` in retrieved chunks | chunk-level precision |
| `refusal_match` | refusal phrasing when refusal expected | refusal *quality* |
| `structured_output_validity` | JSON / markdown-list contracts | schema conformance beyond format |
| `length_compliance` | `metadata.max_words` limits | concision in general |

Rules baked into the implementation: metrics return `None` when not
applicable (excluded from aggregates, never counted as 0), outputs are
clamped to [0,1], and a crashing metric cannot kill a run.

Deterministic metrics are **regression localizers**, not quality oracles.
Example: `token_f1` dropping while the judge holds steady usually means
phrasing changed; `retrieval_hit` dropping pinpoints the retrieval stage
even when end-to-end scores move little.

## 3. LLM-as-a-judge

### Prompt design

* Strict per-dimension anchors (what a 5 vs a 3 vs a 1 means) inside the
  system prompt - a rubric, not a vibe.
* The case's own `rubric` entries are injected per dimension.
* All variable content is XML-tagged (`<question>`, `<context>`,
  `<candidate_answer>`, ...), which improves instruction-following on real
  models and makes the prompt machine-parsable for the offline mock.
* The judge sees no system-version labels (no baseline/candidate priming).
* Output is constrained to a single JSON object; `json_mode` requests
  provider-level JSON mode when available.

### Hardened response handling

| Failure | Handling |
|---|---|
| Prose around JSON / fences / trailing commas | robust extractor (`utils/json_extract.py`) |
| Missing keys, non-numeric, out-of-range scores | validation error → repair retry with the parse error quoted back |
| Persistent malformed output | `JudgeScore(failed=True)` after `max_retries` - never an exception |
| API errors/timeouts | provider-level bounded retries w/ exponential backoff + `Retry-After`, then failed verdict |
| Judge self-math wrong | overall/pass recomputed from dimensions; divergence logged |

### Known judge limitations (why calibration is not optional)

* **Biases**: verbosity preference, self-preference (judge == SUT model),
  position bias, style-over-substance. Mitigations: judge model ≠ SUT model,
  per-case rubrics, recomputed aggregates - but the residual is measurable
  only against humans.
* **Non-determinism**: even at temperature 0 hosted APIs are not guaranteed
  deterministic (batching, MoE routing, silent model updates). Ledger
  therefore treats judge output as *noisy measurement*: statistics operate on
  many cases, and repeated runs on unchanged systems should show deltas
  inside the noise band (verify this! - run baseline-vs-baseline).
* **Drift**: judge model versions change under you. Re-calibrate whenever
  the judge model, prompt, or rubric changes; calibration reports are
  timestamped artifacts for exactly this reason.

## 4. Judge calibration

`ledger calibrate` runs the configured judge over human-labeled
(question, answer) pairs (`datasets/human_labels.jsonl`) and reports:

* per-dimension **MAE / RMSE / bias** (judge − human) on the 1-5 scale,
* **Pearson r** and **Spearman ρ** (rank agreement),
* **agreement-within-1** rate,
* **pass-decision agreement**, **Cohen's κ** with Landis-Koch interpretation,
  and the 2×2 confusion matrix.

Working interpretations (set your own bars):

| Signal | Rough working target |
|---|---|
| MAE per dimension | ≤ 0.5 |
| Agreement within ±1 | ≥ 0.8 |
| Cohen's κ (pass decisions) | ≥ 0.6 ("substantial") |
| Bias | near 0; persistent sign = systematically lenient/harsh |

Why human labels remain the anchor: the judge is the component that turns
text into numbers - if it is wrong, every downstream statistic is precise
nonsense. Calibration does not make the judge right; it makes its error
*measured*, and a measured instrument can be used knowingly (e.g. κ=0.4
means gate on critical-case flips and deterministic metrics, not on 1-point
judge movements).

> The bundled labels are **example annotations authored for this repository**
> (2 mixed-quality labelers, 20 rows) demonstrating the pipeline. Replace
> them with your team's real annotations before drawing conclusions about a
> real judge. Numbers produced with the offline mock judge are plumbing
> checks, not evidence about any real model.

## 5. Statistical regression detection

### Why not `candidate_mean < baseline_mean`?

Single-number comparison ignores sampling noise: with ~40 cases, run-to-run
judge noise alone can move means by more than most "thresholds" people pick.
Naive comparison produces both false alarms (blocking good changes) and
missed regressions (shipping bad ones).

### What Ledger does

1. **Pair per case.** Same dataset ⇒ each case has (baseline, candidate)
   values. Pairing removes between-case variance - the dominant term -
   giving far more power at small n than unpaired tests.
2. **Paired percentile bootstrap** per metric: resample case indices with
   replacement (default 2000 draws, seeded), recompute the mean difference,
   take the 2.5/97.5 percentiles as the CI, and derive a two-sided bootstrap
   p-value (floored at 1/n_samples). No normality assumption; correct for
   bounded, skewed score distributions.
3. **Sign test on pass/fail flips**: exact binomial p-value over newly
   failing vs newly passing cases - assumption-free corroboration.
4. **Policy layer** turns statistics into decisions:
   * `require_significance: true` (default): a beyond-threshold drop only
     FAILs when the CI excludes zero or p < α; otherwise WARNING. This is
     the guard against noise-driven merge blocks.
   * separate bands for overall / categories / critical slice / individual
     metrics; count budgets for newly failing tests and critical failures;
     a judge-failure-rate rule that fails *untrustworthy evaluations*.

### Choosing thresholds

Thresholds are value judgments, not statistics. A defensible starting
process:

1. Run baseline-vs-baseline (unchanged system, twice) a few times → observe
   the noise band of `overall` deltas.
2. Set `overall.absolute` clearly above that band (default 0.03 ≈ 3 points
   on the 0-1 scale).
3. Tighten `critical.absolute` (default 0.02) and keep
   `max_critical_failures: 0`.
4. Keep `require_significance: true` until the dataset is large enough that
   CIs are tight (n ≥ ~100 paired cases).

### Threats to validity (be honest with yourself)

* **Dataset coverage**: the gate is only as good as the golden set. Missing
  categories = blind spots; grow the dataset from production failures.
* **Judge noise floor**: if baseline-vs-baseline deltas approach your
  threshold, the threshold is meaningless - fix measurement first.
* **Multiple comparisons**: with many metrics/categories, some will cross
  thresholds by chance. Ledger mitigates by gating primarily on `overall` +
  criticals + counts, treating per-metric rules as *configured* (opt-in)
  rather than automatic.
* **Goodhart**: any metric that blocks merges will be optimized; keep
  references behavior-level and rotate/extend cases periodically.
* **Mock mode numbers** say nothing about real model quality. They validate
  the harness, determinism, and gate wiring - nothing more.

## 6. Determinism & reproducibility

* Mock provider + hashing embedder + seeded bootstrap ⇒ bit-identical
  offline runs (verified in the test suite).
* Every run records: dataset fingerprint, effective config (redacted),
  seed, ledger/python/platform versions, git commit + dirty flag, prompt
  version, model names.
* Real providers: set `temperature: 0`, pin model versions (avoid `-latest`
  aliases), pass `seed` where supported - and still treat outputs as
  stochastic; that is what the CI/policy layer is for.
