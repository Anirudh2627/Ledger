# Ledger Evaluation Report

**Decision: :white_check_mark: PASS**

- Generated: 2026-09-26T22:13:09+00:00 (ledger 0.1.0, report schema 1.0)
- Baseline: `baseline-v1` (run `20260926T221309Z-baseline-v1-277850`, model `mock-llm-v1`, prompt `v1_grounding`)
- Candidate: `candidate-v2` (run `20260926T221309Z-candidate-v2-d46718`, model `mock-llm-v1`, prompt `v2_concise_cited`)
- Paired cases: 39 (baseline 39, candidate 39)
- Dataset fingerprint match: True

## Overall

| | baseline | candidate | delta | relative | 95% CI | p |
|---|---|---|---|---|---|---|
| **overall** | 0.8985 | 0.8964 | **-0.0021** | -0.2% | [-0.0064, +0.0000] | 0.735 |

## Metrics

| metric | n | baseline | candidate | delta | rel. | CI | p | sig. |
|---|---|---|---|---|---|---|---|---|
| `citation_presence` | 5 | 1.0000 | 1.0000 | +0.0000 | +0.0% | [+0.0000, +0.0000] | 1.000 |
| `exact_match` | 27 | 0.0000 | 0.0000 | +0.0000 | n/a | [+0.0000, +0.0000] | 1.000 |
| `judge_correctness` | 39 | 0.9038 | 0.8974 | -0.0064 | -0.7% | [-0.0192, +0.0000] | 0.735 |
| `judge_groundedness` | 39 | 1.0000 | 1.0000 | +0.0000 | +0.0% | [+0.0000, +0.0000] | 1.000 |
| `judge_instruction_following` | 39 | 0.7372 | 0.7372 | +0.0000 | +0.0% | [+0.0000, +0.0000] | 1.000 |
| `judge_overall` | 39 | 0.8985 | 0.8964 | -0.0021 | -0.2% | [-0.0064, +0.0000] | 0.735 |
| `judge_relevance` | 39 | 0.8462 | 0.8462 | +0.0000 | +0.0% | [+0.0000, +0.0000] | 1.000 |
| `key_term_coverage` | 21 | 1.0000 | 1.0000 | +0.0000 | +0.0% | [+0.0000, +0.0000] | 1.000 |
| `length_compliance` | 1 | 0.0000 | 0.0000 | +0.0000 | n/a | [+0.0000, +0.0000] | 1.000 |
| `normalized_similarity` | 27 | 0.3690 | 0.3569 | -0.0120 | -3.3% | [-0.0371, +0.0042] | 0.294 |
| `overall` | 39 | 0.8985 | 0.8964 | -0.0021 | -0.2% | [-0.0064, +0.0000] | 0.735 |
| `pass_rate` | 39 | 0.9231 | 0.9231 | +0.0000 | +0.0% | [+0.0000, +0.0000] | 1.000 |
| `refusal_match` | 11 | 0.7273 | 0.7273 | +0.0000 | +0.0% | [+0.0000, +0.0000] | 1.000 |
| `retrieval_hit` | 27 | 1.0000 | 1.0000 | +0.0000 | +0.0% | [+0.0000, +0.0000] | 1.000 |
| `structured_output_validity` | 2 | 1.0000 | 1.0000 | +0.0000 | +0.0% | [+0.0000, +0.0000] | 1.000 |
| `token_f1` | 27 | 0.3935 | 0.3759 | -0.0176 | -4.5% | [-0.0476, +0.0006] | 0.078 |

## Categories (overall score)

| category | n | baseline | candidate | delta | rel. | CI | p |
|---|---|---|---|---|---|---|---|
| **critical cases** | 7 | 0.8690 | 0.8690 | +0.0000 | +0.0% | [+0.0000, +0.0000] | 1.000 |
| `adversarial` | 5 | 0.7583 | 0.7583 | +0.0000 | +0.0% | [+0.0000, +0.0000] | 1.000 |
| `ambiguity` | 3 | 0.8611 | 0.8611 | +0.0000 | +0.0% | [+0.0000, +0.0000] | 1.000 |
| `answer_relevance` | 4 | 0.9375 | 0.9375 | +0.0000 | +0.0% | [+0.0000, +0.0000] | 1.000 |
| `edge_case` | 4 | 0.9062 | 0.9062 | +0.0000 | +0.0% | [+0.0000, +0.0000] | 1.000 |
| `factual_correctness` | 6 | 0.9375 | 0.9375 | +0.0000 | +0.0% | [+0.0000, +0.0000] | 1.000 |
| `groundedness` | 4 | 0.9479 | 0.9479 | +0.0000 | +0.0% | [+0.0000, +0.0000] | 1.000 |
| `instruction_following` | 5 | 0.9417 | 0.9417 | +0.0000 | +0.0% | [+0.0000, +0.0000] | 1.000 |
| `refusal` | 4 | 0.8333 | 0.8333 | +0.0000 | +0.0% | [+0.0000, +0.0000] | 1.000 |
| `retrieval_correctness` | 4 | 0.9583 | 0.9375 | -0.0208 | -2.2% | [-0.0625, +0.0000] | 0.593 |

## Test movement

- Regressed: **1**  |  Improved: **0**  |  Stable pass: 35  |  Stable fail: 3
- Newly failing tests: 0
- Newly passing tests: 0
- Critical regressions: **0**
- Pass/fail flip sign test: p=n/a (flips: 0)

## Gate decision

**PASS** - 0 fail rule(s), 0 warning(s).

## Top regressed cases

### `rc-002` - retrieval_correctness (-0.0833)

- Question: What error code indicates a query timeout, and how can it be mitigated?
- Score: 0.9583 -> 0.8750 | passed: True -> True
- Baseline answer: Common error codes. - **H1001 - authentication failure.** The API key is missing, invalid, or expired. See the API reference for per-plan limits. - **H3100 - query timeout.** The query exceeded its timeout (default 30 seconds). Raise [...]
- Candidate answer: Common error codes. - **H1001 - authentication failure.** The API key is missing, invalid, or expired. See the API reference for per-plan limits. - **H3100 - query timeout.** The query exceeded its timeout (default 30 seconds). Common [...]
- Judge (candidate): mock-judge heuristics: expects_refusal=False, answer_refuses=False, ref_f1=0.27, threshold=3.5

## Method

- Comparison: paired per-case deltas over 39 common test ids.
- Statistics: paired_percentile bootstrap, 2000 resamples, confidence 0.95, seed 1337.
- Judge overall/dimension scores normalized from 1-5 to 0-1; overall recomputed by Ledger from configured weights (judge-reported aggregates are diagnostics only).

## Notes

- Scores are on a normalized 0-1 scale (judge dimension scores mapped from 1-5).
- Confidence intervals come from a paired percentile bootstrap over test cases.

## Artifacts

- `json`: `reports/comparison_baseline-v1_vs_candidate-v2_20260926T221309Z.json`
- `markdown`: `reports/comparison_baseline-v1_vs_candidate-v2_20260926T221309Z.md`

---
*Generated by Ledger.*
