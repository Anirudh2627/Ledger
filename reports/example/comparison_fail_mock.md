# Ledger Evaluation Report

**Decision: :x: FAIL**

- Generated: 2026-09-26T22:13:07+00:00 (ledger 0.1.0, report schema 1.0)
- Baseline: `baseline-v1` (run `20260926T221307Z-baseline-v1-d515e8`, model `mock-llm-v1`, prompt `v1_grounding`)
- Candidate: `candidate-degraded-demo` (run `20260926T221307Z-candidate-degraded-demo-b2d182`, model `mock-llm-v1`, prompt `v0_sloppy`)
- Paired cases: 39 (baseline 39, candidate 39)
- Dataset fingerprint match: True

## Overall

| | baseline | candidate | delta | relative | 95% CI | p |
|---|---|---|---|---|---|---|
| **overall** | 0.8985 | 0.3483 | **-0.5502** | -61.2% | [-0.6795, -0.4028] | 0.001 |

## Metrics

| metric | n | baseline | candidate | delta | rel. | CI | p | sig. |
|---|---|---|---|---|---|---|---|---|
| `citation_presence` | 5 | 1.0000 | 0.0000 | -1.0000 | -100.0% | [-1.0000, -1.0000] | 0.001 | :red_circle:
| `exact_match` | 27 | 0.0000 | 0.0000 | +0.0000 | n/a | [+0.0000, +0.0000] | 1.000 |
| `judge_correctness` | 39 | 0.9038 | 0.3333 | -0.5705 | -63.1% | [-0.7436, -0.3716] | 0.001 | :red_circle:
| `judge_groundedness` | 39 | 1.0000 | 0.2821 | -0.7179 | -71.8% | [-0.8462, -0.5641] | 0.001 | :red_circle:
| `judge_instruction_following` | 39 | 0.7372 | 0.4744 | -0.2628 | -35.7% | [-0.3846, -0.1218] | 0.001 | :red_circle:
| `judge_overall` | 39 | 0.8985 | 0.3483 | -0.5502 | -61.2% | [-0.6795, -0.4028] | 0.001 | :red_circle:
| `judge_relevance` | 39 | 0.8462 | 0.3846 | -0.4615 | -54.5% | [-0.5577, -0.3590] | 0.001 | :red_circle:
| `key_term_coverage` | 21 | 1.0000 | 0.0000 | -1.0000 | -100.0% | [-1.0000, -1.0000] | 0.001 | :red_circle:
| `length_compliance` | 1 | 0.0000 | 0.0000 | +0.0000 | n/a | [+0.0000, +0.0000] | 1.000 |
| `normalized_similarity` | 27 | 0.3690 | 0.2432 | -0.1258 | -34.1% | [-0.1998, -0.0532] | 0.001 | :red_circle:
| `overall` | 39 | 0.8985 | 0.3483 | -0.5502 | -61.2% | [-0.6795, -0.4028] | 0.001 | :red_circle:
| `pass_rate` | 39 | 0.9231 | 0.2821 | -0.6410 | -69.4% | [-0.8205, -0.4359] | 0.001 | :red_circle:
| `refusal_match` | 11 | 0.7273 | 1.0000 | +0.2727 | +37.5% | [+0.0000, +0.5455] | 0.068 |
| `retrieval_hit` | 27 | 1.0000 | 0.9444 | -0.0556 | -5.6% | [-0.1481, +0.0000] | 0.262 |
| `structured_output_validity` | 2 | 1.0000 | 0.0000 | -1.0000 | -100.0% | [-1.0000, -1.0000] | 0.001 | :red_circle:
| `token_f1` | 27 | 0.3935 | 0.0522 | -0.3414 | -86.7% | [-0.4114, -0.2693] | 0.001 | :red_circle:

## Categories (overall score)

| category | n | baseline | candidate | delta | rel. | CI | p |
|---|---|---|---|---|---|---|---|
| **critical cases** | 7 | 0.8690 | 0.5833 | -0.2857 | -32.9% | [-0.6786, +0.0893] | 0.143 |
| `adversarial` | 5 | 0.7583 | 0.9583 | +0.2000 | +26.4% | [+0.0000, +0.4000] | 0.167 |
| `ambiguity` | 3 | 0.8611 | 0.4583 | -0.4028 | -46.8% | [-0.7083, +0.0000] | 0.073 |
| `answer_relevance` | 4 | 0.9375 | 0.0833 | -0.8542 | -91.1% | [-0.8750, -0.8333] | 0.001 |
| `edge_case` | 4 | 0.9062 | 0.2917 | -0.6146 | -67.8% | [-0.8542, -0.2188] | 0.004 |
| `factual_correctness` | 6 | 0.9375 | 0.1111 | -0.8264 | -88.1% | [-0.8611, -0.7847] | 0.001 |
| `groundedness` | 4 | 0.9479 | 0.1250 | -0.8229 | -86.8% | [-0.8542, -0.7917] | 0.001 |
| `instruction_following` | 5 | 0.9417 | 0.1000 | -0.8417 | -89.4% | [-0.8750, -0.7917] | 0.001 |
| `refusal` | 4 | 0.8333 | 0.9583 | +0.1250 | +15.0% | [+0.0000, +0.3750] | 0.593 |
| `retrieval_correctness` | 4 | 0.9583 | 0.1042 | -0.8542 | -89.1% | [-0.8750, -0.8125] | 0.001 |

## Test movement

- Regressed: **28**  |  Improved: **3**  |  Stable pass: 8  |  Stable fail: 0
- Newly failing tests: 28 (am-002, am-003, ar-001, ar-002, ar-003, ar-004, ec-001, ec-002, ec-004, fc-001)
- Newly passing tests: 3 (ad-003, ad-005, rf-002)
- Critical regressions: **3** (fc-001, fc-004, rc-004)
- Pass/fail flip sign test: p=0.000 (flips: 31)

## Gate decision

**FAIL** - 9 fail rule(s), 3 warning(s).

| severity | rule | detail |
|---|---|---|
| FAIL | `overall_regression` | overall quality regressed -0.5502 (0.8985 -> 0.3483, CI [-0.6795, -0.4028], p=0.0005) |
| WARNING | `category_nonsignificant` | category 'ambiguity' degraded -0.4028 beyond the configured threshold but lacks statistical support (CI [-0.7083, +0.0000], p=0.073) |
| FAIL | `category_regression` | category 'answer_relevance' regressed -0.8542 (0.9375 -> 0.0833, CI [-0.8750, -0.8333], p=0.0005) |
| FAIL | `category_regression` | category 'edge_case' regressed -0.6146 (0.9062 -> 0.2917, CI [-0.8542, -0.2188], p=0.004) |
| FAIL | `category_regression` | category 'factual_correctness' regressed -0.8264 (0.9375 -> 0.1111, CI [-0.8611, -0.7847], p=0.0005) |
| FAIL | `category_regression` | category 'groundedness' regressed -0.8229 (0.9479 -> 0.1250, CI [-0.8542, -0.7917], p=0.0005) |
| FAIL | `category_regression` | category 'instruction_following' regressed -0.8417 (0.9417 -> 0.1000, CI [-0.8750, -0.7917], p=0.0005) |
| FAIL | `category_regression` | category 'retrieval_correctness' regressed -0.8542 (0.9583 -> 0.1042, CI [-0.8750, -0.8125], p=0.0005) |
| WARNING | `critical_nonsignificant` | critical cases degraded -0.2857 beyond the configured threshold but lacks statistical support (CI [-0.6786, +0.0893], p=0.143) |
| FAIL | `critical_failures` | 3 critical test(s) newly failing (max allowed: 0): fc-001, fc-004, rc-004 |
| WARNING | `metric_nonsignificant` | metric 'retrieval_hit' degraded -0.0556 beyond threshold but is not statistically supported (CI [-0.1481, +0.0000]) |
| FAIL | `newly_failing_tests` | 28 test(s) newly failing (max allowed: 3): am-002, am-003, ar-001, ar-002, ar-003, ar-004, ec-001, ec-002 |

## Top regressed cases

### `ar-002` - answer_relevance (-0.8750)

- Question: How can I subscribe to incident updates for my regions?
- Score: 0.9583 -> 0.0833 | passed: True -> False
- Baseline answer: During incidents, customers should avoid retry storms (use exponential backoff), keep contact details current in workspace settings, and subscribe to the status page for their regions. Live service status is published at `status.helio [...]
- Candidate answer: Based on the provided documents, I don't have enough information to answer this question reliably.
- Judge (candidate): mock-judge heuristics: expects_refusal=False, answer_refuses=True, ref_f1=0.06, threshold=3.5

### `ar-004` - answer_relevance (-0.8750)

- Question: Which ingestion connectors does Helios Ingest support?
- Score: 0.9583 -> 0.0833 | passed: True -> False
- Baseline answer: Data is stored in a compressed columnar format with automatic partition pruning and zone maps. - **Helios Ingest** - managed ingestion pipelines for batch and streaming workloads, with connectors for S3, GCS, Azure Blob, Kafka, and Po [...]
- Candidate answer: Based on the provided documents, I don't have enough information to answer this question reliably.
- Judge (candidate): mock-judge heuristics: expects_refusal=False, answer_refuses=True, ref_f1=0.00, threshold=3.5

### `ec-001` - edge_case (-0.8750)

- Question: What's the monthly price of the Growth plan? 🚀💸 (please answer in English)
- Score: 0.9583 -> 0.0833 | passed: True -> False
- Baseline answer: | Plan | Monthly price | Compute credits / month | Included users | Support | In summary, Starter is priced at $49 per month with 100 compute credits and 3 included users, Growth is priced at $299 per month with 800 credits and 10 use [...]
- Candidate answer: Based on the provided documents, I don't have enough information to answer this question reliably.
- Judge (candidate): mock-judge heuristics: expects_refusal=False, answer_refuses=True, ref_f1=0.09, threshold=3.5

### `fc-002` - factual_correctness (-0.8750)

- Question: How many compute credits does the Growth plan include each month?
- Score: 0.9583 -> 0.0833 | passed: True -> False
- Baseline answer: | Plan | Monthly price | Compute credits / month | Included users | Support | In summary, Starter is priced at $49 per month with 100 compute credits and 3 included users, Growth is priced at $299 per month with 800 credits and 10 use [...]
- Candidate answer: Based on the provided documents, I don't have enough information to answer this question reliably.
- Judge (candidate): mock-judge heuristics: expects_refusal=False, answer_refuses=True, ref_f1=0.08, threshold=3.5

### `fc-004` **[CRITICAL]** - factual_correctness (-0.8750)

- Question: What is the first response time for a SEV1 incident?
- Score: 0.9583 -> 0.0833 | passed: True -> False
- Baseline answer: In summary, SEV1 means the platform is down and receives a first response within 15 minutes, 24/7; SEV2 is a major degradation with a 30-minute response; SEV3 is a minor degradation answered within 4 business hours; and SEV4 covers co [...]
- Candidate answer: Based on the provided documents, I don't have enough information to answer this question reliably.
- Judge (candidate): mock-judge heuristics: expects_refusal=False, answer_refuses=True, ref_f1=0.00, threshold=4.0

### `fc-006` - factual_correctness (-0.8750)

- Question: Which regions does Helios operate in?
- Score: 0.9583 -> 0.0833 | passed: True -> False
- Baseline answer: In summary, Helios operates in four regions: us-east-1 in Virginia, us-west-2 in Oregon, eu-central-1 in Frankfurt, and ap-southeast-1 in Singapore. Regions. Helios is available in four regions: (Sources: 01-platform-overview.md, 05-i [...]
- Candidate answer: Based on the provided documents, I don't have enough information to answer this question reliably.
- Judge (candidate): mock-judge heuristics: expects_refusal=False, answer_refuses=True, ref_f1=0.00, threshold=3.5

### `gr-004` - groundedness (-0.8750)

- Question: What is the default query timeout, and what is the maximum configurable timeout?
- Score: 0.9583 -> 0.0833 | passed: True -> False
- Baseline answer: The default query timeout is **30 seconds**; the maximum configurable timeout is 300 seconds. See the API reference for per-plan limits. - **H3100 - query timeout.** The query exceeded its timeout (default 30 seconds). Raise the limit [...]
- Candidate answer: Based on the provided documents, I don't have enough information to answer this question reliably.
- Judge (candidate): mock-judge heuristics: expects_refusal=False, answer_refuses=True, ref_f1=0.07, threshold=3.5

### `if-001` - instruction_following (-0.8750)

- Question: List the four Helios regions. Respond with a single valid JSON object with the key 'regions' containing an array of region identifiers.
- Score: 0.9583 -> 0.0833 | passed: True -> False
- Baseline answer: {"answer": "In summary, Helios operates in four regions: us-east-1 in Virginia, us-west-2 in Oregon, eu-central-1 in Frankfurt, and ap-southeast-1 in Singapore. Helios is available in four regions: During incidents, customers should a [...]
- Candidate answer: Based on the provided documents, I don't have enough information to answer this question reliably.
- Judge (candidate): mock-judge heuristics: expects_refusal=False, answer_refuses=True, ref_f1=0.00, threshold=3.5

### `if-002` - instruction_following (-0.8750)

- Question: What are the four RBAC workspace roles? Answer in bullet points, one role per bullet.
- Score: 0.9583 -> 0.0833 | passed: True -> False
- Baseline answer: - Authentication and access control. - Single sign-on is supported via **SAML 2.0** and **OpenID Connect (OIDC)** on Growth and Enterprise plans. - **Multi-factor authentication (MFA) is mandatory for all users with the Admin or Owner [...]
- Candidate answer: Based on the provided documents, I don't have enough information to answer this question reliably.
- Judge (candidate): mock-judge heuristics: expects_refusal=False, answer_refuses=True, ref_f1=0.00, threshold=3.5

### `if-005` - instruction_following (-0.8750)

- Question: What is the refund policy for new annual contracts? Cite the source document inline.
- Score: 0.9583 -> 0.0833 | passed: True -> False
- Baseline answer: Subscription fees are refundable pro-rata within the first 30 days of a new annual contract. (Sources: 02-pricing-and-billing.md, 01-platform-overview.md)
- Candidate answer: Based on the provided documents, I don't have enough information to answer this question reliably.
- Judge (candidate): mock-judge heuristics: expects_refusal=False, answer_refuses=True, ref_f1=0.05, threshold=3.5

## Method

- Comparison: paired per-case deltas over 39 common test ids.
- Statistics: paired_percentile bootstrap, 2000 resamples, confidence 0.95, seed 1337.
- Judge overall/dimension scores normalized from 1-5 to 0-1; overall recomputed by Ledger from configured weights (judge-reported aggregates are diagnostics only).

## Notes

- Scores are on a normalized 0-1 scale (judge dimension scores mapped from 1-5).
- Confidence intervals come from a paired percentile bootstrap over test cases.
- Gate decision: FAIL - see violations for the exact rules triggered.

## Artifacts

- `json`: `reports/comparison_baseline-v1_vs_candidate-degraded-demo_20260926T221307Z.json`
- `markdown`: `reports/comparison_baseline-v1_vs_candidate-degraded-demo_20260926T221307Z.md`

---
*Generated by Ledger.*
