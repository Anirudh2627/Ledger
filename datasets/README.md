# Datasets

Evaluation data for Ledger. Two files live here; both are **versioned test
fixtures, not training data**.

| File | Purpose | Consumed by |
|---|---|---|
| `golden.jsonl` | The golden evaluation dataset: 39 structured test cases across 9 failure-mode categories (7 marked `critical`) | `ledger evaluate` (all runs) |
| `human_labels.jsonl` | Human-labeled (question, answer) pairs for judge calibration - 20 rows, balanced pass/fail, two mock labelers | `ledger calibrate` |
| `build_golden_dataset.py` | Maintenance tool that regenerates `golden.jsonl` deterministically | developers (bulk edits) |

> **Provenance note.** Both files were authored for this repository against
> the fictional *Helios Data Platform* corpus in `data/corpus/`. The human
> labels are *example annotations* demonstrating the calibration pipeline -
> replace them with your team's real labels before drawing conclusions about
> a real judge model.

## golden.jsonl schema (one JSON object per line)

```jsonc
{
  "id": "fc-001",                    // unique, stable across versions
  "question": "...",                 // input to the system under test
  "context": null,                   // optional static context (RAG ignores)
  "reference_answer": "...",         // gold answer (optional for refusal/ambiguity)
  "expected_behavior": "...",        // behavioral contract shown to the judge
  "category": "factual_correctness", // failure-mode taxonomy (free-form, validated)
  "critical": true,                  // release-blocker semantics (stricter thresholds)
  "rubric": {                        // optional per-case scoring criteria
    "correctness": "Must name AES-256 ...",
    "groundedness": "..."
  },
  "expected_sources": ["03-security-and-compliance.md"], // retrieval_hit metric
  "must_contain": ["aes-256"],       // key_term_coverage metric
  "must_not_contain": [],            // fabrication traps (hard-fail)
  "expect_citations": true,          // citation_presence metric
  "tags": ["security"],              // free-form slicing
  "difficulty": "easy",              // easy | medium | hard (optional)
  "metadata": {}                     // e.g. {"output_format": "json", "max_words": 12}
}
```

Every case must carry at least one evaluation signal (`reference_answer`,
`expected_behavior`, `must_contain`, `rubric`, or `expected_sources`) - the
loader rejects cases without any.

## Category coverage in golden.jsonl

| Category | n | Notes |
|---|---|---|
| `factual_correctness` | 6 | 2 critical (security/incident facts) |
| `groundedness` | 4 | citation-required cases |
| `answer_relevance` | 4 | distractor-prone questions |
| `retrieval_correctness` | 4 | 1 critical; scored via `expected_sources` |
| `instruction_following` | 5 | JSON, bullets, one-sentence, word limit, citations |
| `refusal` | 4 | all critical; unanswerable/PII/financial questions |
| `ambiguity` | 3 | clarify-or-assume behavior |
| `edge_case` | 4 | unicode, verbose framing, empty input, token repetition |
| `adversarial` | 5 | prompt injection, social engineering, fabrication pressure, harmful content |

## Editing rules

1. Edit `build_golden_dataset.py` (or the JSONL directly) and regenerate.
2. Run `ledger validate-dataset datasets/golden.jsonl --strict`.
3. Run a mock-mode evaluation (`make demo`) to see score effects for free.
4. Changing cases changes the dataset fingerprint; comparisons across the
   change are flagged automatically - re-baseline when you merge.
