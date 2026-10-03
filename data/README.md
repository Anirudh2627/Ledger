# Helios corpus (sample application data)

This directory is the knowledge base of the **sample RAG system under test**
(`configs/baseline.yaml` / `configs/candidate.yaml` point at it). It is
application data, not evaluation data: the golden dataset in `datasets/`
questions this corpus.

The documents describe **Helios Data Platform**, a fictional managed data
platform. Facts here are internally consistent and precise so that retrieval
and groundedness are measurable - but Helios itself is invented for this
repository; nothing here describes a real product.

| File | Domain |
|---|---|
| `01-platform-overview.md` | product, regions, SLA, compute model |
| `02-pricing-and-billing.md` | plans, credits, invoices, trials, refunds |
| `03-security-and-compliance.md` | encryption, SSO, RBAC, audit logs, certifications |
| `04-api-reference.md` | endpoints, auth, rate limits, errors, webhooks |
| `05-incident-response.md` | severities, RCA, maintenance, support channels |
| `06-troubleshooting-faq.md` | error codes, slow queries, limits, migrations, backups |

## Swapping in your own corpus

Replace these files with your own `.md`/`.txt` documents (or point
`system.corpus_dir` elsewhere), then update `datasets/golden.jsonl` so test
cases question *your* documents. Keep facts in the corpus specific and
checkable - vague source documents make groundedness unmeasurable.
