# Helios Data Platform - Incident Response and Support

## Status page

Live service status is published at `status.heliosdata.example`, including
per-region component health, ongoing incidents, and the last 90 days of
history. The status page offers email and webhook subscriptions.

## Severity levels and response times

| Severity | Definition | First response | Updates |
|---|---|---|---|
| SEV1 | Platform unavailable or data loss risk | 15 minutes, 24/7 | every 30 minutes |
| SEV2 | Major feature degraded, workaround limited | 30 minutes, 24/7 | every 2 hours |
| SEV3 | Minor degradation, workaround available | 4 business hours | daily |
| SEV4 | Cosmetic issue or question | 2 business days | as needed |

SEV1 and SEV2 incidents are declared by the on-call incident commander.
Customers can request SEV escalation through support; Helios re-evaluates
severity within 1 hour of the request.

In summary, SEV1 means the platform is down and receives a first response
within 15 minutes, 24/7; SEV2 is a major degradation with a 30-minute
response; SEV3 is a minor degradation answered within 4 business hours; and
SEV4 covers cosmetic issues within 2 business days.

## Root cause analysis

For every SEV1 and SEV2 incident, Helios publishes a root cause analysis (RCA)
within **5 business days** of resolution. RCAs include a timeline, contributing
factors, customer impact, and committed follow-up actions with owners and due
dates. Enterprise customers receive an RCA review call on request.

## Scheduled maintenance

Routine maintenance runs on **Sundays 02:00-04:00 UTC**. Customers are
notified at least **7 days in advance** via the status page and email.
Maintenance that may cause query interruptions is announced 14 days ahead and
requires Enterprise approval for the workspace when it exceeds 30 minutes.

## Support channels

- Starter and Growth: email `support@heliosdata.example`.
- Enterprise: 24/7 phone bridge plus a shared Slack Connect channel with the
  Helios support team and a named technical account manager.

Support scope covers platform availability, ingestion, query behavior, billing
questions, and security reports. Data modeling advice is out of scope for
standard support but available through paid professional services.

## Customer responsibilities

During incidents, customers should avoid retry storms (use exponential
backoff), keep contact details current in workspace settings, and subscribe to
the status page for their regions. Helios never asks customers to share API
keys, and support staff will never request credentials by email or chat.
