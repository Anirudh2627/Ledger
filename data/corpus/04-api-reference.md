# Helios Data Platform - API Reference

## Base URL and versioning

The REST API base URL is `https://api.heliosdata.example/v1`. The API is
versioned by URL path. When a version is deprecated, Helios guarantees at
least **12 months of notice** before shutdown, announced in the changelog and
by email to workspace admins.

## Authentication

All requests require an `Authorization: Bearer <API_KEY>` header. API keys are
created in the workspace settings and expire after 90 days by default.
Requests with missing or expired keys receive `401 unauthorized`.

## Rate limits

Rate limits are enforced per API key:

| Plan | Sustained limit | Burst |
|---|---|---|
| Starter | 60 requests / minute | 2x for up to 10 seconds |
| Growth | 300 requests / minute | 2x for up to 10 seconds |
| Enterprise | 1200 requests / minute | negotiable |

Rate-limited requests receive `429 rate_limited` with a `Retry-After` header
in seconds. Clients should honor `Retry-After` and use exponential backoff.

In summary, the sustained rate limits are 60 requests per minute for Starter,
300 requests per minute for Growth, and 1200 requests per minute for
Enterprise.

## Pagination

List endpoints use cursor-based pagination. The default page size is 50
items; the maximum is 200. Responses include a `next_cursor` field which is
`null` on the final page.

## Standard error codes

| HTTP status | Code | Meaning |
|---|---|---|
| 400 | `invalid_request` | Malformed body or parameters |
| 401 | `unauthorized` | Missing, invalid, or expired API key |
| 403 | `forbidden` | Key lacks permission for this resource |
| 404 | `not_found` | Resource does not exist |
| 429 | `rate_limited` | Rate limit exceeded; see `Retry-After` |
| 500 | `internal_error` | Unexpected server error; safe to retry |

## Webhooks

Webhooks deliver JSON events (query completed, ingest finished, alert fired).
Each delivery is signed with **HMAC-SHA256** using the endpoint's secret; the
signature is sent in the `X-Helios-Signature` header as `t=<timestamp>,v1=<hex
digest>`. Deliveries retry up to 5 times with exponential backoff over 24
hours before the endpoint is disabled.

## Query endpoint (summary)

`POST /v1/queries` accepts `{"sql": "...", "workspace": "...", "timeout_s":
30}`. The default query timeout is **30 seconds**; the maximum configurable
timeout is 300 seconds. Long-running queries return `202 accepted` with a
`query_id` for polling `GET /v1/queries/{query_id}`.
