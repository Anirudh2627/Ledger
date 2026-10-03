# Helios Data Platform - Troubleshooting FAQ

## Common error codes

- **H1001 - authentication failure.** The API key is missing, invalid, or
  expired. Keys expire after 90 days by default. Regenerate the key in
  workspace settings and update your client configuration.
- **H2003 - rate limit exceeded.** Your client exceeded the plan's request
  rate. Honor the `Retry-After` header and apply exponential backoff. See the
  API reference for per-plan limits.
- **H3100 - query timeout.** The query exceeded its timeout (default 30
  seconds). Raise the limit with `SET query_timeout = 120` up to the maximum
  of 300 seconds, or optimize the query.
- **H4022 - insufficient compute credits.** The workspace ran out of credits
  and overage is capped or unavailable. Increase the cap, purchase credits, or
  wait for the monthly reset.

In summary, H1001 is an authentication failure, H2003 means the rate limit was
exceeded, H3100 is a query timeout, and H4022 means insufficient compute
credits.

## Slow queries

For queries slower than expected:

1. Run `EXPLAIN` and look for sequential scans on large tables.
2. Check partition pruning - queries should filter on the partition key.
3. Confirm the cluster has scaled up; queue depth above zero means compute is
   saturated.
4. Materialize repeated aggregations with a scheduled rollup table.

Zone maps accelerate range filters on sorted columns; clustering a table on
its most-filtered column typically reduces scan volume by an order of
magnitude.

## Connection limits

Concurrent connection limits per workspace: **Starter 10, Growth 50,
Enterprise 200**. Connections beyond the limit receive `H1002 - connection
limit reached`. Use the built-in connection pooler (port 6543) to multiplex
many clients over fewer connections.

## Migrating from PostgreSQL

The `pg_migrate` tool copies schema and data from PostgreSQL 12+ with typical
downtime under 5 minutes for databases up to 500 GB. HeliosQL is wire
compatible with PostgreSQL 15, but stored procedures (PL/pgSQL) are not
supported and must be rewritten as client-side logic or scheduled queries.

## Backups and recovery

Automated nightly snapshots are retained for **35 days** on all plans.
Point-in-time recovery (PITR) with a 7-day window is available on Enterprise
plans. Restore operations create a new workspace; in-place overwrite is not
supported to protect against accidental data loss.

## Usage monitoring

Workspace administrators can view compute credit consumption per user and per
query in the Usage dashboard. A daily usage digest email can be enabled in
workspace settings. Alerts can be configured to fire at 50%, 80%, and 100% of
the monthly credit budget.
