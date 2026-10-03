# Helios Data Platform - Overview

Helios Data Platform is a managed cloud data platform for analytics workloads,
built and operated by Helios Data BV, founded in 2019 and headquartered in
Amsterdam, the Netherlands.

## Core components

Helios consists of three primary components:

- **HeliosQL** - the SQL query engine. HeliosQL is wire-compatible with the
  PostgreSQL 15 protocol, supports ANSI SQL:2016 for queries, and adds
  first-class support for semi-structured data (JSONB columns) and time-series
  partitioning.
- **Helios Store** - the columnar storage layer. Data is stored in a
  compressed columnar format with automatic partition pruning and zone maps.
- **Helios Ingest** - managed ingestion pipelines for batch and streaming
  workloads, with connectors for S3, GCS, Azure Blob, Kafka, and PostgreSQL
  logical replication.

## Regions

Helios is available in four regions:

| Region | Location | Notes |
|---|---|---|
| `us-east-1` | Virginia, USA | All features |
| `us-west-2` | Oregon, USA | All features |
| `eu-central-1` | Frankfurt, Germany | All features, EU data residency |
| `ap-southeast-1` | Singapore | All features except time-series tiering (planned) |

Workloads cannot span regions. Cross-region replication is available on the
Enterprise plan only, with a typical replication lag under 60 seconds.

In summary, Helios operates in four regions: us-east-1 in Virginia, us-west-2
in Oregon, eu-central-1 in Frankfurt, and ap-southeast-1 in Singapore.

## Service level objective

The Helios query API carries an uptime SLA of **99.9% per calendar month**,
measured on successful API responses. Enterprise customers can negotiate a
99.95% SLA. SLA credits are applied automatically to the next invoice when
the monthly uptime falls below the commitment; the credit schedule is
published in the service agreement.

## Compute model

Compute runs on auto-scaling clusters. A cluster scales between its configured
minimum and maximum node counts based on query queue depth, with a typical
scale-up latency of 45-90 seconds. Nodes are billed as compute credits (see
the pricing documentation). Idle clusters above the minimum size are scaled
down automatically after 10 minutes without queries.

## Client tooling

Helios provides official SDKs for Python (3.9+), TypeScript (Node 18+), and
Go (1.21+), a command-line tool `heliosctl`, and ODBC/JDBC drivers for BI
tools. All clients authenticate with API keys or SSO tokens as described in
the API reference.
