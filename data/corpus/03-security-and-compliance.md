# Helios Data Platform - Security and Compliance

## Encryption

All data is encrypted **at rest with AES-256** and **in transit with TLS 1.3**
(minimum TLS 1.2 for legacy clients). Encryption keys are managed in a FIPS
140-2 Level 3 validated hardware security module (HSM). Enterprise customers
can bring their own keys (BYOK) through the customer-managed key program,
which integrates with AWS KMS and GCP Cloud KMS.

## Authentication and access control

- Single sign-on is supported via **SAML 2.0** and **OpenID Connect (OIDC)**
  on Growth and Enterprise plans.
- **Multi-factor authentication (MFA) is mandatory for all users with the
  Admin or Owner role.**
- Role-based access control (RBAC) provides four workspace roles: **Viewer,
  Editor, Admin, and Owner**. Permissions are additive and scoped per
  workspace.
- Service accounts use API keys; keys expire after **90 days** by default and
  must be rotated. Key rotation can be automated with the `heliosctl keys
  rotate` command.

## Audit logging

Every API call, query, login, and permission change is written to the audit
log. Audit log retention is **30 days on Starter, 90 days on Growth, and 400
days on Enterprise**. Audit logs can be streamed to a customer S3 bucket or
Splunk HEC endpoint on Enterprise plans.

## Certifications and compliance

Helios maintains:

- **SOC 2 Type II** - audited annually; the current report is available under
  NDA.
- **ISO 27001** - certified, surveillance audits annually.
- **GDPR** - Helios acts as a data processor; a Data Processing Agreement
  (DPA) with Standard Contractual Clauses is available for all customers.

## Data residency

EU customers on the Enterprise plan can enforce EU-only data residency: all
data, backups, and logs remain in the `eu-central-1` (Frankfurt) region.
Residency enforcement is verified by the quarterly access review and covered
by the SOC 2 report.

## Vulnerability management

Helios runs a public vulnerability disclosure program at
`security@heliosdata.example`. Penetration tests are performed twice a year by
an independent firm; executive summaries are shared with Enterprise customers
under NDA. Critical vulnerabilities are patched within 7 days of vendor fix
availability.
