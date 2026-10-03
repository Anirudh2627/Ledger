# Helios Data Platform - Pricing and Billing

## Plans

Helios offers three plans:

| Plan | Monthly price | Compute credits / month | Included users | Support |
|---|---|---|---|---|
| Starter | $49 | 100 | 3 | Email, 2 business days |
| Growth | $299 | 800 | 10 | Email, 1 business day |
| Enterprise | Custom | Negotiated | Unlimited | 24/7 + Slack Connect |

All plans include unlimited read-only viewers, the HeliosQL query engine,
Helios Store, and standard ingestion connectors.

In summary, Starter is priced at $49 per month with 100 compute credits and 3
included users, Growth is priced at $299 per month with 800 credits and 10
users, and Enterprise pricing is custom with unlimited users and 24/7 support.

## Compute credits

One compute credit equals one node-hour of standard compute. Memory-optimized
nodes consume 2 credits per node-hour. Credits reset at the beginning of each
calendar month and **do not roll over** on Starter and Growth plans. Enterprise
customers may negotiate credit rollover in their contract.

Overage is billed at **$0.60 per credit** on Starter and Growth plans.
Overage billing can be capped per workspace in the billing settings; when the
cap is reached, new queries are queued rather than billed, and workspace
administrators receive an email notification.

## Billing cycle and invoices

Billing runs on the calendar month. Invoices are generated on the **1st day of
each month** for the previous month's usage plus the upcoming month's
subscription. Payment terms are **net 15 days** for card payments and net 30
days for Enterprise invoicing. Failed card payments are retried on day 3 and
day 7; workspaces are moved to read-only mode if payment is still outstanding
on day 10.

## Annual billing

Annual prepayment receives a **10% discount** on the subscription fee. Overage
usage is still billed monthly. Downgrading during an annual term takes effect
at renewal; upgrading is prorated immediately.

## Free trial

New workspaces receive a 14-day Growth trial with 200 compute credits. Trials
do not require a credit card. At trial end, workspaces must select a paid plan
or move to read-only mode; data is retained for 90 days in read-only mode
before deletion is scheduled.

## Refunds

Subscription fees are refundable pro-rata within the first 30 days of a new
annual contract. Compute credit overages are not refundable. Refund requests
are handled by billing support and processed within 5 business days.
