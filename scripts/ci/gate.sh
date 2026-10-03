#!/usr/bin/env bash
# CI quality-gate wrapper for `ledger gate`.
#
# Usage: scripts/ci/gate.sh [args passed through to `ledger gate` ...]
#
# Exit-code contract (propagated verbatim):
#   0 = PASS
#   1 = FAIL
#   2 = WARNING (non-blocking unless you pass --fail-on-warning, which maps
#       WARNING to 1 inside ledger itself)
#
# The workflow renders the Markdown report into $GITHUB_STEP_SUMMARY itself
# (so it also works when the gate FAILs); this script only runs the gate.
set -euo pipefail

exec ledger gate "$@"
