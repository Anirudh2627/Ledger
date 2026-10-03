# =============================================================================
# Ledger - developer workflow
# =============================================================================
# Run `make help` for a summary. Most targets assume the package is installed
# in editable mode (make install). Everything runs offline by default using
# the deterministic mock provider.
# =============================================================================

PYTHON ?= python3
VENV   ?= .venv
PIP    := $(VENV)/bin/pip
PY     := $(VENV)/bin/python
LEDGER := $(VENV)/bin/ledger

BASELINE_CONFIG   ?= configs/baseline.yaml
CANDIDATE_CONFIG  ?= configs/candidate.yaml
POLICY_CONFIG     ?= configs/policy.yaml
DATASET           ?= datasets/golden.jsonl

.PHONY: help install install-dev lint format typecheck test test-unit test-integration \
        coverage validate evaluate-baseline evaluate-candidate evaluate-degraded \
        compare gate demo demo-fail calibrate report list-runs clean \
        docker-build docker-demo ci-lint ci-test ci-eval

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
install: ## Create venv and install runtime dependencies
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

install-dev: ## Create venv and install runtime + dev dependencies
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e ".[dev]"

# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------
lint: ## Ruff lint + format check
	$(VENV)/bin/ruff check src tests scripts
	$(VENV)/bin/ruff format --check src tests scripts

format: ## Auto-format with ruff
	$(VENV)/bin/ruff format src tests scripts
	$(VENV)/bin/ruff check --fix src tests scripts

typecheck: ## Static type checking with mypy
	$(VENV)/bin/mypy

test: ## Run all tests
	$(VENV)/bin/pytest

test-unit: ## Run unit tests only
	$(VENV)/bin/pytest -m unit

test-integration: ## Run integration tests only
	$(VENV)/bin/pytest -m integration

coverage: ## Tests with coverage report
	$(VENV)/bin/pytest --cov --cov-report=term-missing

ci-lint: ## CI: non-destructive lint checks
	$(VENV)/bin/ruff check src tests scripts
	$(VENV)/bin/ruff format --check src tests scripts

ci-test: ## CI: tests with coverage
	$(VENV)/bin/pytest --cov --cov-report=term-missing --cov-report=xml

# ---------------------------------------------------------------------------
# Evaluation workflow (offline mock mode by default)
# ---------------------------------------------------------------------------
validate: ## Validate the golden dataset schema
	$(LEDGER) validate-dataset $(DATASET)

evaluate-baseline: ## Evaluate the baseline system version
	$(LEDGER) evaluate --config $(BASELINE_CONFIG)

evaluate-candidate: ## Evaluate the candidate system version
	$(LEDGER) evaluate --config $(CANDIDATE_CONFIG)

evaluate-degraded: ## Evaluate an intentionally degraded candidate (gate demo)
	$(LEDGER) evaluate --config configs/candidate_degraded.yaml

compare: ## Compare the two most recent baseline/candidate runs and write reports
	$(LEDGER) compare --latest --policy $(POLICY_CONFIG)

gate: ## Run comparison as a CI quality gate (non-zero exit on FAIL)
	$(LEDGER) gate --latest --policy $(POLICY_CONFIG)

demo: ## Full offline demo: evaluate baseline + candidate, compare, gate, report
	$(LEDGER) demo

demo-fail: ## Demo of the gate BLOCKING an intentionally degraded candidate
	$(LEDGER) demo --candidate-config configs/candidate_degraded.yaml

calibrate: ## Run judge calibration against human labels
	$(LEDGER) calibrate --labels datasets/human_labels.jsonl --config $(BASELINE_CONFIG)

report: ## Render a report for the most recent run
	$(LEDGER) report --latest

list-runs: ## List recorded evaluation runs
	$(LEDGER) list-runs

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------
docker-build: ## Build the Ledger container image
	docker build -t ledger:latest .

docker-demo: ## Run the offline demo inside the container
	docker compose run --rm ledger demo

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
clean: ## Remove caches and build artifacts (keeps results/reports)
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov coverage.xml
	find . -type d -name __pycache__ -not -path "./$(VENV)/*" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -not -path "./$(VENV)/*" -exec rm -rf {} + 2>/dev/null || true

# CI convenience aggregate used by the evaluation workflow
ci-eval: ## CI: offline evaluation gate (baseline vs candidate, mock mode)
	LEDGER_FORCE_MOCK=1 $(LEDGER) gate --config-baseline $(BASELINE_CONFIG) \
		--config-candidate $(CANDIDATE_CONFIG) --policy $(POLICY_CONFIG)
