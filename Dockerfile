# =============================================================================
# Ledger - reproducible container for LLM regression evaluation
# =============================================================================
# Default entrypoint is the `ledger` CLI. Mock mode needs no secrets:
#
#   docker build -t ledger .
#   docker run --rm -v "$PWD/results:/app/results" -v "$PWD/reports:/app/reports" \
#       ledger demo
#
# Real providers: pass env vars (never bake keys into the image):
#
#   docker run --rm --env-file .env \
#       -e LEDGER_PROVIDER_NAME=openai_compat \
#       -v "$PWD/results:/app/results" ledger evaluate --config configs/baseline.yaml
# =============================================================================
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir .

# Evaluation assets (configs, datasets, SUT corpus) and helper scripts.
COPY configs ./configs
COPY datasets ./datasets
COPY data ./data
COPY scripts ./scripts
COPY Makefile ./

# Run as a non-root user.
RUN useradd --create-home --uid 10001 ledger \
    && mkdir -p /app/results /app/reports /app/experiments \
    && chown -R ledger:ledger /app
USER ledger

# Sanity health check: the package imports and the CLI responds.
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import ledger; print(ledger.__version__)" || exit 1

ENTRYPOINT ["ledger"]
CMD ["--help"]
