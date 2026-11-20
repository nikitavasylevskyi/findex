# syntax=docker/dockerfile:1

# Stage 1: dependencies exactly as in uv.lock, the corpus and the index.
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS build
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
# dependencies first: this layer is reused until uv.lock changes
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project
COPY src ./src
RUN uv sync --locked --no-dev --no-editable
COPY scripts/fetch_corpus.py scripts/
RUN uv run --no-sync python scripts/fetch_corpus.py \
 && uv run --no-sync findex index data -o index.bin --workers 2

# Stage 2: only the virtualenv, the corpus (for snippets) and the index. No uv, no git, no compiler.
FROM python:3.13-slim-bookworm
RUN useradd --create-home --uid 10001 findex
WORKDIR /app
COPY --from=build --chown=findex /app/.venv /app/.venv
COPY --from=build --chown=findex /app/data /app/data
COPY --from=build --chown=findex /app/index.bin /app/index.bin
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    INDEX_PATH=/app/index.bin \
    HOST=0.0.0.0 \
    PORT=8000
USER findex
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health', timeout=4)"
CMD ["findex", "serve"]
