# syntax=docker/dockerfile:1

# Stage 1: dependencies exactly as in uv.lock, the corpus, the index and the embeddings.
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS build
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never HF_HOME=/app/hf
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
# dependencies first: this layer is reused until uv.lock changes
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project
COPY src ./src
RUN uv sync --locked --no-dev --no-editable
COPY scripts/fetch_corpus.py scripts/
RUN uv run --no-sync python scripts/fetch_corpus.py \
 && uv run --no-sync findex index data -o index.bin --workers 2
# downloads the embedding model (~30 MB) into HF_HOME once, at build time
RUN uv run --no-sync findex embed data -o embeddings.npy

# Stage 2: only the virtualenv, the corpus (for snippets), the index, the embeddings and the model.
# No uv, no git, no compiler; the model is never downloaded at run time.
FROM python:3.13-slim-bookworm
RUN useradd --create-home --uid 10001 findex
WORKDIR /app
COPY --from=build --chown=findex /app/.venv /app/.venv
COPY --from=build --chown=findex /app/data /app/data
COPY --from=build --chown=findex /app/index.bin /app/embeddings.npy /app/embeddings.chunks.npy /app/embeddings.json /app/
COPY --from=build --chown=findex /app/hf /app/hf
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/hf \
    HF_HUB_OFFLINE=1 \
    INDEX_PATH=/app/index.bin \
    EMBEDDINGS_PATH=/app/embeddings.npy \
    HOST=0.0.0.0 \
    PORT=8000
USER findex
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health', timeout=4)"
CMD ["findex", "serve"]
