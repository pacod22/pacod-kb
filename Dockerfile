# syntax=docker/dockerfile:1
FROM python:3.13-slim

ARG INSTALL_POSTGRES_EXTRA=1

# git: autocommit; pandoc: DOCX conversion; curl: health checks / compose wait
# libxcb1 libx11-6 libgl1 libglib2.0-0: required by OpenCV (docling OCR)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl git pandoc \
        libxcb1 libx11-6 libxext6 libxrender1 libgl1 libglib2.0-0 libsm6 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.8 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

WORKDIR /app

# 1. Deps-only layer (cacheable; no project source needed)
COPY pyproject.toml uv.lock ./
# pacod-id's Python package, vendored (scripts/vendor-pacod-id.sh; [tool.uv.sources]).
COPY vendor/wheels/ vendor/wheels/
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$INSTALL_POSTGRES_EXTRA" = "1" ]; then \
      uv sync --frozen --no-dev --extra postgres --no-install-project; \
    else \
      uv sync --frozen --no-dev --no-install-project; \
    fi

# 2. Source (after deps so this layer doesn't bust the deps cache)
COPY pipeline/ ./pipeline/
# The log set-up both images share (kblog: pacod-demo/docs/LOGGING.md)
COPY kblog/ ./kblog/
COPY eval/ ./eval/
# hatchling validates pyproject's `readme` when building the project
COPY README.md ./

# 3. Install the project itself (generates .venv/bin/kb console script)
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$INSTALL_POSTGRES_EXTRA" = "1" ]; then \
      uv sync --frozen --no-dev --extra postgres; \
    else \
      uv sync --frozen --no-dev; \
    fi

# Put the venv on PATH so `kb` is available as a bare command
ENV PATH="/app/.venv/bin:$PATH"

# HuggingFace cache backed by a named volume — downloaded once, not on every start
ENV HF_HOME=/model-cache

CMD ["kb", "--help"]
