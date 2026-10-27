# syntax=docker/dockerfile:1.7
# Adapted from ai-kit Dockerfile@ai-kit (two-stage uv pattern); no server, port or healthcheck here.
#
# LifeMate inference image (SPEC 6.3, D1/D2). Organisers mount the test split at /data,
# run the image with --network none and collect result/submission.jsonl.
#
#   docker build -f ops/docker/Dockerfile --build-arg MODEL_DIR=artifacts/models -t lifemate:dev .   # from the repo root
#
# MODEL_DIR is a folder in the build context; it is copied to /app/models. configs/v0.toml
# [llm].model (and [asr].model when backend = "asr") must point at paths under /app/models.

ARG CUDA_IMAGE=nvidia/cuda:12.4.1-runtime-ubuntu22.04

# ---------------------------------------------------------------- builder
FROM ${CUDA_IMAGE} AS builder

COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /uvx /bin/

# uv installs a managed CPython 3.11 into /opt/python so the runtime stage can copy it.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app
RUN uv python install 3.11

# Locked dependencies first: this layer is reused until pyproject.toml / uv.lock change.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project --python 3.11 --extra gpu --extra tokenizer

COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --python 3.11 --extra gpu --extra tokenizer

# ---------------------------------------------------------------- runtime
FROM ${CUDA_IMAGE} AS runtime

ENV PATH="/app/.venv/bin:${PATH}" \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    DO_NOT_TRACK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    HOME=/home/appuser

# A fixed, unprivileged uid: escaping the app must not mean root on the host.
RUN useradd --create-home --uid 10001 appuser
WORKDIR /app

COPY --from=builder /opt/python /opt/python
COPY --from=builder --chown=appuser:appuser /app /app

# Code version for config.json (the image has no .git); Makefile passes the current commit.
ARG CODE_VERSION=unknown
RUN echo "${CODE_VERSION}" > /app/VERSION && chown appuser:appuser /app/VERSION

# Model weights baked in (no network at run time).
ARG MODEL_DIR=artifacts/models
COPY --chown=appuser:appuser ${MODEL_DIR}/ /app/models/

# Fail the build when the run config has no model, or points at a path that is not in the image.
# configs/docker.toml expects /app/models/llm (and /app/models/asr when [asr].backend = "asr").
ARG LIFEMATE_CONFIG=configs/docker.toml
ENV LIFEMATE_CONFIG=${LIFEMATE_CONFIG}
RUN python -c "import tomllib,sys,os; c=tomllib.load(open(os.environ['LIFEMATE_CONFIG'],'rb')); m=c['llm']['model']; m or sys.exit('[llm].model is empty in '+os.environ['LIFEMATE_CONFIG']+'; set it to a path under /app/models'); m.startswith('/') and not os.path.isdir(m) and sys.exit('[llm].model '+m+' is not a folder in the image')"

RUN chmod +x /app/ops/docker/entrypoint.sh && mkdir -p /app/result && chown appuser:appuser /app/result
USER appuser

ENTRYPOINT ["/app/ops/docker/entrypoint.sh"]
