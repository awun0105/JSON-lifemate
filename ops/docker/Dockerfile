# syntax=docker/dockerfile:1.7
# Phỏng theo Dockerfile@ai-kit (mẫu uv hai giai đoạn); ở đây không có server, cổng hay healthcheck.
#
# Image suy luận của LifeMate (SPEC 6.3, D1/D2). Ban tổ chức gắn split kiểm tra vào /data,
# chạy image với --network none và thu result/submission.jsonl.
#
#   docker build -f ops/docker/Dockerfile --build-arg MODEL_DIR=artifacts/models -t lifemate:dev .   # chạy từ gốc repo
#
# MODEL_DIR là một thư mục trong build context; được sao chép vào /app/models. [llm].model trong
# configs/v0.toml (và [asr].model khi backend = "asr") phải trỏ tới đường dẫn dưới /app/models.
#
# Cách hoạt động:
#   1. Giai đoạn builder: cài CPython 3.11 do uv quản lý, cài thư viện đã khoá rồi cài dự án vào /app/.venv.
#   2. Giai đoạn runtime: image CUDA sạch, chép Python và /app từ builder, chạy bằng user không đặc quyền.
#   3. Ghi VERSION, chép trọng số mô hình, kiểm tra config trỏ đúng mô hình có trong image.
#   4. ENTRYPOINT là ops/docker/entrypoint.sh.

# Image CUDA runtime dùng chung cho cả hai giai đoạn.
ARG CUDA_IMAGE=nvidia/cuda:12.4.1-runtime-ubuntu22.04

# ---------------------------------------------------------------- builder
FROM ${CUDA_IMAGE} AS builder

# Lấy binary uv/uvx từ image chính thức, ghim phiên bản.
COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /uvx /bin/

# uv cài CPython 3.11 do uv quản lý vào /opt/python để giai đoạn runtime sao chép được.
# Biên dịch sẵn bytecode, chép file thay vì hard link (cache nằm ở mount khác).
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app
RUN uv python install 3.11

# Thư viện đã khoá được cài trước: layer này được dùng lại cho tới khi pyproject.toml / uv.lock thay đổi.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project --python 3.11 --extra gpu --extra tokenizer

# Sau đó chép toàn bộ mã nguồn và cài chính dự án.
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --python 3.11 --extra gpu --extra tokenizer

# ---------------------------------------------------------------- runtime
FROM ${CUDA_IMAGE} AS runtime

# Chặn mọi truy cập mạng/telemetry của HF và vLLM (chạy với --network none), log không đệm.
ENV PATH="/app/.venv/bin:${PATH}" \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    DO_NOT_TRACK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    HOME=/home/appuser

# uid cố định, không đặc quyền: thoát khỏi ứng dụng không đồng nghĩa có quyền root trên máy chủ.
RUN useradd --create-home --uid 10001 appuser
WORKDIR /app

# Chép Python được quản lý và ứng dụng (kèm .venv) từ builder.
COPY --from=builder /opt/python /opt/python
COPY --from=builder --chown=appuser:appuser /app /app

# Phiên bản mã cho config.json (image không có .git); Makefile truyền vào commit hiện tại.
ARG CODE_VERSION=unknown
RUN echo "${CODE_VERSION}" > /app/VERSION && chown appuser:appuser /app/VERSION

# Trọng số mô hình được đóng sẵn vào image (lúc chạy không có mạng).
ARG MODEL_DIR=artifacts/models
COPY --chown=appuser:appuser ${MODEL_DIR}/ /app/models/

# Dừng build khi config chạy không có mô hình, hoặc trỏ tới đường dẫn không có trong image.
# configs/docker.toml cần /app/models/llm (và /app/models/asr khi [asr].backend = "asr").
ARG LIFEMATE_CONFIG=configs/docker.toml
ENV LIFEMATE_CONFIG=${LIFEMATE_CONFIG}
RUN python -c "import tomllib,sys,os; c=tomllib.load(open(os.environ['LIFEMATE_CONFIG'],'rb')); m=c['llm']['model']; m or sys.exit('[llm].model is empty in '+os.environ['LIFEMATE_CONFIG']+'; set it to a path under /app/models'); m.startswith('/') and not os.path.isdir(m) and sys.exit('[llm].model '+m+' is not a folder in the image')"

# Cho phép chạy entrypoint, tạo result/ thuộc appuser, rồi chuyển sang user không đặc quyền.
RUN chmod +x /app/ops/docker/entrypoint.sh && mkdir -p /app/result && chown appuser:appuser /app/result
USER appuser

ENTRYPOINT ["/app/ops/docker/entrypoint.sh"]
