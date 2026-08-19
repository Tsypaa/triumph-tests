# syntax=docker/dockerfile:1.7
FROM python:3.11-slim AS runtime

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models/huggingface
WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install torch torchvision --index-url "${TORCH_INDEX_URL}"
COPY pyproject.toml README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install ".[birefnet,api]" && \
    useradd --create-home --uid 10001 appuser && mkdir -p /models/huggingface && \
    chown -R appuser:appuser /models

USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10m --retries=3 \
  CMD python -c "import json,urllib.request; d=json.load(urllib.request.urlopen('http://127.0.0.1:8000/api/health',timeout=4)); raise SystemExit(0 if d.get('model_ready') else 1)"
CMD ["uvicorn", "bg_removal.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
