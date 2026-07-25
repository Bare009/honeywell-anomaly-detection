# Scoring service image.
# Build context is the repository root: docker build -f serving/Dockerfile .
#
# Trained artifacts are NOT baked into the image. They are produced on the host by
# training/build_artifacts.py and mounted read-only at /app/artifacts, which keeps the
# image stable while models are retrained.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /app

# libgomp is required by LightGBM at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first so code edits do not invalidate the (large) pip layer.
COPY serving/requirements.txt /app/serving/requirements.txt
RUN pip install --no-cache-dir -r /app/serving/requirements.txt

COPY common/ /app/common/
COPY features/ /app/features/
COPY models/ /app/models/
COPY explainability/ /app/explainability/
COPY serving/ /app/serving/

# Mount point for host-built artifacts (read-only in docker-compose).
RUN mkdir -p /app/artifacts

RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8100

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8100/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8100"]
