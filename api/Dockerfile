# Read API image.
# Build context is the repository root, because this service imports the shared `common`
# package: docker build -f api/Dockerfile .

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Dependencies first so code edits do not invalidate the pip layer.
COPY api/requirements.txt /app/api/requirements.txt
RUN pip install --no-cache-dir -r /app/api/requirements.txt

# The read API shares the store and feedback logic with the scoring plane. These modules only
# depend on `common` (no ML libraries), so the image stays small; the ML-heavy serving modules are
# present but never imported here.
COPY common/ /app/common/
COPY serving/ /app/serving/
COPY api/ /app/api/

# Run unprivileged.
RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
