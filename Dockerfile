# syntax=docker/dockerfile:1
# STL web simulator: React/Vite frontend (built with Node 22) served by the FastAPI backend (Python 3.11).
#   docker build -t stl-websim .
#   docker run --rm -p 8000:8000 -e STL_WORKERS=2 stl-websim      → http://localhost:8000

# ---------------------------------------------------------------- 1. frontend build
FROM node:22-slim AS web
WORKDIR /build/web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build

# ---------------------------------------------------------------- 2. runtime
FROM python:3.11-slim AS runtime
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PORT=8000
# Non-root user (uid 1000, as Hugging Face Spaces requires); /app must stay writable for the numba caches
# (engine/**/__pycache__), the FPT node cache (engine/photo_extension/photo_nodes) and server/.cache.
RUN useradd -m -u 1000 app
WORKDIR /app
COPY server/requirements.txt server/requirements.txt
RUN pip install -r server/requirements.txt
COPY --chown=app:app engine/ engine/
COPY --chown=app:app server/ server/
COPY --chown=app:app scripts/ scripts/
COPY --chown=app:app --from=web /build/web/dist web/dist
RUN chown app:app /app
USER app
# Compile the numba kernels and fill the engine caches at build time (first request is then fast).
RUN python scripts/warmup.py
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/api/health',timeout=4)" || exit 1
# One uvicorn process: the compute process pool (STL_WORKERS) lives inside it.
CMD ["sh", "-c", "exec uvicorn server.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*'"]
