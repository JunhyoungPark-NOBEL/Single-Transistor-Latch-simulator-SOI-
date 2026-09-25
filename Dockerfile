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
# Non-root user; /app must stay writable for the numba caches (engine/**/__pycache__), the FPT node cache
# (engine/photo_extension/photo_nodes) and server/.cache. APP_UID defaults to 1000, as Hugging Face Spaces requires;
# the lab-server kit (deploy/lab) builds with a uid that no account on the host has (STL_APP_UID, default 61000),
# because without user-namespace remapping the container's processes belong to that host uid.
# --no-log-init: a large uid would otherwise blow up /var/log/lastlog and faillog in the image layer.
ARG APP_UID=1000
RUN groupadd -g "$APP_UID" app && useradd --no-log-init -m -u "$APP_UID" -g app app
WORKDIR /app
# requirements go outside /app: copying them to /app/server first would create /app/server as root, and a later
# `COPY --chown` keeps an existing destination directory's owner (server/.cache could then not be created)
COPY server/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
COPY --chown=app:app engine/ engine/
COPY --chown=app:app server/ server/
COPY --chown=app:app scripts/ scripts/
COPY --chown=app:app --from=web /build/web/dist web/dist
# server/.cache/results exists in the image so that a named volume mounted there (deploy/lab) starts out owned by app
RUN mkdir -p server/.cache/results && chown app:app /app /app/server /app/server/.cache /app/server/.cache/results
USER app
# Compile the numba kernels and fill the engine caches at build time (first request is then fast).
RUN python scripts/warmup.py
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/api/health',timeout=4)" || exit 1
# One uvicorn process: the compute process pool (STL_WORKERS) lives inside it.
# FORWARDED_ALLOW_IPS: '*' behind a platform proxy (Render, HF Spaces); 127.0.0.1 when the container is exposed
# directly, so clients cannot spoof X-Forwarded-For (docs/DEPLOY.md); the lab-server kit (deploy/lab) sets it to the
# proxy's internal network
CMD ["sh", "-c", "exec uvicorn server.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips=\"${FORWARDED_ALLOW_IPS:-*}\""]
