FROM python:3.11-slim AS base

# Good defaults for containers
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# -------------------------
# Builder stage
#   - Compiles any wheels that need native deps (lxml, etc.)
#   - Keeps build tooling out of final image
# -------------------------
FROM base AS builder

# Build deps (only in builder)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
  && rm -rf /var/lib/apt/lists/*

# Install deps as wheels for better caching + smaller runtime image
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
  && python -m pip wheel --wheel-dir /wheels -r /app/requirements.txt


# -------------------------
# Runtime stage
# -------------------------
FROM base

# Runtime deps:
# - ca-certificates: required for outbound TLS (Azure OpenAI)
# - libxml2/libxslt/zlib: safe runtime deps if lxml was built/linked
# - libpq5: safe runtime dep for Postgres drivers
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    libxml2 \
    libxslt1.1 \
    zlib1g \
    libpq5 \
  && update-ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Install Python deps from wheels built in the builder stage
COPY --from=builder /wheels /wheels
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-index --find-links=/wheels -r /app/requirements.txt \
  && rm -rf /wheels

# Copy your application code
COPY . /app

# Optional: run as non-root
RUN useradd -m -u 10001 appuser \
  && chown -R appuser:appuser /app
USER appuser

# Expose MCP/HTTP port
EXPOSE 8765

# Simple container health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import sys,urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/health').read(); sys.exit(0)" || exit 1

# Start FastAPI
CMD ["uvicorn", "mcp_server.server:app", "--host", "0.0.0.0", "--port", "8765"]