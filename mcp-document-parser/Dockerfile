FROM python:3.11-slim

# Good defaults for containers
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (kept minimal). If you hit build issues with lxml on slim, uncomment build deps.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
  && rm -rf /var/lib/apt/lists/*

# Install Python deps first (better caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy your application code
COPY . /app

# Expose MCP/HTTP port
EXPOSE 8765

# Start FastAPI
CMD ["uvicorn", "mcp_server.server:app", "--host", "0.0.0.0", "--port", "8765"]