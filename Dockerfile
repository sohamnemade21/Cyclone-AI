# ==============================================================================
# Multi-Stage / Lean Production Dockerfile for CycloneGuard AI
# ==============================================================================
FROM python:3.11-slim AS base

# Set environment flags
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app" \
    PORT=8000 \
    HOST=0.0.0.0 \
    WORKERS=2 \
    ENVIRONMENT=production

# Install system dependencies & curl for container healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create non-privileged user for security
RUN groupadd -g 1001 appuser && \
    useradd -u 1001 -g appuser -m -s /bin/bash appuser

# Copy dependency definitions and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY CycloneGuard_Bay_of_Bengal_India_Enhanced_Real_Dataset.csv* ./
COPY cycloneguard_xgboost_final.joblib* ./
COPY run_server.py .

# Fix permissions for non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose API port
EXPOSE 8000

# Docker Healthcheck Probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

# Launch production server
CMD ["python", "run_server.py"]
