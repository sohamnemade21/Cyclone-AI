# ==============================================================================
# CycloneGuard AI - Production FastAPI + XGBoost Docker Container
# ==============================================================================
FROM python:3.11-slim

# ------------------------------------------------------------------------------
# 1. Environment Configuration
# ------------------------------------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app" \
    PORT=10000 \
    HOST=0.0.0.0 \
    WORKERS=1 \
    LOG_LEVEL=info \
    ENVIRONMENT=production

# ------------------------------------------------------------------------------
# 2. Install System Dependencies for Health Checks
# ------------------------------------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# ------------------------------------------------------------------------------
# 3. Create Non-Root System User for Security Compliance
# ------------------------------------------------------------------------------
RUN groupadd --system appuser && \
    useradd --system --gid appuser --create-home appuser

# ------------------------------------------------------------------------------
# 4. Install Python Dependencies
# ------------------------------------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------------------------------
# 5. Copy Backend Application & Model Artifacts (Excluding Frontend)
# ------------------------------------------------------------------------------
COPY backend/ ./backend/
COPY cycloneguard_xgboost_final.joblib ./
COPY CycloneGuard_Bay_of_Bengal_India_Enhanced_Real_Dataset.csv* ./
COPY run_server.py ./

# ------------------------------------------------------------------------------
# 6. Set File Ownership & Switch to Non-Root User
# ------------------------------------------------------------------------------
RUN chown -R appuser:appuser /app
USER appuser

# Expose container application port
EXPOSE 10000

# ------------------------------------------------------------------------------
# 7. Container Health Check Probe
# ------------------------------------------------------------------------------
HEALTHCHECK --interval=30s \
    --timeout=5s \
    --start-period=15s \
    --retries=3 \
    CMD curl -f http://localhost:10000/healthz || exit 1

# ------------------------------------------------------------------------------
# 8. Start Production Server
# ------------------------------------------------------------------------------
CMD ["python", "run_server.py"]