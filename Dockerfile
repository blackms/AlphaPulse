# AlphaPulse - Production Dockerfile
# Multi-stage build for optimized image size

# Stage 1: Builder
FROM python:3.12-slim as builder

# Install system dependencies for building Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    libffi-dev \
    libssl-dev \
    libsnappy-dev \
    liblz4-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib C library v0.6.4
RUN wget https://github.com/ta-lib/ta-lib/releases/download/v0.6.4/ta-lib_0.6.4_amd64.deb -O ta-lib_0.6.4_amd64.deb \
    && dpkg -i ta-lib_0.6.4_amd64.deb \
    && rm ta-lib_0.6.4_amd64.deb

# Install Poetry
ENV POETRY_VERSION=1.7.1
RUN pip install --no-cache-dir poetry==${POETRY_VERSION}

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Install dependencies (no dev dependencies)
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --only main

# Stage 2: Runtime
FROM python:3.12-slim

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    libpq5 \
    libffi8 \
    libssl3 \
    libsnappy1v5 \
    liblz4-1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy TA-Lib from builder
COPY --from=builder /usr/lib/x86_64-linux-gnu/libta_lib* /usr/lib/x86_64-linux-gnu/

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create non-root user
RUN groupadd -r alphapulse && useradd -r -g alphapulse -u 1000 alphapulse

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=alphapulse:alphapulse src/ ./src/
COPY --chown=alphapulse:alphapulse config/ ./config/
COPY --chown=alphapulse:alphapulse scripts/ ./scripts/

# Create directories for logs and temp files
RUN mkdir -p /app/logs /app/feature_cache /app/trained_models /tmp/prometheus \
    && chown -R alphapulse:alphapulse /app /tmp/prometheus

# Switch to non-root user
USER alphapulse

# Set environment variables
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command (can be overridden in Kubernetes deployment)
CMD ["python", "-m", "uvicorn", "alpha_pulse.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]