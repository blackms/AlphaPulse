FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml setup.py ./
COPY src ./src/

# Install Python dependencies
RUN pip install --no-cache-dir -e .[dev]

# Create necessary directories
RUN mkdir -p /app/logs /app/feature_cache /app/trained_models

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command to run tests
CMD ["pytest"]

# The actual trading system can be started with different entry points:
# docker run image python -m alpha_pulse.examples.demo_paper_trading
# docker run image python -m alpha_pulse.examples.demo_multi_asset_risk