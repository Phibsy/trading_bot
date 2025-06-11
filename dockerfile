# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs data

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash --uid 1000 trading && \
    chown -R trading:trading /app

# Switch to non-root user
USER trading

# Health check
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "
import sys
try:
    from config.settings import BotConfig
    config = BotConfig()
    if not config.alpaca.api_key:
        sys.exit(1)
    sys.exit(0)
except Exception:
    sys.exit(1)
"

# Default command
CMD ["python", "main.py"]
