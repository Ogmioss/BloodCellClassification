# Multi-stage build for Blood Cell Classification application
FROM python:3.11-slim AS base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy application code first (needed for package build)
COPY . .

# Install dependencies using uv
RUN uv sync --frozen --no-dev

# Create necessary directories
RUN mkdir -p logs data models/checkpoints src/data/raw src/data/processed

# Setup Kaggle credentials directory
RUN mkdir -p /root/.kaggle

# Copy Kaggle credentials (required for dataset download)
COPY kaggle.json /root/.kaggle/
RUN chmod 600 /root/.kaggle/kaggle.json

# Make load_dataset.sh executable and run it to download dataset
RUN chmod +x scripts/load_dataset.sh
RUN bash scripts/load_dataset.sh || echo "Dataset download skipped (Kaggle credentials may be missing)"

# Expose Streamlit port
EXPOSE 8501

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run Streamlit application
CMD ["uv", "run", "streamlit", "run", "src/app.py"]
