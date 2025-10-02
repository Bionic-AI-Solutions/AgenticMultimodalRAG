# =============================================================================
# AGENTIC MULTIMODAL RAG SYSTEM - PRODUCTION DOCKERFILE
# =============================================================================
# Multi-stage build for optimized production image

# =============================================================================
# BUILD STAGE
# =============================================================================
FROM python:3.11-slim AS builder

# Set build arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry==1.7.1

# Configure Poetry
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Set work directory
WORKDIR /app

# Copy Poetry files
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry install --only=main --no-root && rm -rf $POETRY_CACHE_DIR

# Copy the virtual environment to a known location
RUN cp -r $(poetry env info --path) /app/.venv

# =============================================================================
# RUNTIME STAGE
# =============================================================================
FROM python:3.11-slim AS runtime

# Set build arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

# Set labels
LABEL org.opencontainers.image.title="Agentic Multimodal RAG System" \
      org.opencontainers.image.description="Advanced RAG system with Milvus integration" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.vendor="Bionic-AI-Solutions" \
      org.opencontainers.image.source="https://github.com/Bionic-AI-Solutions/agentic-multimodal-rag"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app:$PYTHONPATH"

# Create non-root user
RUN groupadd -r raguser && useradd -r -g raguser raguser

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    libmagic1 \
    libmagic-dev \
        libgl1-mesa-dri \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/logs /app/tmp /opt/ai-models && \
    chown -R raguser:raguser /app /opt/ai-models

# Switch to non-root user
USER raguser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
