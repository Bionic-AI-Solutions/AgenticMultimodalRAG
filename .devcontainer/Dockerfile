# =============================================================================
# DEVCONTAINER DOCKERFILE FOR AGENTIC MULTIMODAL RAG SYSTEM
# =============================================================================
# Development container with all tools and dependencies pre-installed
# Note: This Dockerfile works with devcontainer features which will add
# common-utils, git, and docker-in-docker features

FROM python:3.11-slim

# Ensure we're running as root for the build (features will create vscode user)
USER root

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/workspace \
    CONDA_DIR=/opt/conda \
    HF_HOME=/opt/ai-models \
    TRANSFORMERS_CACHE=/opt/ai-models \
    MODEL_DIR=/opt/ai-models

# Install system dependencies
# Note: git and zsh will be installed by devcontainer features, but we install
# them here too to ensure they're available during build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    wget \
    git \
    vim \
    nano \
    zsh \
    sudo \
    ca-certificates \
    libmagic1 \
    libmagic-dev \
    libgl1-mesa-dri \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    bzip2 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniconda.sh && \
    ${CONDA_DIR}/bin/conda clean -afy && \
    ${CONDA_DIR}/bin/conda init bash zsh

# Add conda to PATH
ENV PATH=${CONDA_DIR}/bin:${PATH}

# Accept Conda Terms of Service and create conda environment 'test' with Python 3.11
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda create -n test python=3.11 -y && \
    conda clean -afy

# Install Poetry (configuration will be set in post-create script as vscode user)
RUN pip install --no-cache-dir poetry==1.7.1

# Install Docker CLI (for docker-compose)
RUN curl -fsSL https://download.docker.com/linux/static/stable/x86_64/docker-24.0.7.tgz | \
    tar -xzC /tmp && \
    mv /tmp/docker/docker /usr/local/bin/ && \
    rm -rf /tmp/docker

# Create directories (models will be mounted from host, but ensure it exists)
# Note: User creation is handled by devcontainer features, so we create directories as root
# and let features handle ownership
RUN mkdir -p /models /opt/ai-models /workspace

# Set working directory
WORKDIR /workspace

# Note: Do not switch to vscode user here - let devcontainer features handle user creation
# The features will be applied after this Dockerfile, and they will create the vscode user
# We'll initialize conda in the post-create script instead

