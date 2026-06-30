# ============================================================================
# CT2MAP-HN Dockerfile
# Multi-stage build: NVIDIA CUDA base → PyTorch + MONAI + all dependencies
# ============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Base image with system dependencies
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS base

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# ---------------------------------------------------------------------------
# Stage 2: Install Python dependencies
# ---------------------------------------------------------------------------
FROM base AS dependencies

WORKDIR /tmp

# Copy requirements first (layer caching)
COPY requirements.txt .

# Install PyTorch with CUDA 12.1
RUN pip install --no-cache-dir \
    torch>=2.1.0 \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 3: Application
# ---------------------------------------------------------------------------
FROM dependencies AS app

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY README.md .
COPY configs/ configs/
COPY ct2map/ ct2map/
COPY scripts/ scripts/
COPY demo/ demo/

# Install project in editable mode
RUN pip install --no-cache-dir -e .

# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Streamlit configuration
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# ---------------------------------------------------------------------------
# Ports
# ---------------------------------------------------------------------------
# Streamlit demo dashboard
EXPOSE 8501
# TensorBoard
EXPOSE 6006

# ---------------------------------------------------------------------------
# Default command: launch Streamlit demo
# ---------------------------------------------------------------------------
CMD ["streamlit", "run", "demo/app.py", "--", "--config", "configs/demo.yaml"]
