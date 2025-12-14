FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# System deps
RUN apt-get update && apt-get install -y \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps (INFERENCE ONLY)
RUN pip install --upgrade pip && pip install \
    torch==2.1.2 \
    torchvision==0.16.2 \
    numpy \
    pillow \
    matplotlib \
    opencv-python-headless \
    huggingface_hub

# Copy CellViT code
COPY CellViT-plus-plus /app/CellViT-plus-plus
RUN ln -s /app/CellViT-plus-plus/cellvit /app/cellvit

# Copy notebook + scripts
COPY notebooks /app/notebooks

# Optional: cache HF models
ENV HF_HOME=/app/.hf_cache

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
