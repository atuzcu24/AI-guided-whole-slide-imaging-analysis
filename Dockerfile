# -------------------------------------------------
# Base image: CUDA + cuDNN + Python + PyTorch
# Matches working torch==2.1.2 / CUDA 12.1
# -------------------------------------------------
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# -------------------------------------------------
# Environment settings
# -------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Runtime-configurable paths
ENV INPUT_DIR=/data/input
ENV OUTPUT_DIR=/data/output
ENV RUN_DIR=/data/run
ENV CHECKPOINT=model_best.pth
ENV MAGNIFICATION=40
ENV HF_REPO=""

# -------------------------------------------------
# System dependencies (minimal, inference only)
# -------------------------------------------------
RUN apt-get update && apt-get install -y \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libopenslide0 \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------------------------
# Workdir
# -------------------------------------------------
WORKDIR /app

# -------------------------------------------------
# Copy requirements + code
# -------------------------------------------------
COPY requirements_inference.txt /app/requirements_inference.txt
COPY CellViT-plus-plus /app/CellViT-plus-plus
COPY inference/cellvit_rosie_film_inference.py /app/cellvit_rosie_film_inference.py

# Make CellViT importable
ENV PYTHONPATH=/app/CellViT-plus-plus

RUN apt-get update && apt-get install -y \
    python3-openslide \
    libopenslide0 \
    libopenslide-dev \
    && rm -rf /var/lib/apt/lists/*



# -------------------------------------------------
# Python deps (EXACTLY like your working env)
# -------------------------------------------------
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements_inference.txt

# -------------------------------------------------
# Default command (run_dir OR HF repo)
# -------------------------------------------------
CMD bash -c "\
    if [ -n \"$HF_REPO\" ]; then \
    echo '[INFO] Running inference using HuggingFace repo'; \
    python /app/cellvit_rosie_film_inference.py \
    --input_dir ${INPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --hf_repo ${HF_REPO} \
    --checkpoint ${CHECKPOINT} \
    --magnification ${MAGNIFICATION}; \
    else \
    echo '[INFO] Running inference using local run_dir'; \
    python /app/cellvit_rosie_film_inference.py \
    --input_dir ${INPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --run_dir ${RUN_DIR} \
    --checkpoint ${CHECKPOINT} \
    --magnification ${MAGNIFICATION}; \
    fi"
    