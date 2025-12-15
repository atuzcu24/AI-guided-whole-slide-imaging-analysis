# -------------------------------------------------
# Base image (CUDA + cuDNN)
# -------------------------------------------------
    FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

    # -------------------------------------------------
    # Environment
    # -------------------------------------------------
    ENV DEBIAN_FRONTEND=noninteractive
    ENV PYTHONUNBUFFERED=1
    ENV PYTHONDONTWRITEBYTECODE=1
    
    # These will be overridden at runtime if needed
    ENV INPUT_DIR=/data/input
    ENV OUTPUT_DIR=/data/output
    ENV RUN_DIR=/data/run
    ENV CHECKPOINT=model_best.pth
    ENV MAGNIFICATION=40
    ENV HF_REPO=""
    
    # -------------------------------------------------
    # System dependencies
    # -------------------------------------------------
    RUN apt-get update && apt-get install -y \
        python3 \
        python3-pip \
        python3-dev \
        git \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        && rm -rf /var/lib/apt/lists/*
    
    # -------------------------------------------------
    # Python dependencies (INFERENCE ONLY)
    # -------------------------------------------------
    RUN python3 -m pip install --upgrade pip && \
        python3 -m pip install \
        torch==2.1.2 \
        torchvision==0.16.2 \
        numpy \
        pillow \
        matplotlib \
        opencv-python-headless \
        tqdm \
        huggingface_hub
    
    # -------------------------------------------------
    # Workdir
    # -------------------------------------------------
    WORKDIR /app
    
    # -------------------------------------------------
    # Copy CellViT code + inference script
    # -------------------------------------------------
    # Assumes:
    #   ./CellViT-plus-plus
    #   ./cellvit_inference_folder.py
    COPY CellViT-plus-plus ./CellViT-plus-plus
    COPY cellvit_inference_folder.py ./cellvit_inference_folder.py
    
    # -------------------------------------------------
    # Default command
    # -------------------------------------------------
    CMD bash -c "\
      if [ -n \"$HF_REPO\" ]; then \
        echo '[INFO] Running inference using HuggingFace repo'; \
        python3 cellvit_inference_folder.py \
          --input_dir ${INPUT_DIR} \
          --output_dir ${OUTPUT_DIR} \
          --hf_repo ${HF_REPO} \
          --checkpoint ${CHECKPOINT} \
          --magnification ${MAGNIFICATION}; \
      else \
        echo '[INFO] Running inference using local run_dir'; \
        python3 cellvit_inference_folder.py \
          --input_dir ${INPUT_DIR} \
          --output_dir ${OUTPUT_DIR} \
          --run_dir ${RUN_DIR} \
          --checkpoint ${CHECKPOINT} \
          --magnification ${MAGNIFICATION}; \
      fi"
    