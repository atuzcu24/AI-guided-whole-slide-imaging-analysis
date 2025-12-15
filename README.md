# CellViT + SAM-H + Rosie (FiLM) – Inference & Training Guide

This repository documents **inference**, **Docker/Singularity deployment**, and **training** workflows for the **CellViT + SAM-H + Rosie FiLM fusion model**.

---

## 1. Inference (Local Python Environment)

### 1.1 Create and Activate Environment

```bash
source cellvit_inf_env/bin/activate
pip install --upgrade pip
pip install -r requirements_inference.txt
```

### 1.2 Run Inference

```bash
python cellvit_rosie_film_inference.py \
  --input_dir /projectnb/ec500kb/projects/Fall_2025_Projects/Project_2/AI-guided-whole-slide-imaging-analysis/AI-GUIDED-CLEAN/ProcessedDataset/v1_40x_area20_2/patches_cellvit_p256_pannuke/fold1/images \
  --output_dir ./cellvit_inference_results_film256 \
  --run_dir /projectnb/ec500kb/projects/Fall_2025_Projects/Project_2/AI-guided-whole-slide-imaging-analysis/AI-GUIDED-CLEAN/ProcessedDataset/v1_40x_area20_2/patches_cellvit_p256_pannuke/logs_local_berk/2025-12-05T191217_film256 \
  --checkpoint model_best.pth \
  --magnification 40
```

### 1.3 Model Weights

The contents of `2025-12-05T191217_film256` can be found on HuggingFace:

```
BerkTuzcuBU/SamH-Rosie-FiLM-256
```

---

## 2. Docker Inference

### 2.1 Build Docker Image

```bash
docker buildx build \
  --platform linux/amd64 \
  -t cellvit-inference-amd64 \
  --load \
  .
```

### 2.2 Run with Local `run_dir`

```bash
docker run --rm --gpus all \
  -v /path/to/images:/data/input \
  -v /path/to/output:/data/output \
  -v /path/to/run_dir:/data/run \
  cellvit-inference-amd64
```

### 2.3 Run Using HuggingFace Checkpoints

```bash
docker run --rm --gpus all \
  -e HF_REPO=BerkTuzcuBU/CellViT-FiLM-256-Run \
  -v /path/to/images:/data/input \
  -v /path/to/output:/data/output \
  cellvit-inference-amd64
```

Note: When `HF_REPO` is set, the local `run_dir` is ignored.

---

## 3. Singularity (SCC / HPC)

### 3.1 Convert Docker Image

```bash
docker save cellvit-inference-amd64 > cellvit-inference.tar
singularity build cellvit-inference.sif docker-archive://cellvit-inference.tar
```

### 3.2 Run Inference with Singularity

Singularity **does not create output directories automatically**.

```bash
mkdir -p ./cellvit_inference_results_film256

singularity run --nv \
  -B /projectnb/ec500kb/projects/Fall_2025_Projects/Project_2/AI-guided-whole-slide-imaging-analysis/AI-GUIDED-CLEAN/ProcessedDataset/v1_40x_area20_2/patches_cellvit_p256_pannuke/fold1/images:/data/input \
  -B /projectnb/ec500kb/projects/Fall_2025_Projects/Project_2/AI-guided-whole-slide-imaging-analysis/AI-GUIDED-CLEAN/ProcessedDataset/v1_40x_area20_2/patches_cellvit_p256_pannuke/logs_local_berk/2025-12-05T191217_film256:/data/run \
  -B $(pwd)/cellvit_inference_results_film256:/data/output \
  cellvit-inference.sif
```

---

## 4. Training (CellViT Original Pipeline)

### 4.1 Environment Setup on SCC

```bash
cd CellViT-plus-plus

module unload python3
module load miniconda/25.3.1

setup_scc_condarc.sh
conda --version

conda env create -f environment_verbose.yaml
conda activate cellvit_env
pip install -r requirements.txt
```

### 4.2 GPU & CUDA Checks

```bash
lspci | grep -i nvidia
module list
module avail cuda
module load cuda/12.2
nvcc --version
nvidia-smi
```

If Tesla GPUs are detected:

```bash
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
  --index-url https://download.pytorch.org/whl/cu121
```

### 4.3 libvips (Required)

```bash
module load libvips/8.13.0
```

### 4.4 Sanity Check

```bash
python3 check_environment.py
```

---

## 5. Dataset Structure (Training)

```
ProcessedDataset/
├── fold0/
├── fold1/
├── train_configs/
├── dataset_config.yaml
```

Folds are mapped to train/val/test via config files.

---

## 6. Training Commands

Run from:

```bash
cd /projectnb/ec500kb/projects/Fall_2025_Projects/Project_2/AI-guided-whole-slide-imaging-analysis/AI-GUIDED-CLEAN/ProcessedDataset/v1_40x_area20_2
```

### 6.1 Patch Size 128

```bash
python3 ../../CellViT-plus-plus/cellvit/train_cellvit.py \
  --config ./patches_cellvit_p128_pannuke/train_configs/cellvit_segmentation.yaml
```

Virchow backbone:

```bash
python3 ../../CellViT-plus-plus/cellvit/train_cellvit.py \
  --config ./patches_cellvit_p128_pannuke/train_configs/cellvit_segmentation_virchow.yaml
```

Fusion (SAM-H + Rosie FiLM):

```bash
python3 ../../CellViT-plus-plus/cellvit/train_cellvit.py \
  --config ./patches_cellvit_p128_pannuke/train_configs/samh_rosie_film.yaml
```

### 6.2 Patch Size 256

```bash
python3 ../../CellViT-plus-plus/cellvit/train_cellvit.py \
  --config ./patches_cellvit_p256_pannuke/train_configs/cellvit_segmentation.yaml
```

Virchow backbone:

```bash
python3 ../../CellViT-plus-plus/cellvit/train_cellvit.py \
  --config ./patches_cellvit_p256_pannuke/train_configs/cellvit_segmentation_virchow.yaml
```

Fusion:

```bash
python3 ../../CellViT-plus-plus/cellvit/train_cellvit.py \
  --config ./patches_cellvit_p256_pannuke/train_configs/samh_rosie_film.yaml
```

---

## 7. Notes

* For full WSI inference via `detect_cells.py`, functionality is **experimental and still under development**.
* For additional details, refer to the original **CellViT README.md**.

## Docker Inference

This section explains how to build and run the CellViT FiLM inference container using Docker. The Docker image is intended primarily for local testing and for conversion to Singularity on HPC systems.

### Build the Docker Image

From the root of the repository (where the Dockerfile and `requirements_inference.txt` are located):

```bash
docker buildx build \
  --platform linux/amd64 \
  -t cellvit-inference-amd64 \
  --load \
  .
```

Notes:

* `--platform linux/amd64` is required when building on Apple Silicon (arm64) machines.
* The image includes only inference-time dependencies.

### Run with Local `run_dir`

```bash
docker run --rm --gpus all \
  -v /path/to/images:/data/input \
  -v /path/to/output:/data/output \
  -v /path/to/run_dir:/data/run \
  cellvit-inference-amd64
```

Expected container paths:

* `/data/input`   : directory containing input image patches
* `/data/output`  : directory where inference outputs will be written
* `/data/run`     : CellViT run directory containing `checkpoints/model_best.pth`

### Run Using a HuggingFace Checkpoint

If you prefer to load the model directly from HuggingFace instead of a local `run_dir`:

```bash
docker run --rm --gpus all \
  -e HF_REPO=BerkTuzcuBU/CellViT-FiLM-256-Run \
  -v /path/to/images:/data/input \
  -v /path/to/output:/data/output \
  cellvit-inference-amd64
```

Notes:

* When `HF_REPO` is set, the container ignores `/data/run`.
* The repository must contain the checkpoint and config expected by `cellvit_rosie_film_inference.py`.

### Export Docker Image for SCC / Singularity

To run on BU SCC (or similar HPC systems), convert the Docker image to a Singularity image:

```bash
docker save cellvit-inference-amd64 > cellvit-inference.tar
singularity build cellvit-inference.sif docker-archive://cellvit-inference.tar
```

After conversion, see the **Singularity Inference** section for execution instructions.
