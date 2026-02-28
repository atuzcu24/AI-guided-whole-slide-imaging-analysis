# CellViT + SAM-H + Rosie (FiLM) – Inference & Training Guide

This repository documents **inference**, **Docker/Singularity deployment**, and **training** workflows for the **CellViT + SAM-H + Rosie FiLM fusion model**.

**Paths in examples:** Replace `/path/to/repo` with the path to this repository (AI-GUIDED-CLEAN). Use `/path/to/run_dir` for a training run directory, `/path/to/patches/images` for an image folder, etc., as needed.

---

## 1. Easy inference (PanNuke-style patch evaluation)

Run inference on a **PanNuke-style test fold** (folder of images + labels). The script uses the run directory of a trained model (checkpoints + `config.yaml`). You can vary FiLM conditioning for ablations.

**Script:** `CellViT-plus-plus/cellvit/training/evaluate/inference_cellvit_experiment_pannuke.py`

**Example: normal conditioning (default)**

```bash
python CellViT-plus-plus/cellvit/training/evaluate/inference_cellvit_experiment_pannuke.py \
  --run_dir "/path/to/run_dir" \
  --gpu 0 \
  --conditioning_mode normal \
  --log_film_stats
```

**Example: zeros conditioning (FiLM inputs set to zero)**

```bash
python CellViT-plus-plus/cellvit/training/evaluate/inference_cellvit_experiment_pannuke.py \
  --run_dir "/path/to/run_dir" \
  --gpu 0 \
  --conditioning_mode zeros \
  --results_suffix zeros \
  --log_film_stats
```

**Example: shuffle conditioning (FiLM inputs shuffled)**

```bash
python CellViT-plus-plus/cellvit/training/evaluate/inference_cellvit_experiment_pannuke.py \
  --run_dir "/path/to/run_dir" \
  --gpu 0 \
  --conditioning_mode shuffle \
  --results_suffix shuffle \
  --log_film_stats
```

Results are written under `run_dir` (e.g. `inference_results_normal.json`, `inference_results_zeros.json`, `inference_results_shuffle.json`). Set `--run_dir` to your training run directory (the folder that contains `checkpoints/` and `config.yaml`).

---

## 2. WSI-level inference (whole slide)

For **whole-slide images** (e.g. `.svs`), use `detect_cells.py` with the **checkpoint file** (`.pth`) from a training run. Run from the `CellViT-plus-plus` directory so that `cellvit` is on the path.

**Script:** `CellViT-plus-plus/cellvit/detect_cells.py`

**Example: single WSI**

```bash
cd CellViT-plus-plus

python cellvit/detect_cells.py \
  --model /path/to/run_dir/checkpoints/model_best.pth \
  --outdir ./wsi_inference_output \
  --gpu 0 \
  --resolution 0.25 \
  --batch_size 8 \
  --geojson \
  --enforce_amp \
  process_wsi \
  --wsi_path /path/to/slide.svs
```

Outputs (cells, detection, optional GeoJSON) are written under `--outdir`. Use `process_dataset` with `--wsi_folder` or `--filelist` to run on multiple slides; see `python cellvit/detect_cells.py --help`.

---

## 3. Inference (Local Python Environment, legacy patch script)

Input: Folder consisting of 256x256 RGB lung tissue H&E images (preferably png).
Output: The predictions, including type maps, binary maps and cell classifications overlays with legends.

### 3.1 Create and Activate Environment

```bash
source cellvit_inf_env/bin/activate
pip install --upgrade pip
pip install -r requirements_inference.txt
```

### 3.2 Run Inference

```bash
python cellvit_rosie_film_inference.py \
  --input_dir /path/to/patches/images \
  --output_dir ./cellvit_inference_results_film256 \
  --run_dir /path/to/run_dir \
  --checkpoint model_best.pth \
  --magnification 40
```

### 3.3 Model Weights

The contents of `2025-12-05T191217_film256` can be found on HuggingFace:

```
BerkTuzcuBU/SamH-Rosie-FiLM-256
```

---

## 4. Docker Inference

### 4.1 Build Docker Image

```bash
docker buildx build \
  --platform linux/amd64 \
  -t cellvit-inference-amd64 \
  --load \
  .
```

### 4.2 Run with Local `run_dir`

```bash
docker run --rm --gpus all \
  -v /path/to/images:/data/input \
  -v /path/to/output:/data/output \
  -v /path/to/run_dir:/data/run \
  cellvit-inference-amd64
```

### 4.3 Run Using HuggingFace Checkpoints

```bash
docker run --rm --gpus all \
  -e HF_REPO=BerkTuzcuBU/CellViT-FiLM-256-Run \
  -v /path/to/images:/data/input \
  -v /path/to/output:/data/output \
  cellvit-inference-amd64
```

Note: When `HF_REPO` is set, the local `run_dir` is ignored.

---

## 5. Singularity (SCC / HPC)

### 5.1 Convert Docker Image

```bash
docker save cellvit-inference-amd64 > cellvit-inference.tar
singularity build cellvit-inference.sif docker-archive://cellvit-inference.tar
```

### 5.2 Run Inference with Singularity

Singularity **does not create output directories automatically**.

```bash
mkdir -p ./cellvit_inference_results_film256

singularity run --nv \
  -B /path/to/patches/images:/data/input \
  -B /path/to/run_dir:/data/run \
  -B $(pwd)/cellvit_inference_results_film256:/data/output \
  cellvit-inference.sif
```

---

## 6. Training (CellViT Original Pipeline)

### 6.1 Environment Setup on SCC

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

### 6.2 GPU & CUDA Checks

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

### 6.3 libvips (Required)

```bash
module load libvips/8.13.0
```

### 6.4 Sanity Check

```bash
python3 check_environment.py
```

---

## 7. Dataset structure for training (PanNuke-style)

Training expects a **PanNuke-style dataset**: one folder per fold, with images and labels. Create a dataset root (e.g. `pannuke_hf_cellvit`) and inside it:

- **`fold0/`**, **`fold1/`**, **`fold2/`** (or more): each fold contains  
  - **`images/`** — RGB patches (e.g. `.png`), one file per sample  
  - **`labels/`** — one `.npy` per image, same stem as in `images/` (instance map, nuclei types, etc.; see CellViT/PanNuke docs)  
  - **`types.csv`** — column `img` (filename) and `type` (tissue type per image)  
  - **`cell_count.csv`** (optional) — per-image cell counts for sampling

Example layout:

```
pannuke_hf_cellvit/
├── fold0/
│   ├── images/
│   │   ├── image_001.png
│   │   └── ...
│   ├── labels/
│   │   ├── image_001.npy
│   │   └── ...
│   ├── types.csv
│   └── cell_count.csv
├── fold1/
│   ├── images/
│   ├── labels/
│   ├── types.csv
│   └── cell_count.csv
├── fold2/
│   └── ...
└── train_configs/
    └── cellvit_segmentation_virchow_rosie_film.yaml
```

In the **training config YAML** you set `data.dataset_path` to this root and `data.train_folds`, `data.val_folds`, `data.test_folds` to the fold indices (e.g. `[0]`, `[1]`, `[2]`). Section 8 lists the main config keys and how to run training.

---

## 8. Training commands

1. **Create a training config** (YAML) in your dataset’s `train_configs/` folder. Minimal example for Virchow + ROSIE FiLM:

   - `data.dataset_path`: path to the PanNuke-style root (with `fold0/`, `fold1/`, …)  
   - `data.train_folds`, `data.val_folds`, `data.test_folds`: e.g. `[0]`, `[1]`, `[2]`  
   - `model.backbone`: e.g. `virchow-rosie-film`  
   - `model.pretrained_encoder`: path to Virchow encoder  
   - `model.rosie_weights_path`: path to ROSIE weights  
   - `fusion.film_enabled`, `fusion.film_layers`, etc.  
   - `training.batch_size`, `training.optimizer_hyperparameter.lr`, etc.

   You can copy and adapt an existing run’s `config.yaml` from a training output directory (e.g. under `trainings/<run_name>/config.yaml`) or use a config from `ProcessedDataset/.../train_configs/` in this repo as a template.

2. **Run training** from the dataset root (parent of `train_configs/`), pointing at your config:

```bash
cd /path/to/your/pannuke_dataset_root

python /path/to/repo/CellViT-plus-plus/cellvit/train_cellvit.py \
  --config ./train_configs/your_config.yaml
```

Example if your dataset and configs live under a `ProcessedDataset`-style layout inside the repo:

```bash
cd /path/to/repo/ProcessedDataset/your_dataset_folder
```

### 8.1 Patch size 128

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

### 8.2 Patch size 256

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

## 9. Notes

* WSI inference via `detect_cells.py` (Section 2) is supported for single slides and batch processing; use `--geojson` for QuPath-compatible output.
* For additional details, refer to the original **CellViT README.md** in `CellViT-plus-plus`.

## 10. Docker inference (detailed)

This section explains how to build and run the CellViT FiLM inference container using Docker. The Docker image is intended primarily for local testing and for conversion to Singularity on HPC systems. See **Section 4** for a short version.

### 10.1 Build the Docker image

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

### 10.2 Run with local `run_dir`

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

### 10.3 Run using a HuggingFace checkpoint

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

After conversion, see **Section 5** (Singularity) for execution instructions.
