# AI-guided Whole Slide Imaging Analysis

This repository contains an AI-guided whole slide imaging (WSI) analysis pipeline
based on **CellViT**, with extensions for **Rosie-based feature fusion**.

The project supports:
- **Inference via Docker** (portable and reproducible)
- **Training via native conda environments** (recommended for SCC usage)

---

## Inference (Docker)

Inference can be performed using the provided Dockerfile.

The Docker setup is intended **only for inference**, ensuring portability and
ease of deployment across systems.

> ⚠️ Training is **not** recommended inside Docker due to GPU, and WandB constraints.

Please refer to the Dockerfile and related instructions for running inference.

---

## Training Setup

Training is performed using a **conda environment derived from the official
CellViT pipeline**, with minimal adjustments for Rosie fusion.

### 1. Environment Setup

The environment definition is provided in:

```bash
environment.yaml
```

This environment has been tested on **Boston University SCC** and local machines.

---

### 1a. Change Directory

```bash
cd CellViT-plus-plus
```

---

## Training on SCC (Boston University)

### Module Setup

```bash
# unload conflicting Python module
module unload python3

# load required modules
module load miniconda/25.3.1
module load cuda/12.2
module load libvips/8.13.0
```

> **Note:**  
> The conda environment includes CUDA runtime libraries.  
> On SCC, CUDA is provided via the module system and has been tested with `cuda/12.2`.

---

### (Optional but Recommended) Configure Conda Paths

This ensures that conda environments and packages are created under the project
space rather than the home directory.

```bash
source setup_scc_condarc.sh
```

---

### Create and Activate the Environment

```bash
conda env create -f environment.yaml
conda activate cellvit_rosie_env
```

---

### Sanity Check

```bash
python check_environment.py
```

This script verifies:
- Required Python packages
- CellViT training and inference modules
- GPU availability (if running on a GPU node)

> If no GPU is detected on a login node, this is expected behavior.

---

## Training on Other Systems (Local / Non-SCC)

```bash
conda env create -f environment.yaml
conda activate cellvit_rosie_env
python check_environment.py
```

---

## Notes

- The environment is intentionally comprehensive to ensure compatibility with
  the full WSI processing and CellViT training pipeline.
- Dependency versions (e.g. `wsidicomizer`, `pathopatch`) are aligned to avoid
  known conflicts.
- `torchaudio` is explicitly included to prevent runtime import issues.

---

## Project Structure

- `CellViT-plus-plus/` – Core CellViT training and inference code
- `environment.yaml` – Conda environment for training
- `Dockerfile` – Inference-only Docker image
- `check_environment.py` – Environment sanity check script

---

## Acknowledgements

This work builds upon the **CellViT** framework and integrates Rosie-based
feature representations for enhanced whole slide image analysis.
