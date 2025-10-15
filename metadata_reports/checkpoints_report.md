
---

##  Model Descriptions


###  1. CellViT-VirchowX40
- **Architecture:** Vision Transformer encoder + decoder head  
- **Training:** Supervised on the VirchowX-40 dataset (nucleus instance segmentation)  
- **Input:** 512×512 patches at 40× magnification  
- **Usage:** Standard for multi-class cell segmentation (e.g., epithelial, lymphocyte, macrophage, neutrophil)  
- **Strengths:**
  - Explicit label supervision
  - High Dice and PQ scores across MoNuSAC, CoNSeP, and PanNuke datasets
  - Serves as the baseline CellViT reference

---

### 2. CellViT-SAMX40
- **Architecture:** Same as CellViT, but initialized with SAM-X40’s pretrained ViT encoder  
- **Training:** Distillation → fine-tuning on VirchowX-40 cell labels  
- **Purpose:** Combines foundation-level SAM knowledge (boundaries, context) with supervised cell-type segmentation  
- **Effect:**
  - Faster convergence
  - Better small-cell detection
  - Improved generalization across staining variations  
- **Reported Gains:**
  - +2–4% mean Dice / PQ over CellViT-VirchowX40 baselines  
  - Notably higher accuracy on minority classes (e.g., dead cells)

---

### 3. CellViT-256X40
- **Architecture:** Smaller CellViT variant (ViT-S/B/L)  
- **Training:** 256×256 pixel tiles cropped from VirchowX-40 WSIs  
- **Objective:** Patch-level representation learning (contrastive or supervised)  
- **Purpose:** Efficient feature extraction and foundation initialization for small datasets  
- **Strengths:**
  - Higher spatial detail per nucleus
  - Lightweight (faster on consumer GPUs)
  - Ideal for feature extraction, patch embeddings, or tile classification

---

## 📊 Performance Snapshot (from MedIA 2024 Supplement)

| Dataset | Metric | CellViT-VirchowX40 | CellViT-SAMX40 | CellViT-256X40 |
|:--------|:-------|:------------------:|:---------------:|:---------------:|
| **PanNuke** | Mean Dice | 0.79 | **0.81** | 0.80 |
| **MoNuSAC** | PQ | 0.69 | **0.72** | 0.71 |
| **CoNSeP** | DQ | 0.71 | **0.74** | 0.73 |


---

## 

| Checkpoint | Description |
|-------------|-------------|
| `cellvit-virchowx40.pth` | Supervised model trained on VirchowX-40 |
| `cellvit-samx40.pth` | SAM-informed CellViT model |
| `cellvit-256x40.pth` | Compact CellViT variant with 256×256 input |
| `sam-med2d-vit-h-x40.pth` | SAM model fine-tuned on medical 2D histopathology (VirchowX-40) |



## 🧬 Summary

| Model | Inherits From | Key Idea |
|:------|:---------------|:---------|
| **SAM-X40** | SAM (Meta AI) | Adapted to histopathology; learns universal boundaries |
| **CellViT-VirchowX40** | — | Supervised training on VirchowX-40 for cell-type segmentation |
| **CellViT-SAMX40** | SAM-X40 → CellViT | SAM-informed pretraining + cell-type specialization |
| **CellViT-256X40** | VirchowX-40 (256 px tiles) | Compact, efficient patch-level model for small-data settings |

---



**References**
- *CellViT: Vision Transformers for Multi-Class Cell Segmentation in Histopathology Images*, *Medical Image Analysis (2024)*  
- *Virchow Foundation Model for Pathology*, *bioRxiv (2024)*  
- *SAM-Med2D: Segment Anything in Medical Images*, *arXiv:2310.xxxxx (2024)*
