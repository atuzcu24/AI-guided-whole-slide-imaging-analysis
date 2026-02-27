#!/usr/bin/env python3
"""
Create a TCGA→PanNuke label-space compatibility "mapped view" dataset.

- Symlinks fold*/images from original TCGA dataset
- Copies and remaps labels (NPY type_map, CSV csv_labels, types.csv tissue)
- Does NOT modify the original TCGA dataset.
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

# --- Mappings (TCGA → PanNuke) ---
NUCLEI_MAP = {
    0: 0,  # background → Background
    1: 5,  # epithelial → Epithelial
    2: 2,  # lymphocyte → Inflammatory
    3: 2,  # macrophage → Inflammatory
    4: 2,  # neutrophil → Inflammatory
    5: 0,  # other → Background (conservative)
}

# TCGA tissue "Lung" (name) → PanNuke tissue ID 10
# types.csv stores tissue name; dataset_config maps name→ID
# We keep "Lung" in types.csv; dataset_config.yaml will have Lung: 10
TISSUE_NAME_TO_PANNUKE_ID = {"Lung": 10}

# PanNuke label definitions (target space)
PANNUKE_NUCLEI_TYPES = {
    "Background": 0,
    "Neoplastic": 1,
    "Inflammatory": 2,
    "Connective": 3,
    "Dead": 4,
    "Epithelial": 5,
}

PANNUKE_TISSUE_TYPES = {
    "Adrenal_gland": 0,
    "Bile-duct": 1,
    "Bladder": 2,
    "Breast": 3,
    "Cervix": 4,
    "Colon": 5,
    "Esophagus": 6,
    "HeadNeck": 7,
    "Kidney": 8,
    "Liver": 9,
    "Lung": 10,
    "Ovarian": 11,
    "Pancreatic": 12,
    "Prostate": 13,
    "Skin": 14,
    "Stomach": 15,
    "Testis": 16,
    "Thyroid": 17,
    "Uterus": 18,
}


def remap_type_map(type_map: np.ndarray) -> np.ndarray:
    """Remap TCGA nuclei type IDs to PanNuke IDs (vectorized)."""
    out = np.zeros_like(type_map, dtype=type_map.dtype)
    for src, dst in NUCLEI_MAP.items():
        out[type_map == src] = dst
    return out


def create_mapped_dataset(src_root: Path, dst_root: Path) -> None:
    """Create mapped dataset with symlinked images and remapped labels."""
    folds = [0, 1, 2]

    for fold in folds:
        src_fold = src_root / f"fold{fold}"
        dst_fold = dst_root / f"fold{fold}"

        if not src_fold.is_dir():
            raise FileNotFoundError(
                f"TCGA fold folder not found: {src_fold}\n"
                f"Searched: {src_root}\n"
                f"Found: {list(src_root.iterdir())}"
            )

        # --- Images: symlink ---
        src_images = src_fold / "images"
        dst_images = dst_fold / "images"
        dst_images.parent.mkdir(parents=True, exist_ok=True)
        if not dst_images.exists():
            dst_images.symlink_to(src_images.resolve())

        # --- Labels: copy and remap NPY ---
        src_labels = src_fold / "labels"
        dst_labels = dst_fold / "labels"
        dst_labels.mkdir(parents=True, exist_ok=True)

        if not src_labels.is_dir():
            raise FileNotFoundError(f"Labels folder not found: {src_labels}")

        for npy_path in src_labels.glob("*.npy"):
            data = np.load(npy_path, allow_pickle=True)
            obj = data.item()
            inst_map = obj["inst_map"].copy()
            type_map = obj["type_map"].copy()
            type_map_mapped = remap_type_map(type_map)
            new_obj = {"inst_map": inst_map, "type_map": type_map_mapped}
            np.save(dst_labels / npy_path.name, new_obj)

        # --- types.csv: copy (tissue name "Lung" unchanged; config maps to 10) ---
        src_types = src_fold / "types.csv"
        dst_types = dst_fold / "types.csv"
        if src_types.is_file():
            shutil.copy2(src_types, dst_types)
        else:
            raise FileNotFoundError(f"types.csv not found: {src_types}")

        # --- csv_labels: copy and remap type column ---
        src_csv = src_fold / "csv_labels"
        dst_csv = dst_fold / "csv_labels"
        if src_csv.is_dir():
            dst_csv.mkdir(parents=True, exist_ok=True)
            for csv_path in src_csv.glob("*.csv"):
                # Format: no header, columns x,y,type_id
                arr = np.loadtxt(csv_path, delimiter=",", dtype=np.int64, ndmin=2)
                if arr.size > 0:
                    types_old = arr[:, 2]
                    types_new = np.array([NUCLEI_MAP.get(int(t), 0) for t in types_old])
                    arr[:, 2] = types_new
                np.savetxt(dst_csv / csv_path.name, arr, fmt="%d", delimiter=",")
                # Preserve original format if 3 columns
                if arr.size > 0 and arr.ndim == 2 and arr.shape[1] == 3:
                    pass  # already correct
        else:
            # csv_labels optional
            pass

    # --- mapping.yaml in dataset root ---
    mapping = {
        "description": "TCGA to PanNuke label-space mapping for mapped view dataset",
        "nuclei_mapping": {str(k): v for k, v in NUCLEI_MAP.items()},
        "tissue_mapping": TISSUE_NAME_TO_PANNUKE_ID,
        "target_pannuke": {
            "nuclei_types": PANNUKE_NUCLEI_TYPES,
            "tissue_types": PANNUKE_TISSUE_TYPES,
        },
    }
    with open(dst_root / "mapping.yaml", "w") as f:
        yaml.safe_dump(mapping, f, default_flow_style=False)

    # --- dataset_config.yaml (PanNuke label space) ---
    dataset_config = {
        "name": "TCGA_mapped_to_PanNuke",
        "magnification": 40,
        "patch_size": 256,
        "num_classes": 6,
        "stain_normalized": False,
        "description": "TCGA Lung patches with labels mapped to PanNuke label space",
        "tissue_types": PANNUKE_TISSUE_TYPES,
        "nuclei_types": PANNUKE_NUCLEI_TYPES,
    }
    with open(dst_root / "dataset_config.yaml", "w") as f:
        yaml.safe_dump(dataset_config, f, default_flow_style=False)


def main():
    parser = argparse.ArgumentParser(
        description="Create TCGA→PanNuke mapped view dataset"
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=Path(
            "ProcessedDataset/v1_40x_area20_2/patches_cellvit_p256"
        ).resolve(),
        help="Source TCGA dataset root",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path(
            "ProcessedDataset/v1_40x_area20_2/patches_cellvit_p256_tcga_mapped_to_pannuke"
        ).resolve(),
        help="Output mapped dataset root",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    base = script_dir.parent  # AI-GUIDED-CLEAN
    src_root = args.src if args.src.is_absolute() else base / args.src
    dst_root = args.dst if args.dst.is_absolute() else base / args.dst

    if not src_root.is_dir():
        print(
            f"ERROR: Source dataset not found: {src_root}\n"
            f"Searched: {src_root}\n"
            f"Base: {base}",
            file=sys.stderr,
        )
        sys.exit(1)

    dst_root.mkdir(parents=True, exist_ok=True)
    create_mapped_dataset(src_root, dst_root)
    print(f"Created mapped dataset at: {dst_root}")


if __name__ == "__main__":
    main()
