#!/usr/bin/env python3
"""
Verification for TCGA→PanNuke mapped dataset.

- Prints per-fold class counts before vs after mapping (CSV and NPY)
- Prints tissue ID counts before vs after
- Visualizes 5 random samples: RGB, original type_map, mapped type_map with legends
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np

# Same mapping as create script
NUCLEI_MAP = {
    0: 0, 1: 5, 2: 2, 3: 2, 4: 2, 5: 0,
}
TCGA_NUCLEI_NAMES = {
    0: "background", 1: "epithelial", 2: "lymphocyte",
    3: "macrophage", 4: "neutrophil", 5: "other",
}
PANNUKE_NUCLEI_NAMES = {
    0: "Background", 1: "Neoplastic", 2: "Inflammatory",
    3: "Connective", 4: "Dead", 5: "Epithelial",
}


def count_npy_types(fold_path: Path) -> dict:
    """Count nuclei type occurrences in NPY type_maps."""
    labels_dir = fold_path / "labels"
    if not labels_dir.is_dir():
        return {}
    counts = {}
    for npy_path in labels_dir.glob("*.npy"):
        data = np.load(npy_path, allow_pickle=True)
        obj = data.item()
        tm = obj["type_map"]
        for v in np.unique(tm):
            v = int(v)
            counts[v] = counts.get(v, 0) + int((tm == v).sum())
    return counts


def count_csv_types(fold_path: Path, apply_map: bool = False) -> dict:
    """Count nuclei type from csv_labels. apply_map: apply NUCLEI_MAP to values."""
    csv_dir = fold_path / "csv_labels"
    if not csv_dir.is_dir():
        return {}
    counts = {}
    for csv_path in csv_dir.glob("*.csv"):
        try:
            arr = np.loadtxt(csv_path, delimiter=",", dtype=np.int64, ndmin=2)
        except Exception:
            continue
        if arr.size == 0:
            continue
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] < 3:
            continue
        types = arr[:, 2]
        for t in types:
            t = int(t)
            if apply_map:
                t = NUCLEI_MAP.get(t, 0)
            counts[t] = counts.get(t, 0) + 1
    return counts


def count_tissues(types_csv: Path) -> dict:
    """Count tissue types from types.csv (by name)."""
    if not types_csv.is_file():
        return {}
    counts = {}
    with open(types_csv) as f:
        lines = f.readlines()
    # skip header
    for line in lines[1:]:
        parts = line.strip().split(",")
        if len(parts) >= 2:
            t = parts[1].strip()
            counts[t] = counts.get(t, 0) + 1
    return counts


def load_image(img_path: Path) -> np.ndarray:
    try:
        from PIL import Image
        return np.array(Image.open(img_path))
    except ImportError:
        return None


# Simple color palette for type maps (R,G,B) index 0-5
TYPE_COLORS = np.array([
    [0, 0, 0],       # 0 background - black
    [255, 0, 0],     # 1 - red
    [0, 255, 0],     # 2 - green
    [0, 0, 255],     # 3 - blue
    [255, 255, 0],   # 4 - yellow
    [255, 0, 255],   # 5 - magenta
], dtype=np.uint8)


def type_map_to_rgb(type_map: np.ndarray) -> np.ndarray:
    """Convert type_map (H,W) to RGB (H,W,3) using TYPE_COLORS."""
    h, w = type_map.shape
    flat = np.clip(type_map.ravel().astype(np.int64), 0, 5)
    rgb = TYPE_COLORS[flat].reshape(h, w, 3)
    return rgb


def visualize_samples(
    src_root: Path,
    dst_root: Path,
    n_samples: int = 5,
    out_dir: Path = None,
) -> None:
    """Visualize random samples: RGB, original type_map, mapped type_map."""
    fold0_src = src_root / "fold0"
    fold0_dst = dst_root / "fold0"
    labels_src = list((fold0_src / "labels").glob("*.npy"))
    if not labels_src:
        print("No labels in fold0", file=sys.stderr)
        return

    random.seed(42)
    chosen = random.sample(labels_src, min(n_samples, len(labels_src)))

    if out_dir is None:
        out_dir = dst_root / "verification_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from PIL import Image
    except ImportError:
        print("PIL not available, skipping visualizations", file=sys.stderr)
        return

    for i, npy_path in enumerate(chosen):
        stem = npy_path.stem
        img_path = fold0_src / "images" / (stem + ".png")
        if not img_path.is_file():
            matches = list((fold0_src / "images").glob(f"{stem}*"))
            img_path = matches[0] if matches else None
        if img_path is None or not img_path.is_file():
            continue

        img = load_image(img_path)
        data_src = np.load(npy_path, allow_pickle=True)
        type_src = data_src.item()["type_map"]

        npy_dst = fold0_dst / "labels" / npy_path.name
        data_dst = np.load(npy_dst, allow_pickle=True)
        type_dst = data_dst.item()["type_map"]

        # Build composite: RGB | original type_map | mapped type_map
        h, w = type_src.shape
        gap = 4
        composite = np.zeros((h, w * 3 + gap * 2, 3), dtype=np.uint8)
        composite[:, :, :] = 128  # gray background
        if img is not None:
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            composite[:, :w] = img[:h, :w, :3]
        composite[:, w + gap : 2 * w + gap] = type_map_to_rgb(type_src)
        composite[:, 2 * w + 2 * gap : 3 * w + 2 * gap] = type_map_to_rgb(type_dst)

        out_path = out_dir / f"sample_{i+1}_{stem[:40]}.png"
        Image.fromarray(composite).save(out_path)
        print(f"  Saved: {out_path}")
        # Also save a legend text file
        leg_path = out_dir / f"sample_{i+1}_legend.txt"
        with open(leg_path, "w") as f:
            f.write(f"Sample {i+1}: {stem}\n")
            f.write("Columns: RGB | Original (TCGA) | Mapped (PanNuke)\n")
            f.write("TCGA: 0=bg,1=epith,2=lymph,3=macro,4=neutro,5=other\n")
            f.write("PanNuke: 0=Background,2=Inflammatory,5=Epithelial\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        type=Path,
        default=Path(
            "ProcessedDataset/v1_40x_area20_2/patches_cellvit_p256"
        ),
        help="Original TCGA dataset root",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path(
            "ProcessedDataset/v1_40x_area20_2/patches_cellvit_p256_tcga_mapped_to_pannuke"
        ),
        help="Mapped dataset root",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip visualizations",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    base = script_dir.parent
    src_root = base / args.src
    dst_root = base / args.dst

    if not src_root.is_dir():
        print(f"ERROR: Source not found: {src_root}", file=sys.stderr)
        sys.exit(1)
    if not dst_root.is_dir():
        print(f"ERROR: Mapped dataset not found: {dst_root}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("Per-fold nuclei class counts (NPY type_map)")
    print("=" * 60)
    for fold in [0, 1, 2]:
        src_fold = src_root / f"fold{fold}"
        dst_fold = dst_root / f"fold{fold}"
        if not src_fold.is_dir():
            continue
        print(f"\n--- Fold {fold} ---")
        cnt_src = count_npy_types(src_fold)
        cnt_dst = count_npy_types(dst_fold)
        print("Original (TCGA):")
        for k in sorted(cnt_src.keys()):
            print(f"  {TCGA_NUCLEI_NAMES.get(k, str(k))}: {cnt_src[k]}")
        print("Mapped (PanNuke):")
        for k in sorted(cnt_dst.keys()):
            print(f"  {PANNUKE_NUCLEI_NAMES.get(k, str(k))}: {cnt_dst[k]}")

    print("\n" + "=" * 60)
    print("Per-fold nuclei class counts (CSV csv_labels)")
    print("=" * 60)
    for fold in [0, 1, 2]:
        src_fold = src_root / f"fold{fold}"
        dst_fold = dst_root / f"fold{fold}"
        if not src_fold.is_dir():
            continue
        print(f"\n--- Fold {fold} ---")
        cnt_src = count_csv_types(src_fold, apply_map=False)
        cnt_dst = count_csv_types(dst_fold, apply_map=False)
        if cnt_src:
            print("Original (TCGA):")
            for k in sorted(cnt_src.keys()):
                print(f"  {TCGA_NUCLEI_NAMES.get(k, str(k))}: {cnt_src[k]}")
        if cnt_dst:
            print("Mapped (PanNuke):")
            for k in sorted(cnt_dst.keys()):
                print(f"  {PANNUKE_NUCLEI_NAMES.get(k, str(k))}: {cnt_dst[k]}")

    print("\n" + "=" * 60)
    print("Tissue ID counts (from types.csv)")
    print("=" * 60)
    for fold in [0, 1, 2]:
        src_fold = src_root / f"fold{fold}"
        dst_fold = dst_root / f"fold{fold}"
        types_src = src_fold / "types.csv"
        types_dst = dst_fold / "types.csv"
        if types_src.is_file():
            cnt = count_tissues(types_src)
            print(f"\nFold {fold} original: {cnt}")
        if types_dst.is_file():
            cnt = count_tissues(types_dst)
            print(f"Fold {fold} mapped:   {cnt} (Lung -> PanNuke ID 10)")

    if not args.no_viz:
        print("\n" + "=" * 60)
        print("Sample visualizations")
        print("=" * 60)
        visualize_samples(src_root, dst_root, n_samples=5)

    print("\nVerification complete.")


if __name__ == "__main__":
    main()
