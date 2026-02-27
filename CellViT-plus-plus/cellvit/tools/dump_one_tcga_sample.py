#!/usr/bin/env python3
"""
Diagnostic script for a single TCGA→PanNuke mapped sample.

Loads image + GT labels using the SAME dataset/loader as the evaluation pipeline,
then prints stats and optionally saves visualizations. Use for detecting
channel-order/ID mismatches.

Usage:
  python -m cellvit.tools.dump_one_tcga_sample \\
    --dataset_root /path/to/patches_cellvit_p256_tcga_mapped_to_pannuke \\
    --filename TCGA-5P-A9K0-01Z-00-DX1_1_0_0.png \\
    [--split fold0] [--save_dir ./debug_out]

Run from CellViT-plus-plus directory (or ensure it is in PYTHONPATH):
  cd AI-GUIDED-CLEAN/CellViT-plus-plus
  python -m cellvit.tools.dump_one_tcga_sample ...
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import yaml

# Matplotlib Agg backend for headless (SCC-safe)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root for imports (parent of cellvit = CellViT-plus-plus)
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from cellvit.training.datasets.pannuke import PanNukeDataset


# PanNuke nuclei type ID -> name (for visualization)
NUCLEI_ID_TO_NAME = {
    0: "Background",
    1: "Neoplastic",
    2: "Inflammatory",
    3: "Connective",
    4: "Dead",
    5: "Epithelial",
}
# Fixed colors for type_map viz (R, G, B) per nuclei ID
NUCLEI_COLORS = {
    0: (0.9, 0.9, 0.9),
    1: (1.0, 0.0, 0.0),
    2: (0.0, 1.0, 0.0),
    3: (0.0, 0.0, 1.0),
    4: (0.5, 0.0, 0.5),
    5: (1.0, 1.0, 0.0),
}


def _parse_split(split_arg: str) -> int:
    """Parse --split to fold number. Accepts 'fold0','fold1','fold2' or '0','1','2'."""
    s = split_arg.strip().lower()
    if s.startswith("fold"):
        s = s[4:]
    try:
        n = int(s)
        if n not in (0, 1, 2):
            raise ValueError(f"Expected fold 0, 1, or 2; got {n}")
        return n
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"Invalid --split '{split_arg}'. Use fold0, fold1, fold2, or 0, 1, 2."
        ) from e


def _normalize_filename(name: str) -> str:
    """Ensure filename has .png extension for matching img_names."""
    if not name.lower().endswith(".png"):
        return name + ".png"
    return name


def _find_sample_index(dataset: PanNukeDataset, filename: str) -> int:
    """Find dataset index for given image filename. Raises if not found."""
    norm = _normalize_filename(filename)
    for i, img_name in enumerate(dataset.img_names):
        if img_name == norm:
            return i
    available = dataset.img_names[:5]
    raise FileNotFoundError(
        f"Filename '{filename}' (normalized: '{norm}') not found in dataset. "
        f"Dataset has {len(dataset.img_names)} samples. "
        f"First few: {available}..."
    )


def _load_raw_sample(dataset: PanNukeDataset, index: int):
    """Load image and mask without transforms (raw from disk)."""
    img = dataset.load_imgfile(index)
    mask = dataset.load_maskfile(index)
    return img, mask


def _load_npy_direct(mask_path: Path):
    """Load NPY and return full dict to check for tissue_map."""
    data = np.load(mask_path, allow_pickle=True)
    return data.item()


def _print_image_stats(img: np.ndarray, filename: str) -> None:
    print("\n--- Image ---")
    print(f"  shape: {img.shape}, dtype: {img.dtype}")
    print(f"  min: {np.min(img):.1f}, max: {np.max(img):.1f}")


def _compute_inst_id_gaps(inst_map: np.ndarray) -> Tuple[int, bool, list]:
    """Return (n_gaps, is_contiguous, nonzero_ids_list)."""
    nz = np.unique(inst_map[inst_map > 0])
    nz_list = nz.tolist()
    if len(nz) == 0:
        return 0, True, []
    max_id = int(np.max(inst_map))
    # Gaps = (max_id - 1) - len(nz)  # if we expect 1..max, how many slots are empty
    expected_count = max_id  # ids 1..max_id
    actual_count = len(nz)
    n_gaps = expected_count - actual_count
    contiguous = n_gaps == 0 and np.array_equal(nz, np.arange(1, len(nz) + 1))
    return n_gaps, contiguous, nz_list


def _print_inst_map_stats(inst_map: np.ndarray) -> None:
    print("\n--- inst_map (instance map) ---")
    print(f"  shape: {inst_map.shape}, dtype: {inst_map.dtype}")
    uniq = np.unique(inst_map)
    nz = uniq[uniq > 0]
    n_unique_ids = len(nz)
    n_instance_px = int(np.sum(inst_map > 0) if inst_map.size > 0 else 0)
    n_gaps, contiguous, nz_list = _compute_inst_id_gaps(inst_map)
    print(f"  unique ID count: {len(uniq)} (including 0)")
    print(f"  min: {int(np.min(inst_map))}, max: {int(np.max(inst_map))}")
    print(f"  number of instances (unique IDs > 0): {n_unique_ids}")
    print(f"  instance pixels (>0): {n_instance_px}")
    print(f"  inst_map_nonzero_ids: {nz_list}")
    print(f"  inst_id_gaps: {n_gaps} (missing IDs if relabel to 1..max)")
    if not contiguous:
        print(f"  IDs contiguous (1..N, excluding 0): False")
        print(f"  suggested_fix: PQ implementations that iterate id=1..max may break; "
              "prefer np.unique(inst_map) or relabel per patch.")
    else:
        print(f"  IDs contiguous (1..N, excluding 0): True")


def _print_type_map_stats(type_map: np.ndarray) -> None:
    print("\n--- type_map (nuclei type map) ---")
    print(f"  shape: {type_map.shape}, dtype: {type_map.dtype}")
    uniq, counts = np.unique(type_map, return_counts=True)
    order = np.argsort(-counts)
    print(f"  unique values + pixel counts (top 20):")
    for i in range(min(20, len(order))):
        idx = order[i]
        uid = int(uniq[idx])
        cnt = int(counts[idx])
        name = NUCLEI_ID_TO_NAME.get(uid, f"unknown({uid})")
        print(f"    {uid} ({name}): {cnt} px")
    if 5 not in uniq:
        print("  [WARN] Epithelial (5) is missing in type_map!")


def _print_alignment_stats(inst_map: np.ndarray, type_map: np.ndarray) -> None:
    inst_positive = inst_map > 0
    type_positive = type_map > 0
    both = np.sum(inst_positive & type_positive)
    inst_only = np.sum(inst_positive & ~type_positive)
    type_only = np.sum(~inst_positive & type_positive)
    total = inst_map.size
    print("\n--- Alignment (inst_map vs type_map) ---")
    print(f"  pixels where (inst>0) and (type>0): {both} ({100*both/total:.2f}%)")
    if inst_only > 0:
        print(f"  [WARN] (inst>0) but (type==0): {inst_only} ({100*inst_only/total:.2f}%)")
    if type_only > 0:
        print(f"  [WARN] (type>0) but (inst==0): {type_only} ({100*type_only/total:.2f}%)")


def _print_tissue_info(tissue_str: str, dataset_config: dict, npy_obj: dict, mask_path: Path) -> None:
    print("\n--- Tissue ---")
    print(f"  scalar tissue (from types.csv): '{tissue_str}'")
    tissue_types = dataset_config.get("tissue_types") or {}
    tid = tissue_types.get(tissue_str)
    if tid is not None:
        print(f"  tissue ID (from dataset_config): {tid}")
    else:
        print(f"  [WARN] tissue '{tissue_str}' not found in dataset_config tissue_types")
    if "tissue_map" in npy_obj:
        tm = npy_obj["tissue_map"]
        uniq, cnts = np.unique(tm, return_counts=True)
        print("  tissue_map present:")
        for u, c in zip(uniq, cnts):
            print(f"    {u}: {c} px")
    else:
        print("  tissue_map: not present in NPY (expected for TCGA mapped dataset)")


def _run_scan_ids(dataset: PanNukeDataset, k: int = 200) -> None:
    """Sample K patches and report inst ID diagnostics. Diagnostic only."""
    n_total = len(dataset)
    if n_total == 0:
        print("\n--- scan_ids: no patches in split ---")
        return
    n_sample = min(k, n_total)
    rng = np.random.default_rng(42)
    indices = rng.choice(n_total, size=n_sample, replace=False)
    non_contiguous_count = 0
    max_id_list = []
    gap_records = []  # (filename, n_gaps, max_id)
    for i in indices:
        _, mask = _load_raw_sample(dataset, i)
        inst_map = mask[:, :, 0]
        n_gaps, contiguous, _ = _compute_inst_id_gaps(inst_map)
        if not contiguous:
            non_contiguous_count += 1
        max_id = int(np.max(inst_map)) if np.any(inst_map > 0) else 0
        max_id_list.append(max_id)
        if n_gaps > 0:
            gap_records.append((dataset.img_names[i], n_gaps, max_id))
    pct_non_contig = 100.0 * non_contiguous_count / n_sample
    print("\n" + "=" * 60)
    print("--- scan_ids: inst_map ID diagnostics (sampled %d of %d patches) ---" % (n_sample, n_total))
    print("=" * 60)
    print("  %% of patches with non-contiguous inst IDs: %.1f%%" % pct_non_contig)
    max_id_arr = np.array(max_id_list)
    print("  histogram of max_inst_id:")
    max_val = int(np.max(max_id_arr))
    if max_val == 0:
        hist, _ = np.histogram(max_id_arr, bins=[0, 1])
        print("    [0, 1): %d" % hist[0])
    else:
        bin_edges = [0, 1, 10, 50, 100, 200, 500, 1000]
        bin_edges = [b for b in bin_edges if b <= max_val] + [max_val + 1]
        bin_edges = sorted(set(bin_edges))
        hist, bedges = np.histogram(max_id_arr, bins=bin_edges)
        for j in range(len(hist)):
            if hist[j] > 0:
                print("    [%d, %d): %d" % (bedges[j], bedges[j + 1], hist[j]))
    gap_records.sort(key=lambda x: -x[1])
    if gap_records:
        print("  example filenames with worst gaps (top 5):")
        for fn, ng, mid in gap_records[:5]:
            print("    %s: gaps=%d max_id=%d" % (fn, ng, mid))
    else:
        print("  all sampled patches have contiguous IDs.")


def _visualize(
    img: np.ndarray,
    inst_map: np.ndarray,
    type_map: np.ndarray,
    filename: str,
    stats_summary: str,
    save_path: Path,
) -> None:
    """Save 4-panel PNG: RGB, inst_map (random colors), type_map (fixed colors), binary mask overlay."""
    np.random.seed(42)
    n_inst = int(np.max(inst_map))
    if n_inst > 0:
        cmap_inst = np.zeros((n_inst + 1, 3))
        cmap_inst[0] = (0, 0, 0)
        cmap_inst[1:] = np.random.rand(n_inst, 3)
    else:
        cmap_inst = np.array([[0, 0, 0]])

    type_rgb = np.zeros((*type_map.shape, 3))
    for uid, col in NUCLEI_COLORS.items():
        type_rgb[type_map == uid] = col
    for uid in np.unique(type_map):
        if int(uid) not in NUCLEI_COLORS:
            type_rgb[type_map == uid] = (0.5, 0.5, 0.5)

    inst_rgb = np.zeros((*inst_map.shape, 3))
    for i in range(1, n_inst + 1):
        inst_rgb[inst_map == i] = cmap_inst[i]

    bin_mask = (inst_map > 0).astype(np.float32)
    overlay = img.copy().astype(np.float32) / 255.0
    overlay[:, :, 0] = np.clip(overlay[:, :, 0] * (1 - bin_mask * 0.5) + bin_mask * 0.5, 0, 1)
    overlay[:, :, 1] = np.clip(overlay[:, :, 1] * (1 - bin_mask * 0.5), 0, 1)
    overlay[:, :, 2] = np.clip(overlay[:, :, 2] * (1 - bin_mask * 0.5), 0, 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes[0, 0].imshow(img)
    axes[0, 0].set_title("(a) RGB image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(inst_rgb)
    axes[0, 1].set_title("(b) inst_map (random colors)")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(type_rgb)
    axes[1, 0].set_title("(c) type_map (0=gray, 2=green, 5=yellow)")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title("(d) binary mask overlay (inst>0)")
    axes[1, 1].axis("off")

    fig.suptitle(f"{filename}\n{stats_summary}", fontsize=10, wrap=True)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved visualization to: {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dump one TCGA→PanNuke mapped sample for diagnosis (channel/ID checks)."
    )
    parser.add_argument("--dataset_root", type=Path, required=True,
                        help="Path to patches_cellvit_p256_tcga_mapped_to_pannuke root")
    parser.add_argument("--split", type=str, default="fold1",
                        help="Split/fold: fold0, fold1, fold2 (default: fold1)")
    parser.add_argument("--filename", type=str, default=None,
                        help="Exact image filename (e.g. TCGA-5P-A9K0-01Z-00-DX1_1_0_0.png); required unless --scan_ids")
    parser.add_argument("--save_dir", type=Path, default=None,
                        help="If set, save PNG visualization here")
    parser.add_argument("--scan_ids", action="store_true",
                        help="Sample K=200 patches and report pct non-contiguous inst IDs, histogram, worst-gap examples")
    args = parser.parse_args()

    if not args.filename and not args.scan_ids:
        parser.error("At least one of --filename or --scan_ids is required.")

    dataset_root = args.dataset_root.resolve()
    if not dataset_root.is_dir():
        print(f"ERROR: dataset_root does not exist: {dataset_root}", file=sys.stderr)
        sys.exit(1)

    fold = _parse_split(args.split)

    # Load dataset_config.yaml (required)
    config_path = dataset_root / "dataset_config.yaml"
    if not config_path.exists():
        print(f"ERROR: dataset_config.yaml not found at {config_path}", file=sys.stderr)
        sys.exit(1)
    with open(config_path) as f:
        dataset_config = yaml.safe_load(f)

    # Reuse exact dataset class from evaluation
    dataset = PanNukeDataset(
        dataset_path=dataset_root,
        folds=[fold],
        transforms=None,
        stardist=False,
        regression=False,
    )

    if args.scan_ids:
        _run_scan_ids(dataset, k=200)

    if args.filename:
        idx = _find_sample_index(dataset, args.filename)
        img, mask = _load_raw_sample(dataset, idx)
        inst_map = mask[:, :, 0]
        type_map = mask[:, :, 1]

        npy_obj = _load_npy_direct(dataset.masks[idx])
        if "inst_map" not in npy_obj:
            raise KeyError(
                f"Expected 'inst_map' in NPY {dataset.masks[idx]}. "
                "This dataset format is incompatible."
            )
        if "type_map" not in npy_obj:
            raise KeyError(
                f"Expected 'type_map' in NPY {dataset.masks[idx]}. "
                "This dataset format is incompatible."
            )
        tissue_str = dataset.types.get(dataset.img_names[idx])
        if tissue_str is None:
            raise KeyError(
                f"Image '{dataset.img_names[idx]}' not found in types.csv (fold{fold}). "
                "Each image must have a tissue type entry."
            )

        print("=" * 60)
        print(f"TCGA→PanNuke dump: {dataset.img_names[idx]}")
        print(f"  dataset_root: {dataset_root}")
        print(f"  split: fold{fold}")
        print("=" * 60)

        _print_image_stats(img, dataset.img_names[idx])
        _print_inst_map_stats(inst_map)
        _print_type_map_stats(type_map)
        _print_alignment_stats(inst_map, type_map)
        _print_tissue_info(tissue_str, dataset_config, npy_obj, dataset.masks[idx])

        # Stats summary for visualization title
        n_inst = int(np.sum(inst_map > 0))
        n_unique_inst = len(np.unique(inst_map[inst_map > 0])) if n_inst > 0 else 0
        unique_types = np.unique(type_map)
        stats_summary = f"inst_px={n_inst} unique_inst={n_unique_inst} types={list(unique_types)}"

        if args.save_dir is not None:
            stem = Path(args.filename).stem
            save_path = args.save_dir / f"dump_{stem}.png"
            _visualize(img, inst_map, type_map, dataset.img_names[idx], stats_summary, save_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
