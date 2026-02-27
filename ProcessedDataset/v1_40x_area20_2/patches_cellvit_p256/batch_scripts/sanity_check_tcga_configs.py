#!/usr/bin/env python3
"""
Sanity-check TCGA idinit configs: print backbone, method, condition_source, film_target,
film_init/idinit, lr, seed, dataset from each YAML.

Usage:
  python sanity_check_tcga_configs.py
  python sanity_check_tcga_configs.py path/to/config.yaml
"""
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("pip install pyyaml")
    sys.exit(1)

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIGS_DIR = SCRIPT_DIR.parent / "train_configs"

DEFAULT_CONFIGS = [
    "virchow_baseline_tcga_idinit_lr3e5.yaml",
    "samh_baseline_tcga_idinit_lr3e5.yaml",
    "virchow_rosie_film_z4_idinit_lr3e5_tcga.yaml",
    "virchow_rosie_film_z3z4_idinit_lr3e5_tcga.yaml",
    "virchow_rosie_film_z1z4_idinit_lr3e5_tcga.yaml",
    "samh_rosie_film_z4_idinit_lr3e5_tcga.yaml",
    "samh_rosie_film_z3z4_idinit_lr3e5_tcga.yaml",
    "samh_rosie_film_z1z4_idinit_lr3e5_tcga.yaml",
]


def extract_config_info(cfg: dict) -> dict:
    """Extract fields for table parsing / sanity check."""
    model = cfg.get("model", {})
    fusion = cfg.get("fusion", {})
    training = cfg.get("training", {})
    data = cfg.get("data", {})
    logging_cfg = cfg.get("logging", {})

    backbone = model.get("backbone", "?")
    if "virchow" in str(backbone).lower() and "sam" not in str(backbone).lower():
        backbone_display = "Virchow"
    elif "sam-h" in str(backbone).lower() or "samh" in str(backbone).lower():
        backbone_display = "SAMH"
    else:
        backbone_display = str(backbone)

    film_layers = fusion.get("film_layers") or []
    if not film_layers:
        method = "Baseline"
        film_target = None
        condition_source = "Rosie" if "rosie" in str(backbone).lower() else "None"
    else:
        method = "FiLM"
        condition_source = "Rosie"
        if len(film_layers) == 1:
            film_target = film_layers[0]
        elif set(film_layers) == {"z3", "z4"}:
            film_target = "z3z4"
        elif set(film_layers) == {"z1", "z2", "z3", "z4"}:
            film_target = "z1z4"
        else:
            film_target = "+".join(film_layers)

    film_init = fusion.get("film_init", "?")
    idinit = "true" if film_init == "identity" else "false"
    opt = training.get("optimizer_hyperparameter", {})
    lr = opt.get("lr", "?")
    seed = cfg.get("random_seed", "?")
    log_comment = logging_cfg.get("log_comment", "")
    dataset = "TCGA" if "TCGA" in log_comment or "tcga" in str(data.get("dataset_path", "")).lower() or "patches_cellvit_p256" in str(data.get("dataset_path", "")) else "?"
    batch_size = training.get("batch_size", "?")
    project = logging_cfg.get("project", "?")

    return {
        "backbone": backbone_display,
        "method": method,
        "condition_source": condition_source,
        "film_target": film_target,
        "film_init": film_init,
        "idinit": idinit,
        "lr": lr,
        "seed": seed,
        "dataset": dataset,
        "batch_size": batch_size,
        "project": project,
        "log_comment": log_comment,
    }


def main():
    if len(sys.argv) > 1:
        config_paths = [Path(p) for p in sys.argv[1:]]
    else:
        config_paths = [CONFIGS_DIR / name for name in DEFAULT_CONFIGS]

    print(f"{'Config':<45} {'backbone':<8} {'method':<10} {'cond':<6} {'film':<6} {'idinit':<6} {'lr':<10} {'batch':<6} {'proj':<25}")
    print("-" * 135)

    for path in config_paths:
        if not path.exists():
            print(f"SKIP (not found): {path}")
            continue
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        info = extract_config_info(cfg)
        name = path.name[:43]
        film_tgt = str(info["film_target"]) if info["film_target"] else "-"
        print(f"{name:<45} {info['backbone']:<8} {info['method']:<10} {info['condition_source']:<6} {film_tgt:<6} {info['idinit']:<6} {str(info['lr']):<10} {str(info['batch_size']):<6} {str(info['project']):<25}")

    print("\nAll configs: backbone (Virchow/SAMH), method (Baseline/FiLM), batch_size=16, project=SAMH-cell-segmentation, dataset_path=patches_cellvit_p256, train/val/test folds [0]/[1]/[2].")


if __name__ == "__main__":
    main()
