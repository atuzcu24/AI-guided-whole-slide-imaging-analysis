#!/usr/bin/env python3
"""
CellViT / CellViT+ / FiLM inference on a folder of 256x256 PNG images.

Example (for HuggingFace):
python cellvit_inference_folder.py \
    --input_dir /projectnb/ec500kb/projects/Fall_2025_Projects/Project_2/AI-guided-whole-slide-imaging-analysis/AI-GUIDED-CLEAN/ProcessedDataset/v1_40x_area20_2/patches_cellvit_p256_pannuke/fold0/images \
    --output_dir ./results \
    --hf_repo BerkTuzcuBU/CellViT-FiLM-256-Run \
    --checkpoint model_best.pth \
    --magnification 40

Example (for downloaded folder):
python cellvit_rosie_film_inference.py \
  --input_dir /projectnb/ec500kb/projects/Fall_2025_Projects/Project_2/AI-guided-whole-slide-imaging-analysis/AI-GUIDED-CLEAN/ProcessedDataset/v1_40x_area20_2/patches_cellvit_p256_pannuke/fold1/images \
  --output_dir ./cellvit_inference_results_film256 \
  --run_dir /projectnb/ec500kb/projects/Fall_2025_Projects/Project_2/AI-guided-whole-slide-imaging-analysis/AI-GUIDED-CLEAN/ProcessedDataset/v1_40x_area20_2/patches_cellvit_p256_pannuke/logs_local_berk/2025-12-05T191217_film256 \
  --checkpoint model_best.pth \
  --magnification 40
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# -------------------------------------------------
# CellViT imports
# -------------------------------------------------
sys.path.append("../CellViT-plus-plus")

from huggingface_hub import snapshot_download
from cellvit.training.evaluate.inference_cellvit_experiment_pannuke import InferenceCellViT
from cellvit.training.trainer.trainer_cellvit import CellViTTrainer


# -------------------------------------------------
# Utils
# -------------------------------------------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def save_overlay(orig, overlay, path):
    plt.figure(figsize=(6, 6))
    plt.imshow(overlay)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# -------------------------------------------------
# Main
# -------------------------------------------------
def main(args):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    pred_dir   = output_dir / "predictions"
    vis_dir    = output_dir / "visuals"
    metric_dir = output_dir / "metrics"

    for d in [pred_dir, vis_dir, metric_dir]:
        ensure_dir(d)

    # -------------------------------------------------
    # Load run directory (HF or local)
    # -------------------------------------------------
    if args.hf_repo:
        print(f"[INFO] Downloading HF repo: {args.hf_repo}")
        run_dir = snapshot_download(
            repo_id=args.hf_repo,
            local_dir=output_dir / "run",
            local_dir_use_symlinks=False,
        )
    else:
        run_dir = args.run_dir

    # -------------------------------------------------
    # Build inference helper
    # -------------------------------------------------
    inference = InferenceCellViT(
        run_dir=run_dir,
        gpu=0 if device == "cuda" else None,
        checkpoint_name=args.checkpoint,
        magnification=args.magnification,
    )

    model, val_loader, dataset_config = inference.setup_patch_inference()
    model = model.to(device).eval()

    print("[INFO] Model loaded")
    print("[INFO] Nuclei types:", dataset_config["nuclei_types"])

    # -------------------------------------------------
    # Trainer (for postprocessing only)
    # -------------------------------------------------
    dummy_opt = torch.optim.SGD(model.parameters(), lr=0.1)
    dummy_sched = torch.optim.lr_scheduler.StepLR(dummy_opt, 1000)

    trainer = CellViTTrainer(
        model=model,
        loss_fn_dict={},
        optimizer=dummy_opt,
        scheduler=dummy_sched,
        device=device,
        logger=inference.logger,
        logdir=run_dir,
        num_classes=len(dataset_config["nuclei_types"]),
        dataset_config=dataset_config,
        experiment_config={},
        early_stopping=None,
        log_images=False,
        magnification=args.magnification,
        mixed_precision=False,
    )

    # -------------------------------------------------
    # Get EXACT validation transforms
    # -------------------------------------------------
    ds = val_loader.dataset
    tfm = getattr(ds, "transforms", None) or getattr(ds, "transform", None)
    if tfm is None:
        raise RuntimeError("Dataset has no transform")

    # -------------------------------------------------
    # Inference loop
    # -------------------------------------------------
    image_paths = sorted(input_dir.glob("*.png"))
    print(f"[INFO] Found {len(image_paths)} images")

    summary = {
        "num_images": len(image_paths),
        "checkpoint": args.checkpoint,
        "magnification": args.magnification,
    }

    for img_path in tqdm(image_paths, desc="Running inference"):
        name = img_path.stem

        # Load image
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)

        # Apply SAME transforms
        try:
            out = tfm(image=img_np)
            img_t = out["image"]
        except TypeError:
            img_t = tfm(img)

        if isinstance(img_t, np.ndarray):
            img_t = torch.from_numpy(img_t).permute(2, 0, 1)
        elif img_t.shape[-1] == 3:
            img_t = img_t.permute(2, 0, 1)

        img_batch = img_t.unsqueeze(0).to(device)

        # Forward
        with torch.no_grad():
            raw_out = model(img_batch)

        processed = trainer.unpack_predictions(raw_out)

        binary_map = processed.nuclei_binary_map[0, 1].cpu().numpy()
        hv_map     = processed.hv_map[0].cpu().numpy()
        inst_map   = processed.instance_map[0].cpu().numpy().astype(np.int32)
        type_map   = torch.argmax(processed.nuclei_type_map, dim=1)[0].cpu().numpy()

        # Save raw predictions
        np.savez_compressed(
            pred_dir / f"{name}_preds.npz",
            binary_map=binary_map,
            hv_map=hv_map,
            instance_map=inst_map,
            type_map=type_map,
        )

        # ---------------- Visualizations ----------------
        H, W = binary_map.shape
        orig = cv2.resize(img_np, (W, H))

        # Binary overlay
        bin_color = cv2.applyColorMap((binary_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        bin_color = cv2.cvtColor(bin_color, cv2.COLOR_BGR2RGB)
        overlay_binary = cv2.addWeighted(orig, 0.65, bin_color, 0.35, 0)

        # Instance overlay
        inst_color = np.zeros_like(orig)
        for uid in np.unique(inst_map):
            if uid == 0:
                continue
            inst_color[inst_map == uid] = np.random.randint(0, 255, size=3)

        overlay_inst = cv2.addWeighted(orig, 0.55, inst_color, 0.45, 0)

        save_overlay(orig, overlay_binary, vis_dir / f"{name}_binary.png")
        save_overlay(orig, overlay_inst,   vis_dir / f"{name}_instance.png")

    # -------------------------------------------------
    # Save summary
    # -------------------------------------------------
    with open(metric_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("[DONE] Inference complete")
    print(f"[DONE] Results saved to: {output_dir}")


# -------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--hf_repo", default=None, help="HuggingFace repo id")
    parser.add_argument("--run_dir", default=None, help="Local run dir (if not HF)")
    parser.add_argument("--checkpoint", default="model_best.pth")
    parser.add_argument("--magnification", type=int, default=40)

    args = parser.parse_args()
    main(args)
