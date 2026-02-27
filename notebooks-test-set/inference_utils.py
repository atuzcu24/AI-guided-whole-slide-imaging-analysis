"""
Utility functions for CellViT inference and visualization.
Extracted from inference_notebook_multimodal_pannuke8020.ipynb
"""

from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image

# Ground-truth label semantics (PanNuke)
GT_ID_TO_NAME = {
    0: "background",
    1: "ambiguous",
    2: "epithelial",
    3: "lymphocyte",
    4: "macrophage",
    5: "neutrophil",
}


def remap_type_map(type_map, gt_id_to_name, gt_to_model_name, model_name_to_id):
    """
    Remap type_map from ground-truth ID space to model ID space.
    """
    remapped = np.zeros_like(type_map, dtype=np.int32)
    for gt_id, gt_name in gt_id_to_name.items():
        model_name = gt_to_model_name.get(gt_name, "other")
        remapped[type_map == gt_id] = model_name_to_id[model_name]
    return remapped


def categorical_colormap(n_classes: int):
    """Build RGB colormap for cell types."""
    import matplotlib
    cmap = matplotlib.colormaps.get_cmap(
        "Set2" if n_classes <= 8 else "tab20"
    )
    return cmap(np.linspace(0, 1, n_classes + 1))[:, :3]


def build_type_overlay(rgb, type_map, lut, alpha=0.45):
    """Overlay type map colors on RGB image."""
    rgb = rgb.astype(np.float32) / 255.0
    mask = type_map > 0

    tm = np.clip(type_map, 0, lut.shape[0] - 1).astype(int)
    color_mask = lut[tm]
    color_mask[~mask] = 0.0
    color_mask = np.power(color_mask, 0.7)

    overlay = rgb.copy()
    overlay[mask] = (1 - alpha) * overlay[mask] + alpha * color_mask[mask]
    return overlay


def sanitize_backbone_for_sam(backbone: str) -> str:
    """
    Converts things like 'sam-h-rosie-film' -> 'sam-h'
    Leaves 'sam-h' unchanged.
    """
    if backbone is None:
        return "sam-h"
    s = str(backbone).strip().lower().replace("_", "-")
    if s.startswith("sam-"):
        parts = s.split("-")
        if len(parts) >= 2:
            return "-".join(parts[:2])  # sam-h / sam-l / sam-b
    return s


def build_lut_from_dataset_config(dataset_config):
    """
    Build LUT, id_to_name, and max_cls_id from dataset_config.
    Returns (lut, id_to_name, max_cls_id).
    """
    nuclei_types = dataset_config["nuclei_types"]
    id_to_name = {v: k for k, v in nuclei_types.items()}
    max_cls_id = max(id_to_name.keys())
    lut = categorical_colormap(max_cls_id)
    return lut, id_to_name, max_cls_id


def load_cellvit_inference(
    run_dir,
    checkpoint_name="model_best.pth",
    magnification=40,
    gpu=0,
    device=None,
):
    """
    Load CellViT model and full inference setup (model + test dataloader).
    Triggers "Performing Inference on test set" log. Use load_cellvit_inference_single_patch
    for interactive viewers that only need to infer one patch at a time.
    """
    from cellvit.training.evaluate.inference_cellvit_experiment_pannuke import InferenceCellViT

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    inference = InferenceCellViT(
        run_dir=run_dir,
        gpu=gpu,
        checkpoint_name=checkpoint_name,
        magnification=magnification,
    )

    bb_raw = inference.run_conf.get("model", {}).get("backbone", "sam-h")
    inference.run_conf["model"]["backbone"] = sanitize_backbone_for_sam(bb_raw)

    model, val_loader, dataset_config = inference.setup_patch_inference()
    model = model.to(device).eval()
    return model, val_loader, dataset_config, inference


def load_cellvit_inference_single_patch(
    run_dir,
    checkpoint_name="model_best.pth",
    magnification=40,
    gpu=0,
    device=None,
):
    """
    Load CellViT model for single-patch inference only (no full test set).
    Does NOT trigger "Performing Inference on test set". Use for interactive viewers.

    Returns:
        tuple: (model, transforms, dataset_config, inference)
        Pass transforms to run_cellvit_on_patch(..., transforms=transforms).
    """
    from cellvit.training.evaluate.inference_cellvit_experiment_pannuke import InferenceCellViT

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    inference = InferenceCellViT(
        run_dir=run_dir,
        gpu=gpu,
        checkpoint_name=checkpoint_name,
        magnification=magnification,
    )

    bb_raw = inference.run_conf.get("model", {}).get("backbone", "sam-h")
    inference.run_conf["model"]["backbone"] = sanitize_backbone_for_sam(bb_raw)

    model, transforms, dataset_config = inference.setup_model_only()
    model = model.to(device).eval()
    return model, transforms, dataset_config, inference


def build_postprocess_trainer(
    model,
    inference,
    dataset_config,
    run_dir,
    device=None,
    magnification=40,
):
    """
    Create a lightweight CellViTTrainer ONLY for unpack_predictions.
    No training, no logging, no side effects.
    """
    from cellvit.training.trainer.trainer_cellvit import CellViTTrainer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dummy_opt = torch.optim.SGD(model.parameters(), lr=0.1)
    dummy_sched = torch.optim.lr_scheduler.StepLR(
        dummy_opt, step_size=1000, gamma=1.0
    )

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
        magnification=magnification,
        mixed_precision=False,
    )
    return trainer


def run_cellvit_on_patch(img_np, model, val_loader, trainer, device=None, transforms=None):
    """
    Run CellViT on a single patch and postprocess outputs.

    Args:
        img_np: RGB image as numpy array (H, W, 3).
        model: Loaded CellViT model.
        val_loader: DataLoader (used for transforms when transforms is None). Can be None if transforms given.
        trainer: CellViTTrainer for unpack_predictions.
        device: Device to run on.
        transforms: Optional. Use when using load_cellvit_inference_single_patch (avoids full test set).
    """
    if device is None:
        device = next(model.parameters()).device

    if transforms is not None:
        tfm = transforms
    elif val_loader is not None:
        ds = val_loader.dataset
        tfm = getattr(ds, "transforms", None) or getattr(ds, "transform", None)
    else:
        tfm = None
    if tfm is None:
        raise RuntimeError("Dataset has no transforms; pass transforms= or val_loader with dataset.transforms")

    try:
        img_t = tfm(image=img_np)["image"]
    except TypeError:
        img_t = tfm(Image.fromarray(img_np))

    if isinstance(img_t, np.ndarray):
        img_t = torch.from_numpy(img_t)
    if img_t.ndim == 3 and img_t.shape[-1] == 3:
        img_t = img_t.permute(2, 0, 1)

    img_t = img_t.unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(img_t)

    processed = trainer.unpack_predictions(out)

    return {
        "binary_map": processed.nuclei_binary_map[0, 1].cpu().numpy(),
        "inst_map": processed.instance_map[0].cpu().numpy().astype(int),
        "type_map": torch.argmax(processed.nuclei_type_map, dim=1)[0].cpu().numpy(),
    }


def infer_single(img_np, model, transforms, trainer, device=None):
    """
    Run inference on a single patch using model-only setup (no test dataloader).
    Use with load_cellvit_inference_single_patch.
    """
    return run_cellvit_on_patch(
        img_np, model, None, trainer, device=device, transforms=transforms
    )


def visualize_type_overlay_matrix(
    gt_type_map,
    pred_maps,
    rgb,
    lut,
    id_to_name,
    max_cls_id,
    title="GT vs Models",
):
    """
    Comparison matrix:
      Columns: GT + models
      Rows:    Type Map / Overlay
    """
    model_names = list(pred_maps.keys())
    col_names = ["GT"] + model_names
    n_cols = len(col_names)
    n_rows = 2

    overlay_gt = build_type_overlay(rgb, gt_type_map, lut)
    overlay_preds = {
        name: build_type_overlay(rgb, pred_maps[name], lut)
        for name in model_names
    }

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.2 * n_cols, 4.2 * n_rows),
        constrained_layout=True,
    )

    if n_cols == 1:
        axes = axes.reshape(n_rows, 1)

    for col, name in enumerate(col_names):
        ax = axes[0, col]
        if name == "GT":
            ax.imshow(gt_type_map, cmap="Set2", vmin=0, vmax=max_cls_id)
        else:
            ax.imshow(pred_maps[name], cmap="Set2", vmin=0, vmax=max_cls_id)
        ax.axis("off")

        ax = axes[1, col]
        if name == "GT":
            ax.imshow(overlay_gt)
        else:
            ax.imshow(overlay_preds[name])
        ax.axis("off")

    for col, name in enumerate(col_names):
        axes[0, col].set_title(name, fontsize=13, pad=8)

    axes[0, 0].set_ylabel("Type Map", fontsize=13, rotation=90, labelpad=12)
    axes[1, 0].set_ylabel("Overlay", fontsize=13, rotation=90, labelpad=12)

    fig.suptitle(title, fontsize=16, y=1.02)
    plt.show()

    present = set(np.unique(gt_type_map))
    for pm in pred_maps.values():
        present |= set(np.unique(pm))
    present = sorted(i for i in present if i > 0)

    legend_patches = [
        Patch(color=lut[i], label=f"{i}: {id_to_name[i]}")
        for i in present
    ]

    plt.figure(figsize=(6, 0.6 + 0.3 * len(present)))
    plt.legend(
        handles=legend_patches,
        title="Cell Types",
        frameon=False,
        loc="center left",
    )
    plt.axis("off")
    plt.show()


def visualize_type_overlay_error_matrix(
    gt_type_map,
    pred_maps,
    rgb,
    lut,
    id_to_name,
    max_cls_id,
    title="GT vs Models (Type / Overlay / Error)",
    save_path=None,
):
    """
    Comparison matrix:
      Columns: GT | Model A | Model B | ...
      Rows:    Type Map | Overlay | Error Map

    If save_path is provided, saves the main figure to disk and closes it
    without displaying (useful for batch saving to avoid memory issues).
    """
    model_names = list(pred_maps.keys())
    col_names = ["GT"] + model_names
    n_cols = len(col_names)
    n_rows = 3

    overlay_gt = build_type_overlay(rgb, gt_type_map, lut)
    overlay_preds = {
        name: build_type_overlay(rgb, pred_maps[name], lut)
        for name in model_names
    }

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.4 * n_cols, 4.4 * n_rows),
        constrained_layout=True,
    )

    for col, name in enumerate(col_names):
        ax = axes[0, col]
        if name == "GT":
            ax.imshow(gt_type_map, cmap="Set2", vmin=0, vmax=max_cls_id)
        else:
            ax.imshow(pred_maps[name], cmap="Set2", vmin=0, vmax=max_cls_id)
        ax.axis("off")

        ax = axes[1, col]
        if name == "GT":
            ax.imshow(overlay_gt)
        else:
            ax.imshow(overlay_preds[name])
        ax.axis("off")

        ax = axes[2, col]
        if name == "GT":
            ax.imshow(np.zeros_like(gt_type_map), cmap="gray")
        else:
            pred_map = pred_maps[name]
            misclass = (gt_type_map > 0) & (pred_map != gt_type_map)
            false_pos = (gt_type_map == 0) & (pred_map > 0)
            error_map = np.zeros((*gt_type_map.shape, 3), dtype=np.float32)
            error_map[misclass] = [1.0, 0.0, 0.0]
            error_map[false_pos] = [0.0, 0.0, 1.0]
            ax.imshow(error_map)
        ax.axis("off")

    for col, name in enumerate(col_names):
        axes[0, col].set_title(name, fontsize=13, pad=8)

    axes[0, 0].set_ylabel("Type Map", fontsize=13, labelpad=12)
    axes[1, 0].set_ylabel("Overlay", fontsize=13, labelpad=12)
    axes[2, 0].set_ylabel("Error", fontsize=13, labelpad=12)

    fig.suptitle(title, fontsize=16, y=1.02)

    row_labels = ["Type Map", "Overlay", "Error"]
    for row, label in enumerate(row_labels):
        y = 1 - (row + 0.5) / len(row_labels)
        fig.text(-0.05, y, label, va="center", ha="left", fontsize=13, rotation=90)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    present = set(np.unique(gt_type_map))
    for pm in pred_maps.values():
        present |= set(np.unique(pm))
    present = sorted(i for i in present if i > 0)

    legend_patches = [
        Patch(color=lut[i], label=id_to_name[i])
        for i in present
    ]
    legend_patches.append(Patch(color="red", label="Misclassified nucleus"))
    legend_patches.append(Patch(color="blue", label="False positive (background)"))

    if save_path:
        fig_leg = plt.figure(figsize=(6, 0.6 + 0.3 * len(legend_patches)))
        plt.legend(
            handles=legend_patches,
            frameon=False,
            loc="center left",
        )
        plt.axis("off")
        leg_path = str(save_path).rstrip(".png") + "_legend.png"
        fig_leg.savefig(leg_path, dpi=100, bbox_inches="tight")
        plt.close(fig_leg)
        plt.close("all")
        return

    plt.figure(figsize=(6, 0.6 + 0.3 * len(legend_patches)))
    plt.legend(
        handles=legend_patches,
        frameon=False,
        loc="center left",
    )
    plt.axis("off")
    plt.show()
