# -*- coding: utf-8 -*-
# Running an Experiment Using CellViT cell segmentation network (train the segmentation network)
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen


import os, sys
import numpy as np
import torch
import torch.serialization as ts
import matplotlib.pyplot as plt
from pathlib import Path

# path bootstrap 
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.abspath(current_dir))
sys.path.append(project_root)

# import repo modules
import cellvit.training.trainer.trainer_cellvit as T
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

# --- runtime patches ---
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

#ts.add_safe_globals([np.core.multiarray.scalar])
_orig_torch_load = torch.load
def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _torch_load_compat

import inspect

_old_gen = T.CellViTTrainer.generate_example_image

def _safe_generate_example_image(self, imgs, predictions, gt, num_nuclei_classes, num_images=2):
    # clamp num_images to avoid small last batch crashes
    if imgs is not None:
        num_images = min(num_images, imgs.shape[0])

    # call the original EXACTLY as written
    return _old_gen(
        imgs,            # 1
        predictions,     # 2
        gt,              # 3
        num_nuclei_classes,  # 4
        num_images       # 5
    )

T.CellViTTrainer.generate_example_image = _safe_generate_example_image



def save_val_preview(inputs, outputs, gts, epoch, save_dir, num_classes=6, index=0):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    img = inputs[0].detach().cpu().permute(1, 2, 0).numpy()
    img = ((img * 0.5) + 0.5).clip(0, 1)
    base = (img * 255).astype(np.uint8)

    gt_type   = torch.argmax(gts["nuclei_type_map"],   dim=1)[0].detach().cpu().numpy()
    pred_type = torch.argmax(outputs["nuclei_type_map"], dim=1)[0].detach().cpu().numpy()

    npy_path = Path(save_dir) / f"epoch_{epoch:03d}_idx{index}_pred_type.npy"
    png_path = Path(save_dir) / f"epoch_{epoch:03d}_idx{index}_pred_type.png"
    plt.imsave(png_path, pred_type, cmap="tab10", vmin=0, vmax=max(num_classes-1, 1))
    np.save(npy_path, pred_type.astype(np.uint8))

    denom = max(num_classes - 1, 1)
    pred_rgb = (plt.get_cmap("tab10")(pred_type / denom)[:, :, :3] * 255).astype(np.uint8)
    overlay = pred_rgb if not _HAS_CV2 else cv2.addWeighted(base, 0.6, pred_rgb, 0.4, 0)
    ov_path = Path(save_dir) / f"epoch_{epoch:03d}_idx{index}_overlay.png"
    if _HAS_CV2:
        cv2.imwrite(str(ov_path), overlay[:, :, ::-1])
    else:
        plt.imsave(ov_path, overlay)

    gt_png_path = Path(save_dir) / f"epoch_{epoch:03d}_idx{index}_gt_type.png"
    plt.imsave(gt_png_path, gt_type, cmap="tab10", vmin=0, vmax=max(num_classes-1, 1))

    print(f"✔ Saved: {npy_path.name}, {png_path.name}, {ov_path.name}, {gt_png_path.name}")

T.save_val_preview = save_val_preview


import wandb

os.environ["WANDB__SERVICE_WAIT"] = "300"

from cellvit.training.base_ml.base_cli import ExperimentBaseParser
from cellvit.training.evaluate.inference_cellvit_experiment_pannuke import (
    InferenceCellViT,
)
from cellvit.training.experiments.experiment_cellvit_pannuke import (
    ExperimentCellVitPanNuke,
)

if __name__ == "__main__":
    # Parse arguments
    configuration_parser = ExperimentBaseParser()
    configuration = configuration_parser.parse_arguments()

    if configuration["data"]["dataset"].lower() == "pannuke":
        experiment_class = ExperimentCellVitPanNuke
        
    
    # Setup experiment
    if "checkpoint" in configuration:
        # continue checkpoint
        experiment = experiment_class(
            default_conf=configuration,
            checkpoint=configuration["checkpoint"],
            just_load_model=configuration["just_load_model"],
        )

        outdir = experiment.run_experiment()
        inference = InferenceCellViT(
            run_dir=outdir,
            gpu=configuration["gpu"],
            checkpoint_name=configuration["eval_checkpoint"],
            magnification=configuration["data"].get("magnification", 40),
        )
        (
            trained_model,
            inference_dataloader,
            dataset_config,
        ) = inference.setup_patch_inference()
        inference.run_patch_inference(
            trained_model, inference_dataloader, dataset_config, generate_plots=False
        )
    else:
        experiment = experiment_class(default_conf=configuration)
        if configuration["run_sweep"] is True:
            # run new sweep
            sweep_configuration = experiment_class.extract_sweep_arguments(
                configuration
            )
            os.environ["WANDB_DIR"] = os.path.abspath(
                configuration["logging"]["wandb_dir"]
            )
            sweep_id = wandb.sweep(
                sweep=sweep_configuration, project=configuration["logging"]["project"]
            )
            wandb.agent(sweep_id=sweep_id, function=experiment.run_experiment)
        elif "agent" in configuration and configuration["agent"] is not None:
            # add agent to already existing sweep, not run sweep must be set to true
            configuration["run_sweep"] = True
            os.environ["WANDB_DIR"] = os.path.abspath(
                configuration["logging"]["wandb_dir"]
            )
            wandb.agent(
                sweep_id=configuration["agent"], function=experiment.run_experiment
            )
        else:
            # casual run
            outdir = experiment.run_experiment()
            #Due to CUDA memory issues adding empty cache
            
            torch.cuda.empty_cache()

            inference = InferenceCellViT(
                run_dir=outdir,
                gpu=configuration["gpu"],
                checkpoint_name=configuration["eval_checkpoint"],
                magnification=configuration["data"].get("magnification", 40),
            )
            (
                trained_model,
                inference_dataloader,
                dataset_config,
            ) = inference.setup_patch_inference()
            inference.run_patch_inference(
                trained_model,
                inference_dataloader,
                dataset_config,
                generate_plots=False,
            )
    wandb.finish()
