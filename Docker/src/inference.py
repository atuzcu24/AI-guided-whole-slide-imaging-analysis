import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from cellvit.models.cell_segmentation.cellvit_sam_rosie_film import CellViTSAMRosieFiLM
from huggingface_hub import hf_hub_download


# ------------------------------------------------------
# 1. MODEL REPOSITORY DEFINITIONS
# ------------------------------------------------------

MODEL_REPOS = {
    "SamH_Rosie_FiLM_256": {
        "repo_id": "BerkTuzcuBU/SamH-Rosie-FiLM-256",
        "filename": "SamH-Rosie-FiLM-256.pth",
    }
}


def get_checkpoint(model_name: str) -> str:
    """
    Downloads the checkpoint from HuggingFace Hub if needed.
    Returns the local path.
    """
    repo = MODEL_REPOS[model_name]["repo_id"]
    fname = MODEL_REPOS[model_name]["filename"]
    ckpt_path = hf_hub_download(repo_id=repo, filename=fname)
    return ckpt_path


# ------------------------------------------------------
# 2. MODEL LOADING
# ------------------------------------------------------

def load_model(
        model_name: str,
        device: str = "cpu",
        ckpt_path: str = None,          
        ckpt_dir: str = None           
    ):
    """
    Loads the CellViT-SAM-Rosie-FiLM model for inference.

    Args:
        model_name (str): Name of the model repo on HuggingFace.
        device (str): "cpu" or "cuda".
        ckpt_path (str): Direct path to checkpoint (user-specified).
        ckpt_dir (str): Directory where checkpoints should be downloaded.

    Priority for selecting checkpoint:
        1. Use ckpt_path if provided.
        2. Download using ckpt_dir if provided.
        3. Download using environment variable CELLVIT_CKPT_DIR.
        4. Download into ./checkpoints.
    """

    if model_name != "SamH-Rosie-FiLM-256":
        raise ValueError(f"Unknown model: {model_name}")

    # ------------------------------
    # 1. Instantiate model
    # ------------------------------
    model = CellViTSAMRosieFiLM(
        model_path=None,
        num_nuclei_classes=6,    # background + 5 types
        num_tissue_classes=1,    # Lung only
        vit_structure="sam-h",
        drop_rate=0.0,
        regression_loss=False,
        rosie_hidden_dim=256,
        freeze_cellvit=True,
        freeze_rosie=True,
    )

    # ------------------------------
    # 2. Determine checkpoint path
    # ------------------------------

    
    if ckpt_path is not None:
        print(f"Using user-provided checkpoint: {ckpt_path}")

    else:
        
        ckpt_path = get_checkpoint(
            model_name=model_name,
            filename="model_best.pth",
            ckpt_dir=ckpt_dir
        )
        print(f"Downloaded checkpoint to: {ckpt_path}")

    # ------------------------------
    # 3. Load model weights
    # ------------------------------
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)

    # ------------------------------
    # 4. Final model setup
    # ------------------------------
    model.to(device)
    model.eval()

    print("Model loaded and ready.")
    return model



# ------------------------------------------------------
# 3. IMAGE PREPROCESSING
# ------------------------------------------------------

import albumentations as A
from albumentations.pytorch import ToTensorV2

val_transform = A.Compose([
    A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
    ToTensorV2(),
])

def preprocess_image(img_pil):
    img = np.array(img_pil.convert("RGB"))

    if img.shape[:2] != (256,256):
        raise ValueError("Input must be exactly 256x256 like training patches")

    out = val_transform(image=img)["image"]  # CHW tensor
    return out.unsqueeze(0)



# ------------------------------------------------------
# 4. RUN INFERENCE
# ------------------------------------------------------

def predict(model, img_tensor, device="cpu"):
    """
    Runs SAM-Rosie-FiLM inference.
    Returns:
        - nuclei_binary_map
        - hv_map
        - nuclei_type_map
    """

    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        outputs = model(img_tensor, retrieve_tokens=False)

    return {
        "nuclei_binary_map": outputs["nuclei_binary_map"].cpu(),
        "hv_map": outputs["hv_map"].cpu(),
        "nuclei_type_map": outputs["nuclei_type_map"].cpu(),
    }

