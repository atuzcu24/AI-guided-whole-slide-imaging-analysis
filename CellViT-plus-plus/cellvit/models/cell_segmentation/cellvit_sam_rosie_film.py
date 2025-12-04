import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from cellvit.models.cell_segmentation.cellvit_sam import CellViTSAM


class RosieFiLMZ4(nn.Module):
    """
    FiLM conditioning applied on z4 feature map (encoder output).

    z4: [B, H, W, C_feat]  (C_feat = prompt_embed_dim = 256 for SAM)
    rosie_features: [B, 50]

    We compute gamma, beta in R^{C_feat} and broadcast over H, W.
    """
    def __init__(self, rosie_dim=50, feat_dim=256, hidden_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(rosie_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feat_dim * 2),
        )

    def forward(self, z4, rosie_features):
        """
        z4: [B, H, W, C]
        rosie_features: [B, 50]
        """
        film = self.mlp(rosie_features)         # [B, 2*C]
        gamma, beta = film.chunk(2, dim=-1)     # each [B, C]

        # reshape for broadcasting over H,W
        gamma = gamma.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, C]
        beta  = beta.unsqueeze(1).unsqueeze(1)   # [B, 1, 1, C]

        return z4 * gamma + beta


class CellViTSAMRosieFiLM(CellViTSAM):
    """
    CellViT-SAM + Rosie FiLM fusion on encoder output z4.

    - Keeps original CellViTSAM structure.
    - Adds a frozen ConvNeXt-small (ROSIE backbone) producing a 50-d vector.
    - Uses FiLM (gamma,beta) to modulate z4 before decoders.
    """

    def __init__(
        self,
        model_path,
        num_nuclei_classes,
        num_tissue_classes,
        vit_structure: str = "sam-h",
        drop_rate: float = 0.0,
        regression_loss: bool = False,
        rosie_hidden_dim: int = 256,
        freeze_cellvit=True, 
        freeze_rosie=True,
    ):
        super().__init__(
            model_path=model_path,
            num_nuclei_classes=num_nuclei_classes,
            num_tissue_classes=num_tissue_classes,
            vit_structure=vit_structure,
            drop_rate=drop_rate,
            regression_loss=regression_loss,
        )

        self.freeze_cellvit = freeze_cellvit
        self.freeze_rosie = freeze_rosie

        # --- ROSIE backbone (ConvNeXt-small → 50 outputs) ---
        self.rosie_head_dim = 50
        self.rosie_model = models.convnext_small(weights="IMAGENET1K_V1")
        self.rosie_model.classifier[2] = nn.Linear(
            self.rosie_model.classifier[2].in_features,
            self.rosie_head_dim,
        )

        # Freeze ROSIE
        if self.freeze_rosie:
            for p in self.rosie_model.parameters():
                p.requires_grad = False
            self.rosie_model.eval()

        # Freeze SAM encoder
        if self.freeze_cellvit:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # ImageNet normalization for ROSIE branch
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("rosie_mean", imagenet_mean, persistent=False)
        self.register_buffer("rosie_std", imagenet_std, persistent=False)

        # FiLM module on z4 (C_feat = prompt_embed_dim = 256)
        feat_dim = 1280       # TODO: Fix this part
        self.z4_film = RosieFiLMZ4(
            rosie_dim=self.rosie_head_dim,
            feat_dim=feat_dim,
            hidden_dim=rosie_hidden_dim
        )

    def _rosie_preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert CellViT input (normalized with mean=0.5,std=0.5) into
        ImageNet-normalized input expected by ConvNeXt.

        x is assumed to be in [-1, 1] due to Albumentations normalize (mean=0.5,std=0.5).
        """
        # x in [-1, 1] → bring back to [0,1]
        x_01 = (x + 1.0) / 2.0

        # resize to 224x224 for ConvNeXt
        x_resized = F.interpolate(
            x_01,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )

        # ImageNet normalization
        x_norm = (x_resized - self.rosie_mean) / self.rosie_std
        return x_norm

    def load_pretrained_encoder(self, model_path):
        return super().load_pretrained_encoder(model_path)

    def forward(self, x: torch.Tensor, retrieve_tokens: bool = False):
        """
        Same outputs as CellViTSAM, but z4 is FiLM-modulated using ROSIE.
        """
        assert (
            x.shape[-2] % self.patch_size == 0
        ), "Img must have a shape divisible by patch_size (height)"
        assert (
            x.shape[-1] % self.patch_size == 0
        ), "Img must have a shape divisible by patch_size (width)"

        out_dict = {}

        # 1) Compute ROSIE features (frozen, no grad)
        with torch.no_grad():
            x_rosie = self._rosie_preprocess(x)
            rosie_features = self.rosie_model(x_rosie)  # [B, 50]

        # 2) Original encoder forward
        classifier_logits, _, z = self.encoder(x)
        out_dict["tissue_types"] = self.classifier_head(classifier_logits)

        z0 = x
        z1, z2, z3, z4 = z


        # 3) FiLM on z4 (before permute)
        # z4 currently: [B, H, W, C] (from ViTCellViTDeit)
        raw_z4 = z4.clone()  # save before FiLM
        # Apply FiLM
        z4 = self.z4_film(z4, rosie_features)

        # 4) Permute & decoders (unchanged from CellViTSAM)
        z4 = z4.permute(0, 3, 1, 2)
        z3 = z3.permute(0, 3, 1, 2)
        z2 = z2.permute(0, 3, 1, 2)
        z1 = z1.permute(0, 3, 1, 2)

        if self.regression_loss:
            nb_map = self._forward_upsample(
                z0, z1, z2, z3, z4, self.nuclei_binary_map_decoder
            )
            out_dict["nuclei_binary_map"] = nb_map[:, :2, :, :]
            out_dict["regression_map"] = nb_map[:, 2:, :, :]
        else:
            out_dict["nuclei_binary_map"] = self._forward_upsample(
                z0, z1, z2, z3, z4, self.nuclei_binary_map_decoder
            )

        out_dict["hv_map"] = self._forward_upsample(
            z0, z1, z2, z3, z4, self.hv_map_decoder
        )
        out_dict["nuclei_type_map"] = self._forward_upsample(
            z0, z1, z2, z3, z4, self.nuclei_type_maps_decoder
        )

        if retrieve_tokens:
            out_dict["tokens"] = raw_z4

        return out_dict

