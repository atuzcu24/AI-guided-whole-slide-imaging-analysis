"""
Early fusion of ROSIE marker features with H&E input before CellViT-SAM encoder.

No FiLM. Fuses ROSIE vector or spatial maps by concatenation to input channels,
then expands the encoder's first conv layer to accept the extra channels.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from cellvit.models.cell_segmentation.cellvit_sam import CellViTSAM
from cellvit.models.utils.rosie_markers import marker_names_to_indices


def expand_input_layer(module: nn.Module, new_in_channels: int, mode: str = "zeros") -> bool:
    """
    Find the FIRST Conv2d with in_channels=3 in the encoder and expand it to new_in_channels.
    Copies existing weights to channels 0:3 and initializes extra channels to 0.

    Handles encoder.patch_embed.proj (PatchEmbed's Conv2d).

    Returns True if expansion was performed, False otherwise.
    """
    if mode != "zeros":
        raise ValueError(f"expand_input_layer only supports mode='zeros', got {mode}")

    # Check patch_embed.proj (standard SAM encoder structure)
    proj = getattr(module, "patch_embed", None)
    if proj is not None:
        proj_conv = getattr(proj, "proj", None)
        if isinstance(proj_conv, nn.Conv2d) and proj_conv.in_channels == 3:
            old_conv = proj_conv
            new_conv = nn.Conv2d(
                new_in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )
            with torch.no_grad():
                new_conv.weight[:, :3] = old_conv.weight
                new_conv.weight[:, 3:] = 0
                if old_conv.bias is not None:
                    new_conv.bias.copy_(old_conv.bias)
            proj.proj = new_conv
            print(f"Expanded encoder input channels 3 -> {new_in_channels}; extra channels initialized to 0")
            return True

    # Fallback: recursive search for first Conv2d with in_channels=3
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d) and child.in_channels == 3:
            old_conv = child
            new_conv = nn.Conv2d(
                new_in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )
            with torch.no_grad():
                new_conv.weight[:, :3] = old_conv.weight
                new_conv.weight[:, 3:] = 0
                if old_conv.bias is not None:
                    new_conv.bias.copy_(old_conv.bias)
            setattr(module, name, new_conv)
            print(f"Expanded encoder input channels 3 -> {new_in_channels}; extra channels initialized to 0")
            return True
        expanded = expand_input_layer(child, new_in_channels, mode)
        if expanded:
            return True

    return False


class CellViTSAMRosieEarlyFusion(CellViTSAM):
    """
    CellViT-SAM with early fusion of ROSIE features into input channels.

    - ROSIE branch: frozen ConvNeXt-small -> vector (B,50) or spatial features
    - Early fusion: concatenate ROSIE channels to H&E input before encoder
    - Encoder: expanded to accept 3 + extra channels
    """

    def __init__(
        self,
        model_path: str,
        num_nuclei_classes: int,
        num_tissue_classes: int,
        vit_structure: str = "sam-h",
        drop_rate: float = 0.0,
        regression_loss: bool = False,
        freeze_cellvit: bool = True,
        freeze_rosie: bool = True,
        rosie_weights_path: str | None = None,
        early_fusion_type: str = "vec_broadcast",
        early_fusion_compress_out_channels: int = 8,
        rosie_marker_subset: list[str] | None = None,
        rosie_marker_subset_indices: list[int] | None = None,
        early_fusion_detach_rosie: bool = True,
        debug_forward_log: bool = True,
    ):
        super().__init__(
            model_path=model_path,
            num_nuclei_classes=num_nuclei_classes,
            num_tissue_classes=num_tissue_classes,
            vit_structure=vit_structure,
            drop_rate=drop_rate,
            regression_loss=regression_loss,
        )

        if early_fusion_type not in ("vec_broadcast", "map_compress"):
            raise ValueError(
                f"early_fusion_type must be 'vec_broadcast' or 'map_compress', got {early_fusion_type}"
            )

        self.freeze_cellvit = freeze_cellvit
        self.freeze_rosie = freeze_rosie
        self.early_fusion_type = early_fusion_type
        self.early_fusion_compress_out_channels = early_fusion_compress_out_channels
        self.early_fusion_detach_rosie = early_fusion_detach_rosie
        self.debug_forward_log = debug_forward_log
        self._forward_logged = False

        # Marker subset (same 9 as FiLM configs)
        self.rosie_marker_indices = marker_names_to_indices(
            rosie_marker_subset, rosie_marker_subset_indices
        )
        self.rosie_dim = len(self.rosie_marker_indices)

        # ROSIE backbone (ConvNeXt-small)
        self.rosie_head_dim = 50
        self.rosie_model = models.convnext_small(weights="IMAGENET1K_V1")
        self.rosie_model.classifier[2] = nn.Linear(
            self.rosie_model.classifier[2].in_features,
            self.rosie_head_dim,
        )

        if rosie_weights_path is not None:
            ckpt = torch.load(rosie_weights_path, map_location="cpu")
            state = ckpt
            if isinstance(ckpt, dict):
                state = (
                    ckpt.get("state_dict")
                    or ckpt.get("model_state_dict")
                    or ckpt.get("model")
                    or ckpt
                )
            cleaned = {}
            for k, v in state.items():
                nk = k
                for pref in ("module.", "model.", "net.", "backbone."):
                    if nk.startswith(pref):
                        nk = nk[len(pref) :]
                cleaned[nk] = v
            missing, unexpected = self.rosie_model.load_state_dict(cleaned, strict=False)
            print(f"Loaded ROSIE weights from: {rosie_weights_path}")
            print(f"   missing keys: {len(missing)} | unexpected keys: {len(unexpected)}")

        if self.freeze_rosie:
            for p in self.rosie_model.parameters():
                p.requires_grad = False
            self.rosie_model.eval()

        # ImageNet normalization for ROSIE
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("rosie_mean", imagenet_mean, persistent=False)
        self.register_buffer("rosie_std", imagenet_std, persistent=False)

        # Early fusion extra channels
        if early_fusion_type == "vec_broadcast":
            self.extra_channels = self.rosie_dim
        else:
            self.extra_channels = early_fusion_compress_out_channels
            # For map_compress: need to project ConvNeXt spatial features
            # ConvNeXt-small last stage: 768 channels, 7x7 for 224 input
            self._rosie_spatial_dim = 768
            self.rosie_spatial_compress = nn.Conv2d(
                self._rosie_spatial_dim,
                early_fusion_compress_out_channels,
                kernel_size=1,
            )

        self.input_channels = 3 + self.extra_channels
        self._encoder_expanded = False

        if self.freeze_cellvit:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # decoder0 expects 3 channels for z0; we must use fused_x which has 3+extra
        # The parent CellViT decoder0 is Conv2DBlock(3, 32, ...) - hardcoded 3
        # We need to change it to input_channels. Override decoder0.
        from cellvit.models.cell_segmentation.cellvit import CellViT
        from cellvit.models.utils.blocks import Conv2DBlock
        self.decoder0 = nn.Sequential(
            Conv2DBlock(self.input_channels, 32, 3, dropout=self.drop_rate),
            Conv2DBlock(32, 64, 3, dropout=self.drop_rate),
        )

        print(
            f"[CellViTSAMRosieEarlyFusion] early_fusion_type={early_fusion_type} | "
            f"extra_channels={self.extra_channels} | rosie_markers={self.rosie_dim} | "
            f"detach_rosie={early_fusion_detach_rosie}"
        )

    def load_pretrained_encoder(self, model_path):
        """Load pretrained SAM encoder, then expand input layer for early fusion channels."""
        super().load_pretrained_encoder(model_path)
        if not self._encoder_expanded:
            expand_input_layer(self.encoder, self.input_channels, mode="zeros")
            self._encoder_expanded = True

    def _rosie_preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """H&E normalized [-1,1] -> ImageNet normalized [0,1] resized to 224x224."""
        x_01 = (x + 1.0) / 2.0
        x_resized = F.interpolate(x_01, size=(224, 224), mode="bilinear", align_corners=False)
        return (x_resized - self.rosie_mean) / self.rosie_std

    def _get_rosie_vector(self, x: torch.Tensor) -> torch.Tensor:
        """Return (B, rosie_dim) vector, optionally detached."""
        x_rosie = self._rosie_preprocess(x)
        ctx = torch.no_grad() if self.early_fusion_detach_rosie else torch.enable_grad()
        with ctx:
            rosie_full = self.rosie_model(x_rosie)
        rosie_vec = rosie_full[:, self.rosie_marker_indices]
        if self.early_fusion_detach_rosie:
            rosie_vec = rosie_vec.detach()
        return rosie_vec

    def _get_rosie_spatial(self, x: torch.Tensor) -> torch.Tensor:
        """Return (B, early_fusion_compress_out_channels, H, W) spatial map."""
        x_rosie = self._rosie_preprocess(x)
        ctx = torch.no_grad() if self.early_fusion_detach_rosie else torch.enable_grad()
        with ctx:
            # Forward through ConvNeXt features only (before avgpool + classifier)
            feat = self.rosie_model.features(x_rosie)
        if self.early_fusion_detach_rosie:
            feat = feat.detach()
        # feat: (B, 768, 7, 7)
        compressed = self.rosie_spatial_compress(feat)
        return compressed

    def forward(self, x: torch.Tensor, retrieve_tokens: bool = False):
        assert x.shape[-2] % self.patch_size == 0
        assert x.shape[-1] % self.patch_size == 0
        B, _, H, W = x.shape

        out_dict = {}

        # 1) ROSIE features
        if self.early_fusion_type == "vec_broadcast":
            rosie_vec = self._get_rosie_vector(x)  # (B, K)
            # Broadcast to (B, K, H, W)
            rosie_map = rosie_vec.unsqueeze(-1).unsqueeze(-1).expand(
                B, self.rosie_dim, H, W
            )
        else:
            rosie_map = self._get_rosie_spatial(x)  # (B, C, 7, 7)
            rosie_map = F.interpolate(
                rosie_map, size=(H, W), mode="bilinear", align_corners=False
            )

        # 2) Concatenate: fused_x (B, 3+extra, H, W)
        fused_x = torch.cat([x, rosie_map], dim=1)

        if self.debug_forward_log and not self._forward_logged:
            first_conv = getattr(
                getattr(self.encoder, "patch_embed", None), "proj", None
            )
            in_ch = first_conv.in_channels if first_conv is not None else "?"
            print(f"[EarlyFusion] fused_x shape: {fused_x.shape} | encoder expects in_channels: {in_ch}")
            self._forward_logged = True

        # 3) Encoder forward with fused input
        classifier_logits, _, z = self.encoder(fused_x)
        out_dict["tissue_types"] = self.classifier_head(classifier_logits)

        z0 = fused_x
        z1, z2, z3, z4 = z

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
            out_dict["tokens"] = z4

        return out_dict
