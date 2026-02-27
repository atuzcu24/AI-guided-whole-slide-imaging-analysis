from pathlib import Path
import importlib.util
import json
import logging
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from cellvit.models.cell_segmentation.cellvit_virchow import CellViTVirchow

logger = logging.getLogger(__name__)


def _recompute_rosie_topk(
    dataset_path: str,
    k: int,
    method: str,
    out_json: str,
    seed: int = 42,
    rosie_weights_path: str | None = None,
) -> None:
    """Load compute_rosie_topk and run recompute, then rewrite atomically."""
    project_root = Path(__file__).resolve().parents[5]
    script_path = project_root / "Datasets" / "pannuke_hf_cellvit" / "scripts" / "compute_rosie_topk.py"
    if not script_path.is_file():
        raise FileNotFoundError(f"compute_rosie_topk script not found: {script_path}")
    spec = importlib.util.spec_from_file_location("compute_rosie_topk", script_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["compute_rosie_topk"] = mod
    spec.loader.exec_module(mod)
    mod.run_compute_rosie_topk(
        dataset_path=dataset_path,
        k=k,
        method=method,
        out_json=out_json,
        seed=seed,
        rosie_weights=rosie_weights_path,
    )


class RosieFiLM2D(nn.Module):
    """
    FiLM for 2D feature maps in BCHW format.
      z: [B, C, H, W]
      rosie_features: [B, rosie_dim]
      returns: z * gamma + beta (gamma/beta broadcast over H,W)

    film_init: "default" (PyTorch init) or "identity" (gamma=1, beta=0).
    film_mode: "full" (both gamma and beta) or "beta_only" (gamma=1, learn beta only).
    """
    def __init__(self, rosie_dim: int, feat_dim: int, hidden_dim: int = 256, film_init: str = "default", film_mode: str = "full"):
        super().__init__()
        self.feat_dim = feat_dim
        self.film_mode = (film_mode or "full").lower()
        self.mlp = nn.Sequential(
            nn.Linear(rosie_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feat_dim * 2),
        )
        if film_init == "identity":
            self._init_identity()

    def _init_identity(self) -> None:
        """Initialize last Linear so gamma=1, beta=0 at start."""
        last_linear = self.mlp[2]
        nn.init.zeros_(last_linear.weight)
        with torch.no_grad():
            last_linear.bias[: self.feat_dim] = 1.0
            last_linear.bias[self.feat_dim :] = 0.0

    def forward(self, z: torch.Tensor, rosie_features: torch.Tensor) -> torch.Tensor:
        # z: [B, C, H, W]
        B, C, _, _ = z.shape
        film = self.mlp(rosie_features)      # [B, 2C]
        gamma, beta = film.chunk(2, dim=-1)  # [B, C], [B, C]

        # Force identity (gamma=1, beta=0) for ablation:
        # - film_force_identity_train: always (training + inference)
        # - film_force_identity: inference-only (set at inference time)
        force_id = getattr(self, "film_force_identity_train", False) or (
            (not self.training) and getattr(self, "film_force_identity", False)
        )
        if force_id:
            gamma = torch.ones_like(gamma)
            beta = torch.zeros_like(beta)

        if self.film_mode == "beta_only":
            gamma = torch.ones_like(gamma)

        if getattr(self, "log_film_stats", False):
            g = gamma.detach()
            b = beta.detach()
            if not hasattr(self, "_film_gamma_sum"):
                self._film_gamma_sum = 0.0
                self._film_gamma_sq_sum = 0.0
                self._film_beta_sum = 0.0
                self._film_beta_sq_sum = 0.0
                self._film_n = 0
            self._film_gamma_sum += g.sum().item()
            self._film_gamma_sq_sum += (g ** 2).sum().item()
            self._film_beta_sum += b.sum().item()
            self._film_beta_sq_sum += (b ** 2).sum().item()
            self._film_n += g.numel()

        gamma = gamma.view(B, C, 1, 1)
        beta  = beta.view(B, C, 1, 1)

        return z * gamma + beta

    def get_film_stats_and_reset(self):
        """Return mean/std/min/max for gamma and beta if logged, else None. Resets accumulators."""
        if not getattr(self, "_film_n", 0):
            return None
        n = self._film_n
        mean_g = self._film_gamma_sum / n
        var_g = (self._film_gamma_sq_sum / n) - (mean_g ** 2)
        std_g = (var_g ** 0.5) if var_g > 0 else 0.0
        mean_b = self._film_beta_sum / n
        var_b = (self._film_beta_sq_sum / n) - (mean_b ** 2)
        std_b = (var_b ** 0.5) if var_b > 0 else 0.0
        self._film_gamma_sum = 0.0
        self._film_gamma_sq_sum = 0.0
        self._film_beta_sum = 0.0
        self._film_beta_sq_sum = 0.0
        self._film_n = 0
        return {"gamma_mean": mean_g, "gamma_std": std_g, "beta_mean": mean_b, "beta_std": std_b}

class CellViTVirchowRosieFiLM(CellViTVirchow):
    """
    Virchow + optional Rosie-FiLM fusion on selected feature maps (z1-z4).
    Controlled entirely by config (enable/disable + which layers).
    """

    def __init__(
        self,
        model_virchow_path,
        num_nuclei_classes,
        num_tissue_classes,
        rosie_hidden_dim=256,
        freeze_rosie=True,
        rosie_weights_path=None,
        film_enabled=True,
        film_layers=("z4",),
        film_feat_dims=None,
        debug_print_z_shapes=False,
        rosie_subset_indices=None,
        rosie_topk=None,
        rosie_topk_method="energy",
        rosie_topk_cache_path=None,
        rosie_topk_dataset_path=None,
        rosie_topk_seed=None,
        **kwargs,
    ):
        super().__init__(
            model_virchow_path=model_virchow_path,
            num_nuclei_classes=num_nuclei_classes,
            num_tissue_classes=num_tissue_classes,
        )

        self.debug_print_z_shapes = debug_print_z_shapes
        self._printed_z_shapes = False

        # ---- ROSIE branch (ConvNeXt-small -> 50-d) ----
        self.rosie_head_dim = 50
        self.rosie_model = models.convnext_small(weights="IMAGENET1K_V1")
        self.rosie_model.classifier[2] = nn.Linear(
            self.rosie_model.classifier[2].in_features, self.rosie_head_dim
        )

        if rosie_weights_path is not None:
            ckpt = torch.load(rosie_weights_path, map_location="cpu")
            state = (ckpt.get("state_dict") or ckpt.get("model_state_dict") or ckpt.get("model") or ckpt) \
                    if isinstance(ckpt, dict) else ckpt
            cleaned = {}
            for k, v in state.items():
                nk = k
                for pref in ("module.", "model.", "net.", "backbone."):
                    if nk.startswith(pref):
                        nk = nk[len(pref):]
                cleaned[nk] = v
            self.rosie_model.load_state_dict(cleaned, strict=False)

        self.freeze_rosie = freeze_rosie
        if freeze_rosie:
            for p in self.rosie_model.parameters():
                p.requires_grad = False
            self.rosie_model.eval()

        imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("rosie_mean", imagenet_mean, persistent=False)
        self.register_buffer("rosie_std", imagenet_std, persistent=False)

        # ---- FiLM config ----
        self.film_enabled = film_enabled
        self.film_layers = set([l.lower() for l in (film_layers or tuple())]) if film_enabled else set()
        self.film_force_identity = False  # set at inference for ablation (gamma=1, beta=0)
        self.film_force_identity_train = kwargs.get("film_force_identity_train", False)  # always identity (train+eval)
        self.film_feat_dims = {k.lower(): int(v) for k, v in (film_feat_dims or {}).items()}

        # ---- Rosie channel subset / top-k (optional, no change if absent) ----
        self.rosie_subset_indices = None  # list[int] or None
        if rosie_subset_indices is not None and len(rosie_subset_indices) > 0:
            self.rosie_subset_indices = [int(i) for i in rosie_subset_indices]
        elif rosie_topk is not None and rosie_topk > 0 and rosie_topk_cache_path:
            cache_path = Path(rosie_topk_cache_path)
            loaded = False
            if cache_path.is_file():
                try:
                    with open(cache_path) as f:
                        data = json.load(f)
                    self.rosie_subset_indices = data.get("indices", data.get("indices_list", []))
                    if len(self.rosie_subset_indices) != int(rosie_topk):
                        logger.warning(
                            f"rosie_topk_cache has {len(self.rosie_subset_indices)} indices, expected {rosie_topk}"
                        )
                    loaded = True
                except json.JSONDecodeError:
                    logger.warning(
                        f"rosie_topk_cache_path={cache_path} invalid (partial write?). Recomputing."
                    )
            if not loaded:
                if not rosie_topk_dataset_path:
                    raise ValueError(
                        f"rosie_topk_cache_path={cache_path} missing or corrupt. "
                        "Pass rosie_topk_dataset_path for on-demand recompute."
                    )
                _recompute_rosie_topk(
                    dataset_path=str(rosie_topk_dataset_path),
                    k=int(rosie_topk),
                    method=rosie_topk_method or "energy",
                    out_json=str(cache_path),
                    seed=int(rosie_topk_seed) if rosie_topk_seed is not None else 42,
                    rosie_weights_path=rosie_weights_path,
                )
                with open(cache_path) as f:
                    data = json.load(f)
                self.rosie_subset_indices = data.get("indices", data.get("indices_list", []))
            logger.info(
                "rosie_topk_cache_path=%s loaded=%s"
                % (str(cache_path), loaded)
            )

        self.rosie_dim_for_film = (
            len(self.rosie_subset_indices) if self.rosie_subset_indices else self.rosie_head_dim
        )
        if self.film_enabled and len(self.film_layers) > 0:
            if self.rosie_subset_indices:
                logger.info(
                    f"Using Rosie channels: {self.rosie_subset_indices} | rosie_dim={self.rosie_dim_for_film}"
                )
            else:
                logger.info(f"Using Rosie channels: all {self.rosie_head_dim} | rosie_dim={self.rosie_dim_for_film}")

        film_init = kwargs.get("film_init", "default")
        film_mode = kwargs.get("film_mode", "full")
        self.conditioning_mode_train = kwargs.get("conditioning_mode_train", "normal")
        # Inference-time ablation; set by inference script so hasattr() is True and CLI can override
        self.conditioning_mode = "normal"
        self.conditioning_mode_infer = "normal"
        self.subset_indices = ""

        # NP spatial prior from ROSIE (for L_sup, L_bd losses)
        self.rosie_make_spatial_prior = kwargs.get("rosie_make_spatial_prior", False)
        self.rosie_prior_from = kwargs.get("rosie_prior_from", "rosie_backbone")
        self.rosie_prior_channels = int(kwargs.get("rosie_prior_channels", 50))
        self._rosie_to_marker = None
        self._marker_to_prior = None
        if self.rosie_make_spatial_prior:
            if self.rosie_prior_from == "rosie_backbone":
                # ConvNeXt-small features output 768 channels
                _C = 768
                self._rosie_to_marker = nn.Conv2d(_C, self.rosie_dim_for_film, kernel_size=1)
                self._marker_to_prior = nn.Conv2d(self.rosie_dim_for_film, 1, kernel_size=1)
            else:
                # rosie_classifier_broadcast: marker stack from [B,K] vector
                self._marker_to_prior = nn.Conv2d(self.rosie_dim_for_film, 1, kernel_size=1)

        self.film_blocks = nn.ModuleDict()
        for layer in ("z1", "z2", "z3", "z4"):
            if layer in self.film_layers:
                if layer not in self.film_feat_dims:
                    raise ValueError(f"film_feat_dims['{layer}'] required when enabling FiLM on {layer}")
                self.film_blocks[layer] = RosieFiLM2D(
                    rosie_dim=self.rosie_dim_for_film,
                    feat_dim=self.film_feat_dims[layer],
                    hidden_dim=int(rosie_hidden_dim),
                    film_init=film_init,
                    film_mode=film_mode,
                )

    def _rosie_preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x_01 = (x + 1.0) / 2.0
        x_resized = F.interpolate(x_01, size=(224, 224), mode="bilinear", align_corners=False)
        return (x_resized - self.rosie_mean) / self.rosie_std

    def _rosie_forward_spatial(self, x: torch.Tensor) -> tuple:
        """
        Extract spatial features from ROSIE backbone robustly.
        Returns (feat4d, mode_used) where feat4d is [B,C,h,w] or None.
        mode_used: "spatial" | "not_spatial" | "none"
        """
        feat = None
        if hasattr(self.rosie_model, "forward_features"):
            feat = self.rosie_model.forward_features(x)
        elif hasattr(self.rosie_model, "features"):
            feat = self.rosie_model.features(x)
        elif hasattr(self.rosie_model, "backbone"):
            feat = self.rosie_model.backbone(x)
        else:
            return (None, "none")
        if feat is None or not isinstance(feat, torch.Tensor):
            return (None, "none")
        if feat.dim() != 4:
            return (None, "not_spatial")
        return (feat, "spatial")

    def forward(self, x: torch.Tensor, retrieve_tokens=False):
        bs = x.shape[0]
        input_shape = x.shape[2]
        rescale_value = self.input_rescale_dict[input_shape]
        x_rescaled = F.interpolate(x, size=(rescale_value, rescale_value), mode="area")

        # Clear prior from previous forward (set only when rosie_make_spatial_prior)
        self._last_np_prior_s = None

        # Rosie features
        rosie_features = None
        if self.film_enabled and len(self.film_blocks) > 0:
            # When freeze_rosie: use no_grad. When unfrozen: allow gradients for end-to-end training.
            if self.freeze_rosie:
                with torch.no_grad():
                    rosie_features = self.rosie_model(self._rosie_preprocess(x_rescaled))
            else:
                rosie_features = self.rosie_model(self._rosie_preprocess(x_rescaled))
            if self.rosie_subset_indices is not None:
                rosie_features = rosie_features[:, self.rosie_subset_indices]
            # Inference-time conditioning ablation (zeros / shuffle / subset9)
            if not self.training:
                mode = getattr(self, "conditioning_mode_infer", None) or getattr(
                    self, "conditioning_mode", "normal"
                )
                if mode == "zeros":
                    rosie_features = torch.zeros_like(rosie_features)
                elif mode == "shuffle":
                    B = rosie_features.shape[0]
                    perm = torch.randperm(B, device=rosie_features.device)
                    rosie_features = rosie_features[perm]
                elif mode == "subset9":
                    subset_str = getattr(self, "subset_indices", "")
                    if subset_str:
                        try:
                            indices = [int(x.strip()) for x in subset_str.split(",") if x.strip()]
                            idx_set = set(indices)
                            mask = torch.ones_like(rosie_features, device=rosie_features.device)
                            for i in range(rosie_features.shape[1]):
                                if i not in idx_set:
                                    mask[:, i] = 0
                            rosie_features = rosie_features * mask
                        except ValueError:
                            pass
            elif self.training and getattr(self, "conditioning_mode_train", "normal") == "zeros":
                rosie_features = torch.zeros_like(rosie_features)

        # NP spatial prior s [B,1,H,W] for L_sup / L_bd losses (trainer resizes if needed)
        if self.rosie_make_spatial_prior:
            x_rosie = self._rosie_preprocess(x_rescaled)
            use_spatial = False
            if self.rosie_prior_from == "rosie_backbone":
                with torch.no_grad() if self.freeze_rosie else torch.enable_grad():
                    feat, mode = self._rosie_forward_spatial(x_rosie)
                if feat is not None and mode == "spatial":
                    use_spatial = True
                    if self.freeze_rosie:
                        feat = feat.detach()
                    m = self._rosie_to_marker(feat)  # [B, K, h, w]
                    s = torch.sigmoid(self._marker_to_prior(m))  # [B, 1, h, w]
                    s = F.interpolate(s, size=(rescale_value, rescale_value), mode="bilinear", align_corners=False)
                    self._last_np_prior_s = s
                elif not getattr(self, "_np_prior_fallback_warned", False):
                    logger.warning(
                        "[NP-Prior] ROSIE spatial features unavailable; falling back to classifier_broadcast."
                    )
                    self._np_prior_fallback_warned = True
            if not use_spatial:
                rosie_ctx = torch.no_grad() if self.freeze_rosie else torch.enable_grad()
                if rosie_features is None:
                    with rosie_ctx:
                        rosie_vec = self.rosie_model(x_rosie)
                    if self.rosie_subset_indices is not None:
                        rosie_vec = rosie_vec[:, self.rosie_subset_indices]
                    rosie_features_for_prior = rosie_vec
                else:
                    rosie_features_for_prior = rosie_features
                K_prior = rosie_features_for_prior.shape[1]
                m = rosie_features_for_prior.view(bs, K_prior, 1, 1).expand(bs, K_prior, rescale_value, rescale_value)
                s = torch.sigmoid(self._marker_to_prior(m))  # [B, 1, H, W]
                self._last_np_prior_s = s

        classifier_logits, _, z = self.encoder(x_rescaled)
        z0, z1, z2, z3, z4 = x_rescaled, *z

        patch_dim = [int(d / 14) for d in [x_rescaled.shape[-2], x_rescaled.shape[-1]]]

        # tokens -> BCHW
        z4 = z4[:, 1:, :].transpose(-1, -2).view(bs, self.embed_dim, *patch_dim)
        z3 = z3[:, 1:, :].transpose(-1, -2).view(bs, self.embed_dim, *patch_dim)
        z2 = z2[:, 1:, :].transpose(-1, -2).view(bs, self.embed_dim, *patch_dim)
        z1 = z1[:, 1:, :].transpose(-1, -2).view(bs, self.embed_dim, *patch_dim)

        if self.debug_print_z_shapes and not self._printed_z_shapes:
            print("Virchow z1,z2,z3,z4:", z1.shape, z2.shape, z3.shape, z4.shape)
            self._printed_z_shapes = True

        # Apply FiLM selectively (propagate film_force_identity / film_force_identity_train for ablations)
        if rosie_features is not None:
            film_force_id = getattr(self, "film_force_identity", False)
            film_force_id_train = getattr(self, "film_force_identity_train", False)
            for block in self.film_blocks.values():
                block.film_force_identity = film_force_id
                block.film_force_identity_train = film_force_id_train
            if "z1" in self.film_blocks: z1 = self.film_blocks["z1"](z1, rosie_features)
            if "z2" in self.film_blocks: z2 = self.film_blocks["z2"](z2, rosie_features)
            if "z3" in self.film_blocks: z3 = self.film_blocks["z3"](z3, rosie_features)
            if "z4" in self.film_blocks: z4 = self.film_blocks["z4"](z4, rosie_features)

        out_dict = {"tissue_types": classifier_logits}
        out_dict["nuclei_binary_map"] = self._forward_upsample(z0, z1, z2, z3, z4, self.nuclei_binary_map_decoder, input_shape)
        out_dict["hv_map"] = self._forward_upsample(z0, z1, z2, z3, z4, self.hv_map_decoder, input_shape)
        out_dict["nuclei_type_map"] = self._forward_upsample(z0, z1, z2, z3, z4, self.nuclei_type_maps_decoder, input_shape)

        if retrieve_tokens:
            out_dict["tokens"] = z4
        return out_dict

    def get_film_stats_and_reset(self):
        """Return per-layer gamma/beta stats if log_film_stats was enabled. Resets accumulators."""
        out = {}
        for name, block in self.film_blocks.items():
            if hasattr(block, "get_film_stats_and_reset"):
                s = block.get_film_stats_and_reset()
                if s:
                    out[name] = s
        return out
