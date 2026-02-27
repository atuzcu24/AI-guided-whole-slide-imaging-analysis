import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from cellvit.models.cell_segmentation.cellvit_sam import CellViTSAM
from cellvit.models.utils.rosie_markers import marker_names_to_indices


def _parse_subset_indices(subset_indices_str: str) -> list[int]:
    """Parse comma-separated subset indices string to list of ints (0-based)."""
    if not subset_indices_str or not subset_indices_str.strip():
        return []
    return [int(x.strip()) for x in subset_indices_str.split(",") if x.strip()]


class RosieFiLM(nn.Module):
    """
    Generic FiLM block:
      - input feature map: z  [B, H, W, C]
      - conditioning vec:   r [B, rosie_dim]
      - outputs:            z * gamma + beta (gamma,beta broadcast over H,W)
    Optional: identity init, gated/residual, clamp, scale.
    """
    def __init__(
        self,
        rosie_dim: int,
        feat_dim: int,
        hidden_dim: int = 256,
        film_init: str = "default",
        film_mode: str = "full",
        film_use_gating: bool = False,
        film_gating_init: float = 0.0,
        film_gating_mode: str = "scalar",
        film_scale: float = 1.0,
        film_clamp_gamma: float | None = None,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.film_mode = (film_mode or "full").lower()
        self.film_use_gating = film_use_gating
        self.film_gating_mode = film_gating_mode
        self.film_scale = film_scale
        self.film_clamp_gamma = film_clamp_gamma

        self.mlp = nn.Sequential(
            nn.Linear(rosie_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feat_dim * 2),
        )

        if film_use_gating:
            if film_gating_mode == "scalar":
                self.gate = nn.Parameter(torch.tensor(film_gating_init, dtype=torch.float32))
            else:
                self.gate = nn.Parameter(torch.full((feat_dim,), film_gating_init, dtype=torch.float32))

        if film_init == "identity":
            self._init_identity()

        # FiLM stats accumulation (optional, set from model when log_film_stats=True)
        self.log_film_stats = False
        self._film_stats_sum_abs_gm1 = 0.0
        self._film_stats_sum_gamma = 0.0
        self._film_stats_sum_gamma_sq = 0.0
        self._film_stats_sum_abs_beta = 0.0
        self._film_stats_sum_beta = 0.0
        self._film_stats_sum_beta_sq = 0.0
        self._film_stats_numel = 0

    def get_film_stats(self) -> dict | None:
        """Return gamma/beta stats if accumulated, else None."""
        n = self._film_stats_numel
        if n <= 0:
            return None
        mean_g = self._film_stats_sum_gamma / n
        var_g = (self._film_stats_sum_gamma_sq / n) - (mean_g ** 2)
        std_g = (var_g ** 0.5) if var_g > 0 else 0.0
        mean_b = self._film_stats_sum_beta / n
        var_b = (self._film_stats_sum_beta_sq / n) - (mean_b ** 2)
        std_b = (var_b ** 0.5) if var_b > 0 else 0.0
        return {
            "mean_abs_gamma_minus_1": self._film_stats_sum_abs_gm1 / n,
            "std_gamma": std_g,
            "mean_abs_beta": self._film_stats_sum_abs_beta / n,
            "std_beta": std_b,
        }

    def _init_identity(self) -> None:
        """Initialize last Linear so gamma=1, beta=0 at start."""
        last_linear = self.mlp[2]
        nn.init.zeros_(last_linear.weight)
        with torch.no_grad():
            last_linear.bias[: self.feat_dim] = 1.0
            last_linear.bias[self.feat_dim :] = 0.0

    def forward(self, z: torch.Tensor, rosie_features: torch.Tensor) -> torch.Tensor:
        film = self.mlp(rosie_features)          # [B, 2C]
        gamma, beta = film.chunk(2, dim=-1)      # [B, C], [B, C]

        # Inference-only: force identity (gamma=1, beta=0)
        if (not self.training) and getattr(self, "film_force_identity", False):
            gamma = torch.ones_like(gamma)
            beta = torch.zeros_like(beta)

        if self.film_mode == "beta_only":
            gamma = torch.ones_like(gamma)

        if getattr(self, "log_film_stats", False):
            n = gamma.numel()
            if n > 0:
                self._film_stats_sum_abs_gm1 += (gamma - 1.0).abs().sum().item()
                self._film_stats_sum_gamma += gamma.sum().item()
                self._film_stats_sum_gamma_sq += (gamma ** 2).sum().item()
                self._film_stats_sum_abs_beta += beta.abs().sum().item()
                self._film_stats_sum_beta += beta.sum().item()
                self._film_stats_sum_beta_sq += (beta ** 2).sum().item()
                self._film_stats_numel += n

        if self.film_clamp_gamma is not None:
            gamma_delta = torch.clamp(gamma - 1.0, -self.film_clamp_gamma, self.film_clamp_gamma)
            gamma = 1.0 + gamma_delta

        gamma = gamma.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, C]
        beta = beta.unsqueeze(1).unsqueeze(1)    # [B, 1, 1, C]

        z_mod = z * gamma + beta

        if self.film_use_gating:
            delta = self.film_scale * (z_mod - z)
            gate_val = self.gate
            if self.film_gating_mode == "scalar":
                gate_val = gate_val.view(1, 1, 1, 1)
            else:
                gate_val = gate_val.view(1, 1, 1, -1)
            return z + gate_val * delta
        return z_mod


class CellViTSAMRosieFiLM(CellViTSAM):
    """
    CellViT-SAM with optional Rosie-based FiLM fusion.

    - Rosie branch: frozen ConvNeXt-small -> rosie_head_dim vector (default 50)
    - FiLM: config-controlled modulation on selected encoder features (z1-z4)
    - Baseline behavior: film_enabled=False makes the model equivalent to CellViTSAM
      (except Rosie is computed, but doesn't affect output).
    """

    def __init__(
        self,
        model_path: str,
        num_nuclei_classes: int,
        num_tissue_classes: int,
        vit_structure: str = "sam-h",
        drop_rate: float = 0.0,
        regression_loss: bool = False,
        rosie_hidden_dim: int = 256,
        freeze_cellvit: bool = True,
        freeze_rosie: bool = True,
        rosie_weights_path: str | None = None,
        film_enabled: bool = True,
        film_layers: tuple[str, ...] = ("z4",),
        film_feat_dims: dict[str, int] | None = None,
        film_init: str = "default",
        film_mode: str = "full",
        film_use_gating: bool = False,
        film_gating_init: float = 0.0,
        film_gating_mode: str = "scalar",
        film_scale: float = 1.0,
        film_clamp_gamma: float | None = None,
        unfreeze_cellvit_epoch: int = 0,
        unfreeze_last_n_blocks: int | None = None,
        unfreeze_full_encoder: bool = False,
        debug_print_z_shapes: bool = False,
        rosie_marker_subset: list[str] | None = None,
        rosie_marker_subset_indices: list[int] | None = None,
        conditioning_mode_train: str = "normal",
        conditioning_subset_indices: list[int] | None = None,
        conditioning_dropout: dict | None = None,
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
        self.unfreeze_cellvit_epoch = unfreeze_cellvit_epoch
        # None/missing → 0 (no unfreeze). Full unfreeze requires unfreeze_full_encoder=True.
        self.unfreeze_last_n_blocks = 0 if unfreeze_last_n_blocks is None else unfreeze_last_n_blocks
        self.unfreeze_full_encoder = unfreeze_full_encoder
        self.debug_print_z_shapes = debug_print_z_shapes
        self._printed_z_shapes = False

        # Marker subset: if provided, FiLM uses only selected ROSIE channels
        self.rosie_marker_indices = marker_names_to_indices(
            rosie_marker_subset, rosie_marker_subset_indices
        )
        self.rosie_dim_for_film = len(self.rosie_marker_indices)

        # --------------------
        # ROSIE backbone
        # --------------------
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
                state = (ckpt.get("state_dict") or
                         ckpt.get("model_state_dict") or
                         ckpt.get("model") or
                         ckpt)

            cleaned = {}
            for k, v in state.items():
                nk = k
                for pref in ("module.", "model.", "net.", "backbone."):
                    if nk.startswith(pref):
                        nk = nk[len(pref):]
                cleaned[nk] = v

            missing, unexpected = self.rosie_model.load_state_dict(cleaned, strict=False)
            print(f"✅ Loaded ROSIE weights from: {rosie_weights_path}")
            print(f"   missing keys: {len(missing)} | unexpected keys: {len(unexpected)}")

        if self.freeze_rosie:
            for p in self.rosie_model.parameters():
                p.requires_grad = False
            self.rosie_model.eval()

        # Freeze SAM encoder if requested
        if self.freeze_cellvit:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # ImageNet normalization for ROSIE branch
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("rosie_mean", imagenet_mean, persistent=False)
        self.register_buffer("rosie_std", imagenet_std, persistent=False)

        # --------------------
        # FiLM configuration
        # --------------------
        self.film_enabled = film_enabled
        self.film_layers = set([l.lower() for l in (film_layers or tuple())]) if film_enabled else set()

        self.film_feat_dims = {k.lower(): v for k, v in (film_feat_dims or {}).items()}

        # Per-marker gate: gates ROSIE features before FiLM (input-level, not delta-level).
        # When used, RosieFiLM blocks get film_use_gating=False.
        self.film_gating_mode = film_gating_mode
        self.marker_gate_mlp = None
        if film_enabled and film_use_gating and film_gating_mode == "per_marker":
            self.marker_gate_mlp = nn.Sequential(
                nn.Linear(self.rosie_dim_for_film, self.rosie_dim_for_film),
                nn.Sigmoid(),
            )
            # Init bias so sigmoid starts near 1 (most markers active)
            with torch.no_grad():
                self.marker_gate_mlp[0].bias.fill_(2.0)

        film_use_gating_blocks = film_use_gating and film_gating_mode != "per_marker"
        film_gating_mode_blocks = film_gating_mode if film_gating_mode != "per_marker" else "scalar"

        self.film_blocks = nn.ModuleDict()
        for layer in ("z1", "z2", "z3", "z4"):
            if layer in self.film_layers:
                if layer not in self.film_feat_dims:
                    raise ValueError(
                        f"film_feat_dims['{layer}'] is required when FiLM is enabled on {layer}."
                    )
                self.film_blocks[layer] = RosieFiLM(
                    rosie_dim=self.rosie_dim_for_film,
                    feat_dim=self.film_feat_dims[layer],
                    hidden_dim=rosie_hidden_dim,
                    film_init=film_init,
                    film_mode=film_mode,
                    film_use_gating=film_use_gating_blocks,
                    film_gating_init=film_gating_init,
                    film_gating_mode=film_gating_mode_blocks,
                    film_scale=film_scale,
                    film_clamp_gamma=film_clamp_gamma,
                )
        self._encoder_unfrozen = False

        # Inference-time ablation (set by inference script; ignored if absent)
        self.conditioning_mode = getattr(self, "conditioning_mode", "normal")
        self.subset_indices = getattr(self, "subset_indices", "")
        self.log_film_stats = getattr(self, "log_film_stats", False)

        # Training-time conditioning override (from config fusion.conditioning_mode_train)
        self.conditioning_mode_train = conditioning_mode_train
        self.conditioning_subset_indices = conditioning_subset_indices or []
        self._conditioning_subset_set = set(self.conditioning_subset_indices)
        self._last_rosie_feat_mean: float | None = None
        self._last_rosie_feat_std: float | None = None

        # Per-batch stochastic conditioning dropout (overrides conditioning_mode_train when set)
        self.conditioning_dropout = None
        self._cond_dropout_probs: list[tuple[str, float]] = []  # [(mode, prob), ...]
        self._last_conditioning_mode: str | None = None
        self._cond_mode_counts: dict[str, int] = {}
        if conditioning_dropout is not None and isinstance(conditioning_dropout, dict):
            pn = float(conditioning_dropout.get("p_normal", 0))
            pz = float(conditioning_dropout.get("p_zeros", 0))
            ps = float(conditioning_dropout.get("p_shuffle", 0))
            total = pn + pz + ps
            if total <= 0:
                raise ValueError(
                    "conditioning_dropout: p_normal, p_zeros, p_shuffle must sum to > 0. "
                    f"Got p_normal={pn}, p_zeros={pz}, p_shuffle={ps}"
                )
            if abs(total - 1.0) > 1e-6:
                pn, pz, ps = pn / total, pz / total, ps / total
            self._cond_dropout_probs = [
                ("normal", pn),
                ("zeros", pz),
                ("shuffle", ps),
            ]
            self.conditioning_dropout = conditioning_dropout
            self._cond_mode_counts = {"normal": 0, "zeros": 0, "shuffle": 0}

        # Log fusion/unfreeze config at init
        _mode = "full" if unfreeze_full_encoder else ("last_n" if self.unfreeze_last_n_blocks > 0 else "none")
        print(
            f"[CellViTSAMRosieFiLM] freeze_cellvit={freeze_cellvit} | unfreeze_cellvit_epoch={unfreeze_cellvit_epoch} | "
            f"unfreeze_last_n_blocks={self.unfreeze_last_n_blocks} | unfreeze_full_encoder={unfreeze_full_encoder} | "
            f"unfreeze_mode={_mode}"
        )

    def _encoder_blocks_module_path(self) -> str:
        """Return the module path for the transformer blocks (for logging)."""
        return "cellvit.models.utils.sam_utils.ImageEncoderViT.blocks"

    def unfreeze_encoder_last_n_blocks(self, n: int) -> None:
        """
        Unfreeze only the last n transformer blocks of the SAM encoder.
        Keeps patch_embed, pos_embed, earlier blocks, and neck frozen.
        """
        if n <= 0:
            return
        blocks = getattr(self.encoder, "blocks", None)
        if blocks is None:
            print("[CellViTSAMRosieFiLM] WARNING: encoder has no .blocks, cannot partial unfreeze")
            return
        total = len(blocks)
        start_idx = max(0, total - n)
        unfrozen_indices = list(range(start_idx, total))
        for i in unfrozen_indices:
            for p in blocks[i].parameters():
                p.requires_grad = True
        n_trainable = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        print(
            f"[CellViTSAMRosieFiLM] unfreeze_mode=last_n | Unfroze last {n} blocks (indices {unfrozen_indices}) | "
            f"encoder trainable params: {n_trainable:,} | blocks path: {self._encoder_blocks_module_path()}"
        )

    def unfreeze_encoder(self) -> None:
        """
        Unfreeze encoder when epoch reaches unfreeze_cellvit_epoch.
        - If unfreeze_full_encoder: full unfreeze.
        - Elif unfreeze_last_n_blocks > 0: unfreeze only last N transformer blocks.
        - Else (missing/0): no unfreeze.
        """
        if self._encoder_unfrozen:
            return
        self._encoder_unfrozen = True
        if self.unfreeze_full_encoder:
            print("[CellViTSAMRosieFiLM] unfreeze_mode=full | Encoder fully unfrozen (unfreeze_full_encoder=true)")
            super().unfreeze_encoder()
        elif self.unfreeze_last_n_blocks > 0:
            print(f"[CellViTSAMRosieFiLM] unfreeze_mode=last_n | Unfreezing last {self.unfreeze_last_n_blocks} block(s)")
            self.unfreeze_encoder_last_n_blocks(self.unfreeze_last_n_blocks)
        else:
            print("[CellViTSAMRosieFiLM] unfreeze_mode=none | No unfreeze (unfreeze_last_n_blocks missing/0 and unfreeze_full_encoder=false)")

    def _rosie_preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert CellViT input (normalized to [-1,1]) into ImageNet-normalized [B,3,224,224]
        for ConvNeXt.
        """
        x_01 = (x + 1.0) / 2.0
        x_resized = F.interpolate(x_01, size=(224, 224), mode="bilinear", align_corners=False)
        return (x_resized - self.rosie_mean) / self.rosie_std

    def forward(self, x: torch.Tensor, retrieve_tokens: bool = False):
        assert x.shape[-2] % self.patch_size == 0, "Height must be divisible by patch_size"
        assert x.shape[-1] % self.patch_size == 0, "Width must be divisible by patch_size"

        out_dict = {}

        # 1) ROSIE features (frozen)
        with torch.no_grad():
            x_rosie = self._rosie_preprocess(x)
            rosie_full = self.rosie_model(x_rosie)  # [B, 50]
            # Select marker subset for FiLM conditioning
            rosie_features = rosie_full[:, self.rosie_marker_indices]  # [B, rosie_dim_for_film]

            # Conditioning ablation (inference-time)
            mode = getattr(self, "conditioning_mode", "normal")
            subset_str = getattr(self, "subset_indices", "")
            if mode == "zeros":
                rosie_features = torch.zeros_like(rosie_features)
            elif mode == "shuffle":
                B = rosie_features.shape[0]
                perm = torch.randperm(B, device=rosie_features.device)
                rosie_features = rosie_features[perm]
            elif mode == "subset9":
                indices = _parse_subset_indices(subset_str)
                if indices:
                    idx_set = set(indices)
                    mask = torch.ones_like(rosie_features)
                    for i in range(rosie_features.shape[1]):
                        if i not in idx_set:
                            mask[:, i] = 0
                    rosie_features = rosie_features * mask
            # normal: no change

            # Training-time conditioning: conditioning_dropout overrides conditioning_mode_train
            train_mode = getattr(self, "conditioning_mode_train", "normal")
            cond_dropout = getattr(self, "conditioning_dropout", None)
            cond_dropout_probs = getattr(self, "_cond_dropout_probs", [])

            if self.training and cond_dropout_probs:
                # Per-batch stochastic: sample one mode
                modes = [m for m, _ in cond_dropout_probs]
                probs = [p for _, p in cond_dropout_probs]
                sampled_mode = random.choices(modes, weights=probs, k=1)[0]
                train_mode = sampled_mode
                self._last_conditioning_mode = sampled_mode
                if hasattr(self, "_cond_mode_counts"):
                    self._cond_mode_counts[sampled_mode] = self._cond_mode_counts.get(sampled_mode, 0) + 1
            elif self.training and train_mode != "normal":
                self._last_conditioning_mode = train_mode

            if self.training and train_mode != "normal":
                if train_mode == "zeros":
                    rosie_features = torch.zeros_like(rosie_features)
                elif train_mode == "shuffle":
                    B = rosie_features.shape[0]
                    perm = torch.randperm(B, device=rosie_features.device)
                    rosie_features = rosie_features[perm]
                elif train_mode == "subset" and self._conditioning_subset_set:
                    mask = torch.ones_like(rosie_features)
                    for i in range(rosie_features.shape[1]):
                        if i not in self._conditioning_subset_set:
                            mask[:, i] = 0
                    rosie_features = rosie_features * mask
                # normal: no change

            # Per-marker gate: w = sigmoid(MLP(rosie)), rosie_gated = w * rosie
            self._last_marker_gate = None
            self._last_marker_gate_w = None
            self._last_rosie_feat_std_before_gate = None
            self._last_rosie_feat_std_after_gate = None
            if self.marker_gate_mlp is not None:
                with torch.no_grad():
                    self._last_rosie_feat_std_before_gate = float(rosie_features.std().item())
                w = self.marker_gate_mlp(rosie_features)
                rosie_features = w * rosie_features
                self._last_marker_gate = w
                self._last_marker_gate_w = w.detach()
                with torch.no_grad():
                    self._last_rosie_feat_std_after_gate = float(rosie_features.std().item())

            # Log rosie_features mean/std when train override or dropout is active (for verification)
            if self.training and train_mode != "normal":
                with torch.no_grad():
                    m = rosie_features.mean().item()
                    s = rosie_features.std().item()
                    self._last_rosie_feat_mean = float(m) if not (m != m) else 0.0  # nan guard
                    self._last_rosie_feat_std = float(s) if not (s != s) else 0.0  # nan guard

        # 2) CellViT-SAM encoder forward
        classifier_logits, _, z = self.encoder(x)
        out_dict["tissue_types"] = self.classifier_head(classifier_logits)

        z0 = x
        z1, z2, z3, z4 = z  # each: [B, H, W, C]

        if self.debug_print_z_shapes and (not self._printed_z_shapes):
            print("z1,z2,z3,z4:", z1.shape, z2.shape, z3.shape, z4.shape)
            self._printed_z_shapes = True

        # 3) Optional FiLM modulation (config-controlled)
        raw_z4 = z4.clone()  # for retrieve_tokens
        log_film = getattr(self, "log_film_stats", False)
        film_force_id = getattr(self, "film_force_identity", False)
        if self.film_enabled:
            for block in self.film_blocks.values():
                block.log_film_stats = log_film
                block.film_force_identity = film_force_id
            if "z1" in self.film_blocks: z1 = self.film_blocks["z1"](z1, rosie_features)
            if "z2" in self.film_blocks: z2 = self.film_blocks["z2"](z2, rosie_features)
            if "z3" in self.film_blocks: z3 = self.film_blocks["z3"](z3, rosie_features)
            if "z4" in self.film_blocks: z4 = self.film_blocks["z4"](z4, rosie_features)

        # 4) Decoders expect BCHW
        z4 = z4.permute(0, 3, 1, 2)
        z3 = z3.permute(0, 3, 1, 2)
        z2 = z2.permute(0, 3, 1, 2)
        z1 = z1.permute(0, 3, 1, 2)

        if self.regression_loss:
            nb_map = self._forward_upsample(z0, z1, z2, z3, z4, self.nuclei_binary_map_decoder)
            out_dict["nuclei_binary_map"] = nb_map[:, :2, :, :]
            out_dict["regression_map"] = nb_map[:, 2:, :, :]
        else:
            out_dict["nuclei_binary_map"] = self._forward_upsample(
                z0, z1, z2, z3, z4, self.nuclei_binary_map_decoder
            )

        out_dict["hv_map"] = self._forward_upsample(z0, z1, z2, z3, z4, self.hv_map_decoder)
        out_dict["nuclei_type_map"] = self._forward_upsample(
            z0, z1, z2, z3, z4, self.nuclei_type_maps_decoder
        )

        if retrieve_tokens:
            out_dict["tokens"] = raw_z4

        return out_dict

    def get_film_stats(self) -> dict:
        """Return per-layer FiLM gamma/beta stats from all film_blocks."""
        if not hasattr(self, "film_blocks") or not self.film_blocks:
            return {}
        stats = {}
        for name, block in self.film_blocks.items():
            if hasattr(block, "get_film_stats"):
                layer_stats = block.get_film_stats()
                if layer_stats is not None:
                    stats[name] = layer_stats
        return stats
