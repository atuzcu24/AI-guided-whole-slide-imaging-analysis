# -*- coding: utf-8 -*-
"""
CellViT-SAM with proxy-conditioned FiLM.

Extracts Sobel magnitude and Hematoxylin channel from RGB input inside forward,
builds conditioning vector c = [mean(hema), std(hema), mean(sobel), std(sobel)],
and FiLM-modulates encoder z4. No dataset changes; input stays RGB (3 channels).
"""

from pathlib import Path
from typing import List, Optional, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from cellvit.models.cell_segmentation.cellvit_sam import CellViTSAM
from cellvit.models.cell_segmentation.cellvit_sam_rosie_film import RosieFiLM


# Ruifrok & Johnston H&E stain matrix (columns: H, E; rows: R,G,B)
STAIN_MATRIX_HE = torch.tensor(
    [
        [0.650, 0.072],
        [0.704, 0.990],
        [0.286, 0.105],
    ],
    dtype=torch.float32,
)


def _stain_deconv_hematoxylin_torch(rgb: torch.Tensor) -> torch.Tensor:
    """Extract Hematoxylin channel via Ruifrok & Johnston. rgb: [B,3,H,W] in [0,1]."""
    assert rgb.dim() == 4 and rgb.shape[1] == 3, f"Expected rgb [B,3,H,W], got {rgb.shape}"
    eps = 1e-8
    rgb_clamp = torch.clamp(rgb, min=1e-6, max=1.0)
    od = -torch.log(rgb_clamp)
    od_flat = od.permute(0, 2, 3, 1).reshape(-1, 3)
    W_he = STAIN_MATRIX_HE.to(rgb.device)
    v1, v2 = W_he[:, 0], W_he[:, 1]
    v3 = torch.linalg.cross(
        v1.unsqueeze(0),
        v2.unsqueeze(0),
    ).squeeze(0)
    v3 = v3 / (torch.linalg.norm(v3) + eps)
    W = torch.column_stack([v1, v2, v3])
    Q = torch.linalg.pinv(W)
    conc = od_flat @ Q.T

    B_, _, H_, W_ = rgb.shape
    B_, H_, W_ = int(B_), int(H_), int(W_)
    hema = torch.nan_to_num(conc[:, 0].contiguous().reshape(B_, H_, W_), nan=0.0, posinf=0.0, neginf=0.0)
    return hema


class CellViTSAMProxyFiLM(CellViTSAM):
    """
    CellViT-SAM with proxy-conditioned FiLM (no ROSIE, no dataset changes).

    - Input: standard RGB [B,3,H,W]
    - Proxy extraction inside forward: Sobel magnitude + Hematoxylin (H&E deconv)
    - Conditioning vector c = [mean(hema), std(hema), mean(sobel), std(sobel)] per sample
    - FiLM modulates z4 via gamma*z4 + beta from MLP(c)
    """

    def __init__(
        self,
        model_path: Union[Path, str],
        num_nuclei_classes: int,
        num_tissue_classes: int,
        vit_structure: Literal["SAM-B", "SAM-L", "SAM-H"] = "SAM-H",
        drop_rate: float = 0,
        regression_loss: bool = False,
        film_layers: tuple[str, ...] = ("z4",),
        film_feat_dims: dict[str, int] | None = None,
        film_init: str = "default",
        rosie_hidden_dim: int = 256,
        conditioning_mode_train: str = "normal",
        conditioning_mode_infer: str = "normal",
        normalize_mean: Optional[List[float]] = None,
        normalize_std: Optional[List[float]] = None,
    ):
        super().__init__(
            model_path=model_path,
            num_nuclei_classes=num_nuclei_classes,
            num_tissue_classes=num_tissue_classes,
            vit_structure=vit_structure,
            drop_rate=drop_rate,
            regression_loss=regression_loss,
            input_channels=3,
        )
        self.conditioning_mode_train = conditioning_mode_train
        self.conditioning_mode_infer = conditioning_mode_infer
        self.conditioning_mode = conditioning_mode_infer

        feat_dim_z4 = film_feat_dims.get("z4", 1280) if film_feat_dims else 1280
        self.film_layers = set(l.lower() for l in film_layers) if film_layers else set()
        self.film_feat_dims = dict(film_feat_dims or {"z4": feat_dim_z4})
        self.film_enabled = bool(self.film_layers)

        self.proxy_dim = 4
        self.film_blocks = nn.ModuleDict()
        if "z4" in self.film_layers:
            self.film_blocks["z4"] = RosieFiLM(
                rosie_dim=self.proxy_dim,
                feat_dim=self.film_feat_dims["z4"],
                hidden_dim=rosie_hidden_dim,
                film_init=film_init,
                film_use_gating=False,
                film_gating_init=0.0,
                film_gating_mode="scalar",
                film_scale=1.0,
                film_clamp_gamma=None,
            )

        self.log_film_stats = False
        self._film_stats_logged_once = False
        self._x_rgb_logged_once = False
        self._c_finite_checked_once = False

        sobel_x = torch.tensor(
            [[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer("_sobel_kernel_x", sobel_x.repeat(1, 1, 1, 1))
        self.register_buffer("_sobel_kernel_y", sobel_y.repeat(1, 1, 1, 1))

        _mean = normalize_mean if normalize_mean is not None else [0.5, 0.5, 0.5]
        _std = normalize_std if normalize_std is not None else [0.5, 0.5, 0.5]
        if len(_mean) == 1:
            _mean = _mean * 3
        if len(_std) == 1:
            _std = _std * 3
        self.register_buffer(
            "_norm_mean",
            torch.tensor(_mean, dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "_norm_std",
            torch.tensor(_std, dtype=torch.float32).view(1, 3, 1, 1),
        )

        print(
            f"[CellViTSAMProxyFiLM] film_enabled={self.film_enabled} | "
            f"film_layers={sorted(self.film_layers)} | "
            f"conditioning_mode_train={conditioning_mode_train} | infer={conditioning_mode_infer}"
        )

    def _extract_proxy_features(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B,3,H,W] Albumentations-normalized. De-normalize to [0,1] RGB for proxies."""
        eps = 1e-8
        mean = self._norm_mean.to(x.device)
        std = self._norm_std.to(x.device)
        rgb = (x * std + mean).clamp(1e-6, 1.0)

        if self.training and not self._x_rgb_logged_once:
            with torch.no_grad():
                self._x_rgb_logged_once = True
                mn, mx = rgb.min().item(), rgb.max().item()
                print(f"[CellViTSAMProxyFiLM] epoch1 x_rgb min={mn:.4f} max={mx:.4f}")

        B, C, H, W = x.shape

        gray = (
            0.2989 * rgb[:, 0:1, :, :]
            + 0.5870 * rgb[:, 1:2, :, :]
            + 0.1140 * rgb[:, 2:3, :, :]
        )
        pad = (1, 1, 1, 1)
        gray_pad = F.pad(gray, pad, mode="replicate")
        gx = F.conv2d(gray_pad, self._sobel_kernel_x, padding=0)
        gy = F.conv2d(gray_pad, self._sobel_kernel_y, padding=0)
        sq = (gx ** 2 + gy ** 2).clamp(min=1e-12)
        sobel = torch.sqrt(sq).squeeze(1)
        sobel = torch.nan_to_num(sobel, nan=0.0, posinf=0.0, neginf=0.0)

        hema = _stain_deconv_hematoxylin_torch(rgb)

        hema_flat = hema.view(B, -1)
        mn, mx = hema_flat.min(dim=1)[0], hema_flat.max(dim=1)[0]
        denom = (mx - mn + eps).clamp(min=1e-6)
        hema_norm = (hema - mn.view(B, 1, 1)) / denom.view(B, 1, 1)
        hema_norm = torch.clamp(hema_norm, 0.0, 1.0)
        hema_norm = torch.nan_to_num(hema_norm, nan=0.0, posinf=1.0, neginf=0.0)
        m_h = hema_norm.view(B, -1).mean(dim=1)
        s_h = hema_norm.view(B, -1).std(dim=1) + eps

        sobel_flat = sobel.view(B, -1)
        m_s = sobel_flat.mean(dim=1)
        s_s = sobel_flat.std(dim=1) + eps

        c = torch.stack([m_h, s_h, m_s, s_s], dim=1)
        assert c.shape == (B, 4), f"Expected c [B,4], got {c.shape}"

        if self.training and not self._c_finite_checked_once:
            self._c_finite_checked_once = True
            if not torch.isfinite(c).all():
                nbad = (~torch.isfinite(c)).sum().item()
                cfin = c[torch.isfinite(c)]
                cmin = cfin.min().item() if cfin.numel() > 0 else float("nan")
                cmax = cfin.max().item() if cfin.numel() > 0 else float("nan")
                print(
                    f"[CellViTSAMProxyFiLM] epoch1 c non-finite: count={nbad} "
                    f"min={cmin} max={cmax}"
                )
                assert torch.isfinite(c).all(), f"c has {nbad} non-finite values"
        c = torch.nan_to_num(c, nan=0.0, posinf=1.0, neginf=0.0)
        return c

    def forward(self, x: torch.Tensor, retrieve_tokens: bool = False):
        assert x.shape[-2] % self.patch_size == 0
        assert x.shape[-1] % self.patch_size == 0

        out_dict = {}

        c = self._extract_proxy_features(x)

        mode = self.conditioning_mode_train if self.training else self.conditioning_mode_infer
        if mode == "zeros":
            c = torch.zeros_like(c, device=c.device)
        elif mode == "shuffle":
            B = c.shape[0]
            perm = torch.randperm(B, device=c.device)
            c = c[perm]

        if not self.training and getattr(self, "_infer_debug_conditioning", False):
            self._infer_debug_conditioning = False
            s0 = c[0].sum().item() if c.shape[0] > 0 else float("nan")
            s1 = c[1].sum().item() if c.shape[0] > 1 else float("nan")
            print(
                f"[CellViTSAMProxyFiLM] inference conditioning_mode_infer={mode} "
                f"c[0].sum()={s0:.4f} c[1].sum()={s1:.4f}"
            )

        if self.training and not self._film_stats_logged_once:
            with torch.no_grad():
                self._film_stats_logged_once = True
                mh, sh = c[:, 0].mean().item(), c[:, 0].std().item()
                ms, ss = c[:, 2].mean().item(), c[:, 3].mean().item()
                print(
                    f"[CellViTSAMProxyFiLM] epoch1 c stats: "
                    f"hema_mean={mh:.4f} hema_std={sh:.4f} "
                    f"sobel_mean={ms:.4f} sobel_std={ss:.4f}"
                )

        classifier_logits, _, z = self.encoder(x)
        out_dict["tissue_types"] = self.classifier_head(classifier_logits)

        z0 = x
        z1, z2, z3, z4 = z

        log_film = getattr(self, "log_film_stats", False)
        film_force_id = getattr(self, "film_force_identity", False)
        if self.film_enabled:
            for block in self.film_blocks.values():
                block.log_film_stats = log_film
                block.film_force_identity = film_force_id
            if "z4" in self.film_blocks:
                z4 = self.film_blocks["z4"](z4, c)

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

        out_dict["hv_map"] = self._forward_upsample(z0, z1, z2, z3, z4, self.hv_map_decoder)
        out_dict["nuclei_type_map"] = self._forward_upsample(
            z0, z1, z2, z3, z4, self.nuclei_type_maps_decoder
        )

        if retrieve_tokens:
            out_dict["tokens"] = z4

        return out_dict

    def get_film_stats(self) -> dict:
        if not hasattr(self, "film_blocks") or not self.film_blocks:
            return {}
        stats = {}
        for name, block in self.film_blocks.items():
            if hasattr(block, "get_film_stats"):
                layer_stats = block.get_film_stats()
                if layer_stats is not None:
                    stats[name] = layer_stats
        return stats
