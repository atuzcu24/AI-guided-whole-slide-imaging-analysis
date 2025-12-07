import torch
import torch.nn as nn
import torch.nn.functional as F

from cellvit.models.cell_segmentation.cellvit_virchow import CellViTVirchow


class VirchowFiLM(nn.Module):
    """
    FiLM modulation for Virchow encoder output z4.
    rosie_features: [B, D]
    z4: [B, H, W, C] -> permuted to [B, C, H, W]
    """

    def __init__(self, rosie_dim=50, feat_dim=256, hidden_dim=256):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(rosie_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feat_dim * 2),
        )

    def forward(self, z4, rosie_feats):
        # z4: [B, C, H, W]
        B, C, H, W = z4.shape

        gamma_beta = self.mlp(rosie_feats)  # [B, 2*C]
        gamma, beta = gamma_beta[:, :C], gamma_beta[:, C:]

        gamma = gamma.view(B, C, 1, 1)
        beta = beta.view(B, C, 1, 1)

        return z4 * gamma + beta
        

class CellViTVirchowRosieFiLM(CellViTVirchow):
    """
    Virchow + FiLM applied on the z4 feature map before decoder.
    """

    def __init__(self, model_virchow_path, num_nuclei_classes, num_tissue_classes,
                 rosie_dim=50, film_hidden_dim=256, **kwargs):
        
        super().__init__(
            model_virchow_path=model_virchow_path,
            num_nuclei_classes=num_nuclei_classes,
            num_tissue_classes=num_tissue_classes,
        )

        # FiLM module over z4 (Virchow has out_channels=256)
        self.film = VirchowFiLM(
            rosie_dim=rosie_dim,
            feat_dim=self.embed_dim,
            hidden_dim=film_hidden_dim,
        )

    def forward(self, x: torch.Tensor, rosie_features=None, retrieve_tokens=False):
        # --- Run Virchow encoder ---
        bs = x.shape[0]
        input_shape = x.shape[2]
        rescale_value = self.input_rescale_dict[input_shape]

        # Virchow expects rescaled input
        x_rescaled = F.interpolate(x, size=(rescale_value, rescale_value), mode="area")

        classifier_logits, _, z = self.encoder(x_rescaled)
        
        # z = [z1, z2, z3, z4]
        z0, z1, z2, z3, z4 = x_rescaled, *z

        # reshape patches → spatial maps
        patch_dim = [int(d / 14) for d in [x_rescaled.shape[-2], x_rescaled.shape[-1]]]

        z4 = z4[:, 1:, :].transpose(-1, -2).view(-1, self.embed_dim, *patch_dim)
        z3 = z3[:, 1:, :].transpose(-1, -2).view(-1, self.embed_dim, *patch_dim)
        z2 = z2[:, 1:, :].transpose(-1, -2).view(-1, self.embed_dim, *patch_dim)
        z1 = z1[:, 1:, :].transpose(-1, -2).view(-1, self.embed_dim, *patch_dim)

        # --- FiLM only modifies z4 ---
        if rosie_features is not None:
            z4 = self.film(z4, rosie_features)

        out_dict = {}
        out_dict["tissue_types"] = classifier_logits

        # --- Now decode using FiLM-modulated z4 ---
        out_dict["nuclei_binary_map"] = self._forward_upsample(
            z0, z1, z2, z3, z4, self.nuclei_binary_map_decoder, input_shape
        )
        out_dict["hv_map"] = self._forward_upsample(
            z0, z1, z2, z3, z4, self.hv_map_decoder, input_shape
        )
        out_dict["nuclei_type_map"] = self._forward_upsample(
            z0, z1, z2, z3, z4, self.nuclei_type_maps_decoder, input_shape
        )

        if retrieve_tokens:
            out_dict["tokens"] = z4

        return out_dict

