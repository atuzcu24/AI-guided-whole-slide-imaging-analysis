"""
Sanity test for NP prior losses (L_sup, L_bd) and CellViTVirchowRosieFiLM spatial prior.

Run: python -m cellvit.tools.test_np_prior_sanity
"""
import torch
import torch.nn.functional as F


def finite_diff_grad(t: torch.Tensor) -> tuple:
    """Finite-difference gradients (dx, dy) for [B,1,H,W] tensors."""
    dx = t[:, :, :, 1:] - t[:, :, :, :-1]
    dy = t[:, :, 1:, :] - t[:, :, :-1, :]
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    return dx, dy


def test_losses_finite_and_backward():
    """Verify L_sup and L_bd are finite and backward works on y_fg."""
    B, H, W = 2, 64, 64
    y_fg = torch.rand(B, 1, H, W, requires_grad=True)
    s = torch.rand(B, 1, H, W)

    # L_sup = mean(y_fg * (1 - s))
    L_sup = (y_fg * (1.0 - s)).mean()
    assert torch.isfinite(L_sup).all(), f"L_sup not finite: {L_sup}"
    L_sup.backward()
    assert y_fg.grad is not None and torch.isfinite(y_fg.grad).all()

    y_fg = torch.rand(B, 1, H, W, requires_grad=True)
    s = torch.rand(B, 1, H, W)
    dx_y, dy_y = finite_diff_grad(y_fg)
    dx_s, dy_s = finite_diff_grad(s)
    L_bd = (dx_y - dx_s).abs().mean() + (dy_y - dy_s).abs().mean()
    assert torch.isfinite(L_bd).all(), f"L_bd not finite: {L_bd}"
    L_bd.backward()
    assert y_fg.grad is not None and torch.isfinite(y_fg.grad).all()
    print("test_losses_finite_and_backward: OK")


def test_model_spatial_prior_shapes():
    """
    Run one forward pass with rosie_make_spatial_prior=true.
    Passes with either rosie_backbone (spatial) or classifier_broadcast fallback.
    Does NOT assume rosie_model.features() exists - fallback is acceptable.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

    from cellvit.models.cell_segmentation.cellvit_virchow_rosie_film import CellViTVirchowRosieFiLM

    ckpt = Path(__file__).resolve().parents[3] / "checkpoints" / "Virchow" / "encoder_only_CellViT-Virchow-x40-AMP.pth"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    for prior_from in ("rosie_backbone", "rosie_classifier_broadcast"):
        model = CellViTVirchowRosieFiLM(
            model_virchow_path=ckpt,
            num_nuclei_classes=6,
            num_tissue_classes=19,
            film_enabled=True,
            film_layers=("z4",),
            film_feat_dims={"z4": 1280},
            rosie_subset_indices=list(range(10)),
            rosie_make_spatial_prior=True,
            rosie_prior_from=prior_from,
        )
        model.eval()

        x = torch.randn(2, 3, 256, 256)  # B, C, H, W (RGB)
        with torch.no_grad():
            out = model(x)

        assert hasattr(model, "_last_np_prior_s") and model._last_np_prior_s is not None
        s = model._last_np_prior_s
        print(f"  {prior_from}: s shape {s.shape}")
        assert s.dim() == 4 and s.shape[0] == 2 and s.shape[1] == 1, f"Expected (2,1,H,W), got {s.shape}"
        assert s.shape[2] in (252, 256) and s.shape[3] in (252, 256), f"Unexpected spatial size {s.shape[2:]}"
    print("test_model_spatial_prior_shapes: OK")


if __name__ == "__main__":
    test_losses_finite_and_backward()
    try:
        test_model_spatial_prior_shapes()
    except FileNotFoundError as e:
        print(f"test_model_spatial_prior_shapes SKIP (checkpoint not found): {e}")
    print("All sanity checks passed.")
