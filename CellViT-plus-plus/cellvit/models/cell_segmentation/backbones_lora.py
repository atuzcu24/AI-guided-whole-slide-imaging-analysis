from typing import Callable, List, Tuple, Type
from einops import rearrange
import torch
import torch.nn as nn
import math
from cellvit.models.base.vision_transformer import VisionTransformer
from cellvit.models.utils.sam_utils import ImageEncoderViT
from cellvit.models.utils.uni_utils import TimmVisionTransformer
from cellvit.models.utils.virchow_utils import SwiGLUPacked
import torch.nn.functional as F 

# -------------------------------------------------------------------------
# LoRA Linear Layer
# -------------------------------------------------------------------------
class LoRALinear(nn.Linear):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        bias: bool = True, 
        device=None, 
        dtype=None,
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.
    ):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.r = r
        self.lora_alpha = lora_alpha
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        
        # 冻结原始权重
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'lora_A'):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = F.linear(x, self.weight, self.bias)
        if self.r > 0:
            result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
        return result


class ViTCellViTDeitL(ImageEncoderViT):
    def __init__(
        self,
        extract_layers: List[int],
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
        # LoRA 参数
        use_lora: bool = False,
        lora_r: int = 4,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        **kwargs,
    ) -> None:
        # ✅ 正确调用 SAM ImageEncoderViT 的 init
        # 这里的参数列表必须严格对应 segment_anything 的定义
        super().__init__(
            img_size,
            patch_size,
            in_chans,
            embed_dim,
            depth,
            num_heads,
            mlp_ratio,
            out_chans,
            qkv_bias,
            norm_layer,
            act_layer,
            use_abs_pos,
            use_rel_pos,
            rel_pos_zero_init,
            window_size,
            global_attn_indexes,
        )
        self.extract_layers = extract_layers
        self.use_lora = use_lora
        
        # LoRA 注入逻辑
        if self.use_lora:
            print(f"🔥 Injecting LoRA (r={lora_r}) into Backbone...")
            for i, blk in enumerate(self.blocks):
                # MLP 替换
                if hasattr(blk.mlp, "fc1"): fc1, fc2 = "fc1", "fc2"
                elif hasattr(blk.mlp, "lin1"): fc1, fc2 = "lin1", "lin2"
                else: continue
                
                # 替换 fc1
                old_fc1 = getattr(blk.mlp, fc1)
                new_fc1 = LoRALinear(old_fc1.in_features, old_fc1.out_features, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
                new_fc1.weight = old_fc1.weight
                if old_fc1.bias is not None: new_fc1.bias = old_fc1.bias
                setattr(blk.mlp, fc1, new_fc1)

                # 替换 fc2
                old_fc2 = getattr(blk.mlp, fc2)
                new_fc2 = LoRALinear(old_fc2.in_features, old_fc2.out_features, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
                new_fc2.weight = old_fc2.weight
                if old_fc2.bias is not None: new_fc2.bias = old_fc2.bias
                setattr(blk.mlp, fc2, new_fc2)

                # Attention (QKV) 替换
                qkv_name = "qkv_proj" if hasattr(blk.attn, "qkv_proj") else "qkv"
                if hasattr(blk.attn, qkv_name):
                    old_qkv = getattr(blk.attn, qkv_name)
                    new_qkv = LoRALinear(old_qkv.in_features, old_qkv.out_features, bias=old_qkv.bias is not None, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
                    new_qkv.weight = old_qkv.weight
                    if old_qkv.bias is not None: new_qkv.bias = old_qkv.bias
                    setattr(blk.attn, qkv_name, new_qkv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        extracted_layers = []
        x = self.patch_embed(x)

        if self.pos_embed is not None:
            token_size = x.shape[1]
            x = x + self.pos_embed[:, :token_size, :token_size, :]

        for depth, blk in enumerate(self.blocks):
            x = blk(x)
            if depth + 1 in self.extract_layers:
                extracted_layers.append(x)
        output = self.neck(x.permute(0, 3, 1, 2))
        _output = rearrange(output, "b c h w -> b c (h w)")

        return torch.mean(_output, axis=-1), output, extracted_layers


class ViTCellViTDeit(ImageEncoderViT):
    """For a parameter description see ViTCellViT"""

    def __init__(
        self,
        extract_layers: List[int],
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
    ) -> None:
        super().__init__(
            img_size,
            patch_size,
            in_chans,
            embed_dim,
            depth,
            num_heads,
            mlp_ratio,
            out_chans,
            qkv_bias,
            norm_layer,
            act_layer,
            use_abs_pos,
            use_rel_pos,
            rel_pos_zero_init,
            window_size,
            global_attn_indexes,
        )
        self.extract_layers = extract_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        extracted_layers = []
        x = self.patch_embed(x)

        if self.pos_embed is not None:
            token_size = x.shape[1]
            x = x + self.pos_embed[:, :token_size, :token_size, :]

        for depth, blk in enumerate(self.blocks):
            x = blk(x)
            if depth + 1 in self.extract_layers:
                extracted_layers.append(x)
        output = self.neck(x.permute(0, 3, 1, 2))
        _output = rearrange(output, "b c h w -> b c (h w)")

        return torch.mean(_output, axis=-1), output, extracted_layers


class ViTCellViTUNI(TimmVisionTransformer):
    """For a parameter description see ViTCellViT and TimmVisionTransformer"""

    def __init__(
        self,
        extract_layers: List[int],
        img_size: int = 224,
        patch_size: int = 16,
        depth: int = 24,
        num_heads: int = 16,
        embed_dim: int = 1024,
        num_classes: int = 0,
        init_values: float = 1e-5,
        dynamic_img_size: bool = True,
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            depth=depth,
            num_heads=num_heads,
            embed_dim=embed_dim,
            num_classes=0,
            init_values=init_values,
            dynamic_img_size=dynamic_img_size,
        )
        self.extract_layers = extract_layers
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        extracted_layers = []
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        for depth, blk in enumerate(self.blocks):
            x = blk(x)
            if depth + 1 in self.extract_layers:
                extracted_layers.append(x)

        # x = self.forward_head(x)
        output = self.head(x[:, 0])

        return output, x[:, 0], extracted_layers


class ViTCellViTVirchow(TimmVisionTransformer):
    """For a parameter description see ViTCellViT and TimmVisionTransformer"""

    def __init__(
        self,
        extract_layers: List[int],
        img_size: int = 224,
        patch_size: int = 14,
        depth: int = 32,
        num_heads: int = 16,
        embed_dim: int = 1280,
        num_classes: int = 0,
        init_values: float = 1e-5,
        dynamic_img_size: bool = True,
        reg_tokens: int = 0,
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            depth=depth,
            num_heads=num_heads,
            embed_dim=embed_dim,
            num_classes=0,
            init_values=init_values,
            dynamic_img_size=dynamic_img_size,
            global_pool="",
            act_layer=torch.nn.SiLU,
            mlp_layer=SwiGLUPacked,
            mlp_ratio=5.3375,
            reg_tokens=reg_tokens,
        )
        self.extract_layers = extract_layers
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        extracted_layers = []
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        for depth, blk in enumerate(self.blocks):
            x = blk(x)
            if depth + 1 in self.extract_layers:
                extracted_layers.append(x)

        # x = self.forward_head(x)
        output = self.head(x[:, 0])

        return output, x[:, 0], extracted_layers
