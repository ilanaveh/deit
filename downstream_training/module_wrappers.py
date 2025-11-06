"""
05/11/25
Wrappers for Timm's modules.
Location of Timm's original modules: /usr/local/lib/python3.10/dist-packages/timm/models/vision_transformer.py

1. VitMasks -
"""

from timm.models.vision_transformer import VisionTransformer
import torch
import torch.nn as nn
from typing import Optional
from timm.models._manipulate import checkpoint_seq


class VitMasks(VisionTransformer):
    def forward_features(self, x: torch.Tensor, patch_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Original method: timm.models.vision_transformer.VisionTransformer.forward_features
        Here: add option to apply mask to patches before feeding to encoder.
        Args:
            x: Input tensor [B, 3, H, W]
            patch_mask: Optional tensor [B, num_patches] with 1=keep, 0=mask.
                        CLS/reg tokens are automatically kept.
        """
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        x = self.norm(x)
        return x
