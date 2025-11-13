"""
05/11/25
Wrappers for Timm's modules.
Location of Timm's original modules: /usr/local/lib/python3.10/dist-packages/timm/models/vision_transformer.py

1. VitMask -
"""

from timm.models.vision_transformer import VisionTransformer
import torch
import torch.nn as nn
from typing import Optional
from timm.models._manipulate import checkpoint_seq


class VitMask(VisionTransformer):
    def forward_features(self, x: torch.Tensor, patch_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Original method: timm.models.vision_transformer.VisionTransformer.forward_features
        Here: add option to apply mask to patches before feeding to encoder.
        Args:
            x: Input tensor [B, 3, H, W]
            patch_mask: Optional tensor [B, num_patches] with 1=keep, 0=mask.
                        CLS/reg tokens are automatically kept.
        """
        # same as original:
        x = self.patch_embed(x)
        x = self._pos_embed(x)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Apply Mask (New) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if patch_mask is not None:
            patch_mask = patch_mask.to(dtype=x.dtype, device=x.device)

            # Always keep prefix tokens (CLS/reg)
            if self.num_prefix_tokens > 0:
                ones_prefix = torch.ones((x.shape[0], self.num_prefix_tokens), dtype=x.dtype, device=x.device)
                patch_mask = torch.cat([ones_prefix, patch_mask], dim=1)

            # Expand mask to embedding dimension
            patch_mask = patch_mask.unsqueeze(-1)  # [B, N, 1]

            # Apply mask
            x = x * patch_mask
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # Same as original:
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward(self, x: torch.Tensor, patch_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Define new forward (same as original in VisionTransformer, but uses the new forward_features method
        x = self.forward_features(x, patch_mask)
        x = self.forward_head(x)
        return x
