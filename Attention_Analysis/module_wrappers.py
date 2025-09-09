"""
20/5/25
Wrappers for Timm's modules.
Location of Timm's original modules: /usr/local/lib/python3.10/dist-packages/timm/models/vision_transformer.py

1. Attention module: a modified forward method, that enables access to the attention-maps (used for attention analysis).

2. ViTWithFeatures: a modified forward method, that enables access to features (used for lora pair-loss)
"""

from timm.models.vision_transformer import Attention
import torch
import torch.nn.functional as F
import copy
import torch.nn as nn


class AttentionWithAttnMap(Attention):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)

            self.last_attn = copy.deepcopy(attn)  # IN: Add this line, to enable access to Attention Map.

            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ViTWithFeatures(nn.Module):
    """
    Wrapper for timm VisionTransformer that allows returning both
    classification logits and the pooled features (CLS or avg pooled).
    """
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base = base_model

    def forward(self, x: torch.Tensor, return_features: bool = False):
        # Run backbone up to normalized token sequence
        x = self.base.forward_features(x)

        # Pool (CLS token by default, or avg depending on config)
        feats = self.base.pool(x)

        # Apply fc norm + dropout
        feats = self.base.fc_norm(feats)
        logits = self.base.head_drop(feats)
        logits = self.base.head(logits)

        if return_features:
            return logits, feats
        return logits
