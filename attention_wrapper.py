"""
20/5/25
A wrapper for Timm's Attention module, with a modified forward method, that enables access to the attention-maps.
Location of Timm's original Attention class: /usr/local/lib/python3.10/dist-packages/timm/models/vision_transformer.py
"""

from timm.models.vision_transformer import Attention
import torch
import torch.nn.functional as F


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

            self.last_attn = attn  # IN: Add this line, to enable access to Attention Map.

            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
