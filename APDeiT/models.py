# 7/1/26
# Copy model defenitions from APViT/Paddle/ppcls/arch/backbone/model_zoo/apvit.py for adding IRSEV2 backbone.
# Replace paddle.nn with torch.nn
# for now, use regular vision transformer for creating deit, if later on want to add distillation --> copy
#   DistilledVisionTransformer from deit/models.py
# Add register_model for APDeit: copy 'deit_base_patch16_224' from deit/models, and update it to be


# Imports copied from deit/models:
import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
from functools import partial


# Add register_model for APDeit: copy 'deit_base_patch16_224' from deit/models
@register_model
def deit_base_patch16_224(pretrained=False, **kwargs):
    # Filter out the 'pretrained_cfg' & 'pretrained_cfg_overlay' arguments:
    kwargs.pop('pretrained_cfg', None)
    kwargs.pop('pretrained_cfg_overlay', None)

    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

