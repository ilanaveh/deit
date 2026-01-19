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

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Imports copied from APViT/Paddle/ppcls/arch/backbone/model_zoo/irse_v2.py, but replaced 'paddle' with 'torch':
from torch.nn import Linear
from torch.nn import Conv2d
from torch.nn import BatchNorm1d
from torch.nn import BatchNorm2d
from torch.nn import PReLU
from torch.nn import Sigmoid
from torch.nn import Dropout
from torch.nn import MaxPool2d
from torch.nn import AdaptiveAvgPool2d
from torch.nn import Sequential
from collections import namedtuple

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Modules from APViT/Paddle/ppcls/arch/backbone/model_zoo/irse_v2.py:
class Flatten(nn.Layer):

    def forward(self, input):
        return input.view(input.size(0), -1)


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = input / norm
    return output


class SEModule(nn.Layer):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(channels, channels // reduction, kernel_size=1,
            padding=0, bias_attr=False)
        # torch2paddle.xavier_normal_(self.fc1.weight.data)
        self.relu = nn.ReLU()
        self.fc2 = Conv2d(channels // reduction, channels, kernel_size=1,
            padding=0, bias_attr=False)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class bottleneck_IR(nn.Layer):

    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(Conv2d(in_channel, depth, (1,
                1), stride, bias_attr=False), BatchNorm2d(depth))
        self.res_layer = Sequential(BatchNorm2d(in_channel), Conv2d(
            in_channel, depth, (3, 3), (1, 1), 1, bias_attr=False), PReLU(
            depth), Conv2d(depth, depth, (3, 3), stride, 1, bias_attr=False
            ), BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class bottleneck_IR_SE(nn.Layer):

    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(Conv2d(in_channel, depth, (1,
                1), stride, bias_attr=False), BatchNorm2d(depth))
        self.res_layer = Sequential(BatchNorm2d(in_channel), Conv2d(
            in_channel, depth, (3, 3), (1, 1), 1, bias_attr=False), PReLU(
            depth), Conv2d(depth, depth, (3, 3), stride, 1, bias_attr=False
            ), BatchNorm2d(depth), SEModule(depth, 16))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    """A named tuple describing a ResNet block."""


def get_block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth,
        depth, 1) for i in range(num_units - 1)]


def get_blocks(num_layers):
    if num_layers == 8:
        blocks = [get_block(in_channel=64, depth=64, num_units=3)]
    elif num_layers == 16:
        blocks = [get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4)]
    elif num_layers == 34:
        blocks = [get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4), get_block(
            in_channel=128, depth=256, num_units=9)]
    elif num_layers == 44:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14)]
    elif num_layers == 50:
        blocks = [get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4), get_block(
            in_channel=128, depth=256, num_units=14), get_block(in_channel=\
            256, depth=512, num_units=3)]
    elif num_layers == 100:
        blocks = [get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13), get_block(
            in_channel=128, depth=256, num_units=30), get_block(in_channel=\
            256, depth=512, num_units=3)]
    elif num_layers == 152:
        blocks = [get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8), get_block(
            in_channel=128, depth=256, num_units=36), get_block(in_channel=\
            256, depth=512, num_units=3)]
    return blocks


class IRSEV2(nn.Layer):

    def __init__(self, input_size, num_layers, mode='ir', with_head=False,
        pretrained=None, return_index=(2,)):
        super().__init__()
        assert input_size[0] in [112, 224
            ], 'input_size should be [112, 112] or [224, 224]'
        assert num_layers in [0, 8, 16, 34, 44, 50, 100, 152
            ], 'num_layers should be 50, 100 or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        self.num_layers = num_layers
        self.return_index = return_index
        if num_layers == 0:
            return
        self.with_head = with_head
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias_attr
            =False), BatchNorm2d(64), PReLU(64))
        if with_head:
            if input_size[0] == 112:
                self.output_layer = Sequential(BatchNorm2d(512), Dropout(),
                    Flatten(), Linear(512 * 7 * 7, 512), BatchNorm1d(512))
            else:
                self.output_layer = Sequential(BatchNorm2d(512), Dropout(),
                    Flatten(), Linear(512 * 14 * 14, 512), BatchNorm1d(512))
        modules = []
        max_stage = max(return_index)
        for block in blocks[:max_stage+1]:
            block_module = []
            for bottleneck in block:
                block_module.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))
            modules.append(Sequential(*block_module))
        self.body = nn.LayerList(modules)

        if pretrained:
            self.init_weights(pretrained)

    def forward(self, x):
        if self.num_layers == 0:
            return x
        x = self.input_layer(x)
        output = []
        return_index = set(self.return_index)
        for index, m in enumerate(self.body):
            x = m(x)
            if index in return_index:
                output.append(x)
        if self.with_head:
            x = self.output_layer(x)

        return output

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# Modules from APViT/Paddle/ppcls/arch/backbone/model_zoo/apvit.py:
class LinearClsHead(nn.Layer):
    """Linear classifier head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        loss (dict): Config of classification loss.
    """

    def __init__(self, num_classes, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        # self.loss = nn.CrossEntropyLoss()
        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')
        self._init_layers()

    def _init_layers(self):
        self.fc = nn.Linear(self.in_channels, self.num_classes, weight_attr=nn.initializer.Constant(value=0.))

    # def init_weights(self):
    #     constant_init(self.fc, val=0, bias=0)

    def forward(self, x, gt_label=None):
        cls_score = self.fc(x)
        # if gt_label is not None:
        #     losses = self.loss(cls_score, gt_label)
        #     return losses
        # else:
        #     return cls_score
        return cls_score


class APViT(nn.Layer):
    """Attentive Pooling ViT"""

    def __init__(self, class_num=7):
        super().__init__()
        self.extractor = IRSEV2(input_size=(112, 112), num_layers=44, mode='ir', return_index=(2,))
        self.vit = PoolingViT(input_type='feature', num_patches=196,
            embed_dim=768, depth=8, num_heads=8, mlp_ratio=3, qkv_bias=False, norm_layer_eps=1e-06,
            in_channels=[256],
            attn_method='SUM_ABS_1',    # CNN Attention method， SUM_ABS_1
            cnn_pool_config=dict(keep_num=160, exclude_first=False),
            vit_pool_configs=dict(keep_rates=[1.] * 6 + [0.9] * 6, exclude_first=True, attn_method='SUM')
            )
        self.head = LinearClsHead(num_classes=class_num, in_channels=768)
        # load_pretrain
        data = torch.load('weights/ir.pdparams')
        self.extractor.set_state_dict(data)
        data = torch.load('weights/vit.pdparams')
        self.vit.set_state_dict(data)

    def forward(self, img, gt_label=None):
        x = self.extractor(img)
        x = self.vit(x)
        x = self.head(x, gt_label)
        return x

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


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
