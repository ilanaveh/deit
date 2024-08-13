"""
13/08/2024
Probe intermediate layers from deit, and train on AffectNet.
"""

from timm.models import create_model
import torch.nn as nn
import numpy as np


def create_model_probed(model_path, block_ind, num_classes=2):
    """
    create model based on a pretrained deit, with a FC layer after the chosen block from deit.
    Freeze all parameters
    except for the new FC layer.
    :param model_path: path to trained deit checkpoint, from which we want to start.
    :param block_ind: number of block we want to probe from (its output) - between 0-11.
    :return:
    """

    # Create deit model with parameters according to those given in main.py:
    model = create_model(
        'deit_base_patch16_224',
        pretrained=False,
        num_classes=1000,
        drop_rate=0,
        drop_path_rate=0.1,
        drop_block_rate=None,
        img_size=224
    )

    model.cuda()
    model.eval()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Remove all blocks downstream to chosen block: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    inds2rmv = range(len(model.blocks)-1, block_ind, -1)

    for i in inds2rmv:
        del model.blocks[i]

    # ~~~~~~~~~~~~~~~~~~~~~~~ Replace last FC layer with a new one, for binary classification: ~~~~~~~~~~~~~~~~~~~~~~~~~
    model.head = nn.Linear(model.embed_dim, num_classes)  # embed_dim = 768 (don't change, only change num_classes).

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Freeze weights of all layers, except for last FC: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    for param in model.parameters():
        param.requires_grad = False

    model.head.bias.requires_grad = True
    model.head.weight.requires_grad = True

    return model