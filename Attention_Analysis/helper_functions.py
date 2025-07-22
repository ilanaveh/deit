"""
4/6/25
Helper functions for Attention_Analysis code.
"""
from timm.models import create_model
import torch
import os
from attention_wrapper import AttentionWithAttnMap
from torchvision import transforms
from PIL import Image, ImageFilter


class GaussianBlur(object):
    """Apply Gaussian blur filter with the given sigma to the input PIL Image.
    Args:
        sigma (int): Desired Gaussian blur level sigma

    Taken from: W:\dannyh\work\code\PyTorch\vggface2_lookdir\datasets\custom_transforms.
   """

    def __init__(self, sigma):
        assert isinstance(sigma, int)
        self.sigma = sigma

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.
        Returns:
            PIL Image: Rescaled image.
        """
        img = img.filter(ImageFilter.GaussianBlur(radius=self.sigma))

        return img

    def __repr__(self):
        return self.__class__.__name__ + '(sigma={0})'.format(self.sigma)


def replace_attention_with_map(model):
    for i, block in enumerate(model.blocks):
        orig = block.attn
        block.attn = AttentionWithAttnMap(
            dim=orig.qkv.in_features,
            num_heads=orig.num_heads,
            qkv_bias=True,
            attn_drop=orig.attn_drop.p,
            proj_drop=orig.proj_drop.p,
        )
        block.attn.load_state_dict(orig.state_dict())  # preserve pretrained weights


def get_model_with_attn(model_path=None):
    # (Based on code in intermediate / deit_probe_intermediate.py)
    # Create deit model with parameters according to those given in main.py:
    model = create_model(
        'deit_base_patch16_224',
        pretrained=True,
        num_classes=1000,
        drop_rate=0,
        drop_path_rate=0.1,
        drop_block_rate=None,
        img_size=224
    )

    model.eval()

    # Replace Attention blocks, with modified blocks that enable access to attention maps:
    replace_attention_with_map(model)

    # Turn fused_attn to false, so we get access to attention-maps (relies on adding line 101 to 'attention_wrapper.py')
    for block in model.blocks:
        block.attn.fused_attn = False

    # Load trained checkpoint:
    if model_path:
        deit_checkpoint = torch.load(os.path.join(model_path, 'best_checkpoint.pth'), map_location='cpu')
        model.load_state_dict(deit_checkpoint['model'])

    return model


def load_and_preprocess_img(img_full_pth, blur, show_im_with_blur):
    # Preprocessing

    transforms_list = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

    if blur:
        transforms_list = [GaussianBlur(blur)] + transforms_list

    transform = transforms.Compose(transforms_list)

    # Load image
    img_pil = Image.open(img_full_pth).convert("RGB")
    input_tensor = transform(img_pil).unsqueeze(0)  # Shape: (1, 3, 224, 224)

    # Also keep original image for overlay
    if show_im_with_blur and blur:
        original_image = transforms.Resize(224)(transforms.CenterCrop(224)(GaussianBlur(blur)(img_pil)))
    else:
        original_image = transforms.Resize(224)(transforms.CenterCrop(224)(img_pil))

    return original_image, input_tensor
