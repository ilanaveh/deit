"""
14/5/25
Visualize Transformer attention maps (similar to Andrey's figure 6 in cvpr paper).
Based on Andrey's code:
andreyg\Projects\Variable_Resolution_DETR\Programming\detr_var\EXPERIMENTS\attention_visualizer\sequence_runner_attn_vis
(also saved it here (in 'Andrey' dir).
"""
import matplotlib
from timm.models import create_model
from torchvision import transforms
from PIL import Image, ImageFilter
import os
import os.path as osp
import torch
import numpy as np
import matplotlib.pyplot as plt
from attention_wrapper import AttentionWithAttnMap
from collections import defaultdict

# -------------------- CONFIG --------------------
img_pth = '/home/projects/bagon/shared/imagenet'
img_sub_dir = 'train/n04479046'
img_name = 'n04479046_15'

save_dir = 'figures'
save_figs = True

layer_indices = [4]  # np.arange(12)
attn_map_mode = 'mean'  # 'mean' (mean across attention heads of each layer) / 'all'
patch_coord = [5, 5]  # [3, 5] - left eye. [5, 5] - mouth
cmp_mode = 'models'  # 'models' (fig for each layer, compare models) / 'layers' (fig for each model, compare layers)
if cmp_mode == 'layers':
    layer_indices = np.arange(12)

# Insert model name, or '' for original (pretrained):
model_names = ['', 'deit_blur0_tmp_new', 'deit_blur16_tmp_new', 'deit_blur32_tmp_new',
               'deit_blur0-16_tmp_fix_bug', 'deit_blur16-32_tmp', 'deit_blur0-32_tmp_new']

# Insert blur sigma:
blur = 8  # Input blur
show_im_with_blur = False  # whether to visualize images with chosen input blur (if False - visualize high-res).


# -------------------- UTILITIES --------------------
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


def visualize_all_heads_by_block(attn_maps, token_index, layer_indices, marker="rectangle", grid_size=14, mode="mean",
                                 save_figs=0):
    """

    :param attn_maps:
    :param token_index:
    :param layer_indices:
    :param marker: 'rectangle' / 'dot' / 'crosshair' / 'None'
    :param grid_size:
    :param mode: 'all' (all heads, in separate subplots) / 'mean' (only mean across heads)
    :param save_figs:
    :return:
    """

    save_full_pth = osp.join(save_dir, img_name, f'Blur{blur}', mdl) if mdl else \
        osp.join(save_dir, img_name, f'Blur{blur}', 'original')
    if save_figs:
        if not osp.isdir(osp.join(save_dir, img_name)):
            print(f"Creating new Directory: {osp.join(save_dir, img_name)}")
            os.mkdir(osp.join(save_dir, img_name))

        if not osp.isdir(osp.join(save_dir, img_name, f'Blur{blur}')):
            print(f"Creating new Directory: {osp.join(save_dir, img_name, f'Blur{blur}')}")
            os.mkdir(osp.join(save_dir, img_name, f'Blur{blur}'))

        if not osp.isdir(save_full_pth):
            print(f"Creating new Directory: {save_full_pth}")
            os.mkdir(save_full_pth)

    patch_size = 224 // grid_size

    for i, attn in zip(layer_indices, attn_maps):
        attn = attn[0]  # (heads, tokens, tokens)

        if mode == "mean":
            token_attn = attn[:, token_index, 1:].mean(0)  # (196,)
            token_attn = token_attn.reshape(grid_size, grid_size).detach().cpu().numpy()
            token_attn = np.clip(token_attn, 0, None)
            token_attn = (token_attn - token_attn.min()) / (token_attn.max() - token_attn.min())
            attn_resized = np.kron(token_attn, np.ones((patch_size, patch_size)))

            # Map to RGBA with variable alpha (s.t. low attention values would be transparent)
            cmap = matplotlib.colormaps['jet']
            colors = cmap(attn_resized)
            max_alpha = 0.6
            colors[..., 3] = attn_resized * max_alpha

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(original_image)
            ax.imshow(colors)

            if token_index != 0 and marker != "none":
                patch_row, patch_col = divmod(token_index - 1, grid_size)
                x = patch_col * patch_size - 1
                y = patch_row * patch_size

                if marker == "rectangle":
                    rect = plt.Rectangle((x, y), patch_size, patch_size,
                                         edgecolor='white', facecolor='none', linewidth=2)
                    ax.add_patch(rect)
                elif marker == "crosshair":
                    center_x = x + patch_size // 2
                    center_y = y + patch_size // 2
                    ax.axhline(y=center_y, color='white', linestyle='--', linewidth=1)
                    ax.axvline(x=center_x, color='white', linestyle='--', linewidth=1)

            fig_ttl = f"Layer {i} - Token {token_index} Attention - Mean over heads"
            ax.set_title(fig_ttl)
            ax.axis('off')
            plt.tight_layout()

        elif mode == "all":

            fig, axs = plt.subplots(3, 4, figsize=(12, 9))
            axs = axs.flatten()

            # Loop over attention heads:
            for h in range(attn.shape[0]):
                token_attn = attn[h, token_index, 1:]  # (196,)
                token_attn = token_attn.reshape(grid_size, grid_size).detach().cpu().numpy()
                token_attn = np.clip(token_attn, 0, None)
                token_attn = (token_attn - token_attn.min()) / (token_attn.max() - token_attn.min())
                attn_resized = np.kron(token_attn, np.ones((patch_size, patch_size)))

                # Map to RGBA with variable alpha (s.t. low attention values would be transparent)
                cmap = matplotlib.colormaps['jet']
                colors = cmap(attn_resized)
                max_alpha = 0.6
                colors[..., 3] = attn_resized * max_alpha

                ax = axs[h]
                ax.imshow(original_image)
                ax.imshow(colors)

                if token_index != 0 and marker != "none":
                    patch_row, patch_col = divmod(token_index - 1, grid_size)
                    x = patch_col * patch_size
                    y = patch_row * patch_size

                    if marker == "rectangle":
                        rect = plt.Rectangle((x, y), patch_size, patch_size,
                                             edgecolor='white', facecolor='none', linewidth=2)
                        ax.add_patch(rect)

                    elif marker == "crosshair":
                        center_x = x + patch_size // 2
                        center_y = y + patch_size // 2
                        ax.axhline(y=center_y, color='white', linestyle='--', linewidth=1)
                        ax.axvline(x=center_x, color='white', linestyle='--', linewidth=1)

                ax.set_title(f"Head {h}")
                ax.axis('off')

            # Hide unused subplots (if heads < 12)
            for j in range(attn.shape[0], len(axs)):
                axs[j].axis('off')

            fig_ttl = f"Layer {i} - Token {token_index} Attention"
            plt.suptitle(fig_ttl, fontsize=14)
            plt.tight_layout()

        else:
            raise ValueError(f"Unsupported mode: {mode}. Choose 'all' or 'mean'.")

        if save_figs:
            fig_ttl_full = fig_ttl+' - blurred' if (show_im_with_blur and blur) else fig_ttl
            if not osp.isfile(osp.join(save_full_pth, fig_ttl_full + '.png')):
                fig.savefig(osp.join(save_full_pth, fig_ttl_full + '.png'))
            else:
                print(f"Figure of layer {i} from model {mdl} already exists => not saving.")
        else:
            plt.show()

        plt.close()


def patch_to_index(coord_list, grid_size=14):
    return 1 + coord_list[0] * grid_size + coord_list[1]


# -------------------- MAIN PROCESS --------------------
QUERY_TOKEN_INDEX = patch_to_index(patch_coord)  # 1 + 7*14 + 7 = center patch

if not model_names:
    model_names = ['']  # empty model name => use original (pretrained) model.

all_models_attn = {layer: {} for layer in layer_indices}

for mdl in model_names:
    model_path = osp.join(
        '/home/projects/bagon/ilanaveh/code/Transformers/deit/out/jobs_from_scratch_main_tmp_code/', mdl) \
        if mdl else ''
    # -------------------------------
    # 1. Load deit model (Based on intermediate/deit_probe_intermediate.py):
    # -------------------------------

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

    # -------------------------------
    # 2. Load and preprocess image:
    # -------------------------------
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
    img_pil = Image.open(osp.join(img_pth, img_sub_dir, img_name + '.JPEG')).convert("RGB")
    input_tensor = transform(img_pil).unsqueeze(0)  # Shape: (1, 3, 224, 224)

    # Also keep original image for overlay
    if show_im_with_blur and blur:
        original_image = transforms.Resize(224)(transforms.CenterCrop(224)(GaussianBlur(blur)(img_pil)))
    else:
        original_image = transforms.Resize(224)(transforms.CenterCrop(224)(img_pil))

    # -------------------------------
    # 3. Register forward hooks:
    # -------------------------------

    # arrays in which we save the featuremaps
    enc_self_attn_weights = {x: [] for x in layer_indices}  # one list for each layer

    # hooks = [
    #     model.blocks[x].attn.attn_drop.register_forward_hook(
    #         lambda self, input, output: enc_self_attn_weights[x].append(output))
    #
    #     for x in layer_indices
    # ]

    # -------------------------------
    # 4. Forward Pass
    # -------------------------------
    with torch.no_grad():
        _ = model(input_tensor)

    # for hook in hooks:
    #     hook.remove()

    for x in layer_indices:
        all_models_attn[x][mdl or 'original'] = model.blocks[x].attn.last_attn

# -------------------- VISUALIZATION --------------------
if cmp_mode == 'models':
    for layer in layer_indices:
        fig, axs = plt.subplots(2, 4, figsize=(9, 4.5))
        axs = axs.flatten()
        # axs[4]._visible = False
        for i, mdl in enumerate(model_names):
            name = mdl or 'original'
            attn = all_models_attn[layer][name][0]  # shape: (heads, tokens, tokens)

            # Get token attention map
            if attn_map_mode == "mean":
                token_attn = attn[:, QUERY_TOKEN_INDEX, 1:].mean(0)
            else:
                token_attn = attn[0, QUERY_TOKEN_INDEX, 1:]  # just head 0

            token_attn = token_attn.reshape(14, 14).detach().cpu().numpy()
            token_attn = np.clip(token_attn, 0, None)
            token_attn = (token_attn - token_attn.min()) / (token_attn.max() - token_attn.min())
            attn_resized = np.kron(token_attn, np.ones((16, 16)))  # upsample to 224x224

            # Map to RGBA with variable alpha (s.t. low attention values would be transparent)
            cmap = matplotlib.colormaps['jet']
            colors = cmap(attn_resized)
            max_alpha = 0.6
            colors[..., 3] = attn_resized * max_alpha

            ax = axs[i] if i < 4 else axs[i+1]

            ax.imshow(original_image)
            ax.imshow(colors)

            if QUERY_TOKEN_INDEX != 0:
                patch_row, patch_col = divmod(QUERY_TOKEN_INDEX - 1, 14)
                x = patch_col * 16
                y = patch_row * 16
                rect = plt.Rectangle((x, y), 16, 16, edgecolor='white', facecolor='none', linewidth=2)
                ax.add_patch(rect)

            ax.set_title(f"{name.split('_tmp')[0]}", fontsize=10)
            ax.axis('off')

        axs[4].axis('off')
        fig_ttl = f"Layer {layer} - Token {QUERY_TOKEN_INDEX} - Input Blur {blur}"
        plt.suptitle(fig_ttl, fontsize=14)
        # plt.tight_layout()
        # plt.subplots_adjust(hspace=0, top=0.9)  # increase space between rows
        # plt.t

        if save_figs:
            save_nm = f"Compare Models - {fig_ttl} - Blurred.png" if (show_im_with_blur and blur) \
                else f"Compare Models - {fig_ttl}.png"
            out_path = osp.join(save_dir, img_name, save_nm)
            plt.savefig(out_path)
        else:
            plt.show()

elif cmp_mode == 'layers':
    for mdl in model_names:
        fig, axs = plt.subplots(3, 4, figsize=(12, 9))
        axs = axs.flatten()
        name = mdl or 'original'

        for i, layer in enumerate(layer_indices):
            attn = all_models_attn[layer][name][0]  # shape: (heads, tokens, tokens)

            # Get token attention map
            if attn_map_mode == "mean":
                token_attn = attn[:, QUERY_TOKEN_INDEX, 1:].mean(0)
            else:
                token_attn = attn[0, QUERY_TOKEN_INDEX, 1:]  # just head 0

            token_attn = token_attn.reshape(14, 14).detach().cpu().numpy()
            token_attn = np.clip(token_attn, 0, None)
            token_attn = (token_attn - token_attn.min()) / (token_attn.max() - token_attn.min())
            attn_resized = np.kron(token_attn, np.ones((16, 16)))  # upsample to 224x224

            # Map to RGBA with variable alpha (s.t. low attention values would be transparent)
            cmap = matplotlib.colormaps['jet']
            colors = cmap(attn_resized)
            max_alpha = 0.6
            colors[..., 3] = attn_resized * max_alpha

            ax = axs[i]

            ax.imshow(original_image)
            ax.imshow(colors)

            if QUERY_TOKEN_INDEX != 0:
                patch_row, patch_col = divmod(QUERY_TOKEN_INDEX - 1, 14)
                x = patch_col * 16
                y = patch_row * 16
                rect = plt.Rectangle((x, y), 16, 16, edgecolor='white', facecolor='none', linewidth=2)
                ax.add_patch(rect)

            ax.set_title(f"Layer {layer}", fontsize=10)
            ax.axis('off')

        axs[4].axis('off')
        fig_ttl = f"Model '{name}' - Token {QUERY_TOKEN_INDEX} - Input Blur {blur}"
        plt.suptitle(fig_ttl, fontsize=14)

        if save_figs:
            save_nm = f"{fig_ttl} - All Layers - Blurred.png" if (show_im_with_blur and blur) \
                else f"{fig_ttl} - All Layers.png"
            out_path = osp.join(save_dir, img_name, save_nm)
            plt.savefig(out_path)
        else:
            plt.show()

plt.close()

print('done')
