"""
14/5/25
Visualize Transformer attention maps (similar to Andrey's figure 6 in cvpr paper).
Based on Andrey's code:
andreyg\Projects\Variable_Resolution_DETR\Programming\detr_var\EXPERIMENTS\attention_visualizer\sequence_runner_attn_vis
(also saved it here (in 'Andrey' dir).
"""
from timm.models import create_model
from torchvision import transforms
from PIL import Image
import os
import os.path as osp
import torch
import numpy as np
import matplotlib.pyplot as plt
from attention_wrapper import AttentionWithAttnMap

img_pth = '/home/projects/bagon/shared/imagenet'
img_sub_dir = 'train/n04479046'
img_name = 'n04479046_15'

save_dir = 'figures'
save_figs = True

layer_indices = [0, 4, 11]  # np.arange(12)
attn_map_mode = 'mean'  # 'mean' (mean across attention heads of each layer) / 'all'

# insert model name, or '' for original (pretrained):
model_names = ['', '', 'deit_blur16_tmp_new', 'deit_blur32_tmp_new', '']


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

    save_full_pth = osp.join(save_dir, img_name, mdl) if mdl else osp.join(save_dir, img_name, 'original')
    if save_figs and not osp.isdir(save_full_pth):
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

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(original_image)
            ax.imshow(attn_resized, cmap='jet', alpha=0.4)

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

                ax = axs[h]
                ax.imshow(original_image)
                ax.imshow(attn_resized, cmap='jet', alpha=0.4)

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
            if not osp.isfile(osp.join(save_full_pth, fig_ttl + '.png')):
                fig.savefig(osp.join(save_full_pth, fig_ttl + '.png'))
            else:
                print(f"Figure of layer {i} from model {mdl} already exists => not saving.")
        else:
            plt.show()


def patch_to_index(row, col, grid_size=14):
    return 1 + row * grid_size + col


QUERY_TOKEN_INDEX = patch_to_index(3, 5)  # 1 + 7*14 + 7 = center patch

if not model_names:
    model_names = ['']  # empty model name => use original (pretrained) model.

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
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load image
    img_pil = Image.open(osp.join(img_pth, img_sub_dir, img_name + '.JPEG')).convert("RGB")
    input_tensor = transform(img_pil).unsqueeze(0)  # Shape: (1, 3, 224, 224)

    # Also keep original image for overlay
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
        enc_self_attn_weights[x].append(model.blocks[x].attn.last_attn)


    # -------------------------------
    # 5. Visualize Attention Overlay
    # -------------------------------

    attn_maps = [enc_self_attn_weights[x][0] for x in layer_indices]

    # Visualize
    visualize_all_heads_by_block(attn_maps, token_index=QUERY_TOKEN_INDEX, layer_indices=layer_indices, marker="rectangle",
                                 save_figs=save_figs, mode=attn_map_mode)
    # visualize_attention_overlay(enc_self_attn_weights[0], QUERY_TOKEN_INDEX)

print('done')
