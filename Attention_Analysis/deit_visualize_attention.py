"""
14/5/25
Visualize Transformer attention maps (similar to Andrey's figure 6 in cvpr paper).
Based on Andrey's code:
andreyg\Projects\Variable_Resolution_DETR\Programming\detr_var\EXPERIMENTS\attention_visualizer\sequence_runner_attn_vis
(also saved it here (in 'Andrey' dir).
"""
import matplotlib
from timm.models import create_model
import json
import os
import os.path as osp
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helper_functions import get_model_with_attn, load_and_preprocess_img
from collections import defaultdict

# -------------------- CONFIG --------------------
img_pth = '/home/projects/bagon/shared/imagenet'
img_cat = 'n04479046'
img_sub_dir = osp.join('train', img_cat)
img_name = 'n04479046_15'

save_dir = 'figures'
save_figs = False
save_original = False  # whether to save original image (after transforms, but without overlayed attention map)

# Choose whether to test model's performance on input image
test_performance = True
save_performance_dir = 'models_performance_per_image'

# Insert blur sigma:
blur = 16  # Input blur
show_im_with_blur = True  # whether to visualize images with chosen input blur (if False - visualize high-res).

layer_indices = [4]  # np.arange(12)
head_mode = 'mean'  # 'mean' (mean across attention heads of each layer) / 'all'
lyr_mode = 'mean'  # 'mean' (mean across attention heads of each layer) / 'all'

patch_coord = []  # [] - CLS token, [3, 5] - left eye. [5, 5] - mouth. [5, 8] / [6, 9] - jacket collar. [] - jacket
cmp_mode = 'layers'  # 'models' (fig for each layer, compare models) / 'layers' (fig for each model, compare layers)
if cmp_mode == 'layers':
    layer_indices = np.arange(12)

# Insert model name, or '' for original (pretrained):
model_names = ['', 'deit_blur0_tmp_new', 'deit_blur16_tmp_new', 'deit_blur32_tmp_new',
               'deit_blur0-16_tmp_fix_bug', 'deit_blur16-32_tmp', 'deit_blur0-32_tmp_new']


# -------------------- UTILITIES --------------------


def patch_to_index(coord_list, grid_size=14):
    if coord_list:
        return 1 + coord_list[0] * grid_size + coord_list[1]
    return 0  # for [CLS] token


def get_attn_map(attn_tensor, head_mode):
    """

    :param attn_tensor: shape: [heads, tokens, tokens] (either for specific layer, or mean across layers)
    :param head_mode: 'mean' / 'all'
    :return:
    """
    # Get token attention map (start from index 1, since 0 is [CLS] token)
    if head_mode == "mean":
        token_attn = attn_tensor[:, QUERY_TOKEN_INDEX, 1:].mean(0)
    else:
        token_attn = attn_tensor[0, QUERY_TOKEN_INDEX, 1:]  # just head 0

    token_attn = token_attn.reshape(14, 14).detach().cpu().numpy()
    token_attn = np.clip(token_attn, 0, None)
    token_attn = (token_attn - token_attn.min()) / (token_attn.max() - token_attn.min())
    attn_resized = np.kron(token_attn, np.ones((16, 16)))  # upsample to 224x224

    # Map to RGBA with variable alpha (s.t. low attention values would be transparent)
    cmap = matplotlib.colormaps['jet']
    colors = cmap(attn_resized)
    max_alpha = 0.6
    colors[..., 3] = attn_resized * max_alpha

    return colors


# -------------------- MAIN PROCESS --------------------
QUERY_TOKEN_INDEX = patch_to_index(patch_coord)  # 1 + 7*14 + 7 = center patch

if not model_names:
    model_names = ['']  # empty model name => use original (pretrained) model.

all_models_attn = {layer: {} for layer in layer_indices}
if test_performance:
    all_models_performance = {}

for mdl in model_names:
    model_path = osp.join(
        '/home/projects/bagon/ilanaveh/code/Transformers/deit/out/jobs_from_scratch_main_tmp_code/', mdl) \
        if mdl else ''
    # -------------------------------
    # 1. Load deit model:
    # -------------------------------
    model = get_model_with_attn(model_path)

    # -------------------------------
    # 2. Load and preprocess image:
    # -------------------------------
    img_full_pth = osp.join(img_pth, img_sub_dir, img_name + '.JPEG')
    original_image, input_tensor = load_and_preprocess_img(img_full_pth, blur, show_im_with_blur)

    # -------------------------------
    # 3. Forward Pass
    # -------------------------------

    with torch.no_grad():
        out = model(input_tensor)

    if test_performance:
        with open(osp.join('..', 'out_from_save_class_to_idx', 'class_to_idx.txt'), 'r') as file:
            class_to_idx = json.load(file)
        target_idx = class_to_idx[img_cat]
        out_idx_ascending = np.argsort(out[0]).numpy()
        out_idx_descending = out_idx_ascending[::-1]  # reverse order, so highest is first.

        pos_of_target = np.where(out_idx_descending == target_idx)[0][0]

        if pos_of_target < 5:
            print(f"Target is in top-5 prediction of model {mdl}")
        else:
            print(f"Target was not predicted correctly by model {mdl}")

        all_models_performance[mdl or 'original'] = pos_of_target

    for x in layer_indices:
        all_models_attn[x][mdl or 'original'] = model.blocks[x].attn.last_attn

if test_performance:
    performance_df_pth = osp.join(save_performance_dir, f'models_performance_for_img_{img_name}')
    # If dataframe already exists, load it & add new data. Otherwise, create dataframe from dictionary:
    if osp.isfile(performance_df_pth):
        perf_df = pd.read_csv(performance_df_pth)
        need2add = (not perf_df['input_blur'].isin([blur])[0]) or (mdl not in perf_df)
        if need2add:
            for mdl in model_names:
                mdl = mdl or 'original'
                perf_df.at[blur, mdl] = all_models_performance[mdl]
    else:
        # turn to list for conversion to df:
        perf_dict_for_df = {mdl: [perf] for (mdl, perf) in all_models_performance.items()}
        # Add input blur:
        perf_dict_for_df['input_blur'] = blur
        # Convert to dataframe:
        perf_df = pd.DataFrame(perf_dict_for_df)
        # Move 'input_blur' to be first:
        col = perf_df.pop('input_blur')
        perf_df.insert(0, 'input_blur', col)

    # Save dataframe:
    perf_df.to_csv(performance_df_pth, index=False)
# -------------------- VISUALIZATION --------------------
if cmp_mode == 'models':
    for layer in layer_indices:
        fig, axs = plt.subplots(2, 4, figsize=(9, 4.5))
        axs = axs.flatten()

        for i, mdl in enumerate(model_names):
            name = mdl or 'original'
            attn = all_models_attn[layer][name][0]  # shape: (heads, tokens, tokens)

            colors = get_attn_map(attn, head_mode)  # Get token attention map

            ax = axs[i] if i < 4 else axs[i + 1]

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

            # Save copy of original image (after Transforms, but without attention map):
            if save_original:
                # Create figure:
                fig_ori = plt.figure()
                ax = plt.gca()
                ax.imshow(original_image)
                ax.axis('off')
                plt.tight_layout()

                # Save figure:
                save_ori_nm = f"{img_name}_clean - Blurred.png" if (show_im_with_blur and blur) else f"{img_name}_clean"
                out_path_ori = osp.join(save_dir, img_name, save_ori_nm)
                plt.savefig(out_path_ori)

        else:
            plt.show()

        plt.close()

elif cmp_mode == 'layers':
    for mdl in model_names:
        name = mdl or 'original'

        if lyr_mode == 'mean':
            fig = plt.figure()
            ax = plt.gca()

            # Get new tensor, which holds Attention-Maps of all layers along its first dimension
            # (shape: [layers, heads, tokens, tokens])
            attn_all_lyrs = torch.cat([all_models_attn[lyr][name][0].unsqueeze(0) for lyr in layer_indices], dim=0)

            # Then, get the mean across layers:
            attn_mean_lyrs = attn_all_lyrs.mean(0)

            colors = get_attn_map(attn_mean_lyrs, head_mode)  # get attention map (mean across layers)

            ax.imshow(original_image)
            ax.imshow(colors)

            if QUERY_TOKEN_INDEX != 0:
                patch_row, patch_col = divmod(QUERY_TOKEN_INDEX - 1, 14)
                x = patch_col * 16
                y = patch_row * 16
                rect = plt.Rectangle((x, y), 16, 16, edgecolor='white', facecolor='none', linewidth=2)
                ax.add_patch(rect)

            fig_ttl = f"Model '{name}' - Token {QUERY_TOKEN_INDEX} - Input Blur {blur} - Mean Across Layers"
            ax.set_title(fig_ttl, fontsize=10)
            ax.axis('off')

        elif lyr_mode == 'all':
            fig, axs = plt.subplots(3, 4, figsize=(12, 9))
            axs = axs.flatten()

            for i, layer in enumerate(layer_indices):
                attn = all_models_attn[layer][name][0]  # ([0] is just for extracting tensor, there are no other indices)
                # attn shape: (heads, tokens, tokens)

                colors = get_attn_map(attn, head_mode)

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
            fig_ttl = f"Model '{name}' - Token {QUERY_TOKEN_INDEX} - Input Blur {blur} - All Layers"
            plt.suptitle(fig_ttl, fontsize=14)

        if save_figs:
            save_nm = f"{fig_ttl} - Blurred.png" if (show_im_with_blur and blur) else f"{fig_ttl}.png"
            out_path = osp.join(save_dir, img_name, save_nm)
            plt.savefig(out_path)

            # Save copy of original image (after Transforms, but without attention map):
            if save_original:
                # Create figure:
                fig_ori = plt.figure()
                ax = plt.gca()
                ax.imshow(original_image)
                ax.axis('off')
                plt.tight_layout()

                # Save figure:
                save_ori_nm = f"{img_name}_clean - Blurred.png" if (show_im_with_blur and blur) else f"{img_name}_clean"
                out_path_ori = osp.join(save_dir, img_name, save_ori_nm)
                plt.savefig(out_path_ori)

        else:
            plt.show()

        plt.close()

print('done')
