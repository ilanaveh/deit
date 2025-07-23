"""
Create an image similar to Fig7 (right) in yjr vit paper: https://arxiv.org/pdf/2010.11929
Code based on reproduce_git_sayakpaul.py
"""

"""
Repository: https://github.com/sayakpaul/probing-vits/tree/main (see attention-distance plot that I want to reproduce
under "Visualizing mean attention distances" in README.

* Code for computing the mean attention distance:
    https://github.com/sayakpaul/probing-vits/blob/main/notebooks/mean-attention-distance-1k.ipynb

* Code of model (where multiheadattention (MHA) is implemented:
     https://github.com/sayakpaul/probing-vits/blob/main/vit/models.py

Explanation:
Their implementation is in tensorflow, and it's difficult for me to understand how the MHA is computed. So for a sanity
check, try to recreate their attention-distance plot in the deit model (as I did in attention_distances.py, but without
averaging over heads).

"""

import timm
import torch

import numpy as np
import os
import os.path as osp
from helper_functions import load_and_preprocess_img, get_model_with_attn
import torch
import matplotlib.pyplot as plt
import pickle
import json
from attention_distances import model_out_dict, color_map, get_color_for_model

save_fig = True
sep_panels = True
color_heads = True

layer_indices = np.arange(12)
head_indices = np.arange(12)
model_names = ['pretrained', 'deit_blur0_tmp_new', 'deit_blur16_tmp_new', 'deit_blur0-16_tmp_fix_bug']

blur = 0  # input blur
n_patches = 14  # property of deit (14 patches in each row/column -> total 196 patches).
patch_size = 16

limit_n_ims = 10

mdl_label_dict = {
    'pretrained': 'pretrained',
    'deit_blur0_tmp_new': 'high-res',
    'deit_blur16_tmp_new': 'Uniform blur (16)',
    'deit_blur0-16_tmp_fix_bug': 'Variable blur (0-16)'
}

# Add 'pretrained' to color_map:
color_map['pretrained'] = 'magenta'

# Color map for heads:
head_color_dict = {0: 'yellow',
                   1: 'orange',
                   2: 'olive',
                   3: 'blue',
                   4: 'green',
                   5: 'black',
                   6: 'purple',
                   7: 'brown',
                   8: 'pink',
                   9: 'grey',
                   10: 'red',
                   11: 'cyan'}

# Get Imagenet info:
with open('imagenet1000_clsidx_to_labels.txt', 'r') as file:
    imagenet_idx_to_lbl = json.load(file)

with open(osp.join('..', 'out_from_save_class_to_idx', 'class_to_idx.txt'), 'r') as file:
    imagenet_class_to_idx = json.load(file)

# Go over all images in 'trenchcoat' category in validation set:
img_pth = '/home/projects/bagon/shared/imagenet'
img_dataset = 'train'
# img_cat = 'n09472597'  # volcano
# img_cat = 'n01532829'  # house finch
img_cat = 'n04479046'  # trenchcoat
# img_cat = 'n07697313'  # cheeseburger

img_lbl = imagenet_idx_to_lbl[f"{imagenet_class_to_idx[img_cat]}"]
# img_name = 'n04479046_15'

fig_name = osp.join(f'../Attention_Analysis/from_reproduce_vit_figure7/'
                    f'distances_scatter_{img_cat}_{img_lbl}_{len(model_names)}models.png')
fig_name = fig_name.replace('.png', '_sep_panels.png') if sep_panels else fig_name
fig_name = fig_name.replace('.png', '_clrd_heads.png') if color_heads else fig_name


def main():
    # for each model, layer and head - a list of attention-maps for all images (each attnetion-map includes all tokens):
    all_attn = {mdl: {layer: {head: [] for head in head_indices} for layer in layer_indices} for mdl in model_names}

    # for each model, layer and head - a list of distances for all images:
    all_distances = {mdl: {layer: {head: [] for head in head_indices} for layer in layer_indices} for mdl in
                     model_names}

    for mdl in model_names:
        print(mdl)
        # Get attention maps, for each layer:
        # -------------------------------
        # 1. Load deit model:
        # -------------------------------
        if mdl == 'pretrained':
            # don't pass model_path, so by default original (pretrained) deit model is loaded.
            model = get_model_with_attn()
        else:
            model_path = osp.join('/home/projects/bagon/ilanaveh/code/Transformers/deit/', model_out_dict[mdl], mdl)
            model = get_model_with_attn(model_path)

        # -------------------------------
        # 2. Load and preprocess image:
        # -------------------------------

        for (i, img_name) in enumerate(os.listdir(osp.join(img_pth, img_dataset, img_cat))):
            if i > limit_n_ims:
                break

            if img_name == 'Thumbs.db':
                continue
            print(f"image {i}")
            img_full_pth = osp.join(img_pth, img_dataset, img_cat, img_name)
            original_image, input_tensor = load_and_preprocess_img(img_full_pth, blur, show_im_with_blur=False)

            # -------------------------------
            # 3. Forward Pass
            # -------------------------------
            with torch.no_grad():
                out = model(input_tensor)

            for lyr in layer_indices:
                # get attention for each layer, do not average over heads:
                lyr_attn = np.squeeze(model.blocks[lyr].attn.last_attn)  # [nheads=12, ntokens=197, ntokens=197]

                # Remove [CLS] token (which isn't relevant for distances:
                lyr_attn_no_CLS = lyr_attn[:, 1:, 1:]  # [nheads=12, ntokens=196, ntokens=196]

                # Insert to dictionary:
                for h in head_indices:
                    cur_head_attn = lyr_attn_no_CLS[h]  # [ntokens=196, ntokens=196]

                    all_attn[mdl][lyr][h] = cur_head_attn

                    # Initialize a list to store all scaled distances for layer and head (one item for each token)
                    all_scaled_distances = []

                    for token_idx in range(
                            cur_head_attn.shape[0]):  # 0-195 (0 is not [CLS] token, since it was removed)
                        token_attn = cur_head_attn[token_idx, :]

                        # Renormalize (normalization was lost, since we removed [CLS] token):
                        token_attn = token_attn / token_attn.sum()
                        assert np.round(token_attn.sum(), 3) == 1

                        # Get attention grid of token. Shape: [14, 14]
                        token_attn_grid = token_attn.reshape(n_patches, n_patches).detach().cpu().numpy()

                        patch_x, patch_y = divmod(token_idx, n_patches)
                        grid_x, grid_y = torch.meshgrid(torch.arange(n_patches), torch.arange(n_patches), indexing='ij')
                        distances = torch.sqrt((grid_x - patch_x) ** 2 +
                                               (grid_y - patch_y) ** 2)

                        distances = distances * patch_size

                        # Scale distances by the pixel values in the map
                        scaled_distances = distances * token_attn_grid

                        # Sum the scaled distances and add to list
                        all_scaled_distances.append(
                            scaled_distances.sum())  # mean over tokens (use sum since we had softmax)

                    # Compute the average scaled distance
                    avg_scaled_distance = np.mean(all_scaled_distances)
                    all_distances[mdl][lyr][h].append(avg_scaled_distance)

    # plot:
    if sep_panels:
        fig, subplots = plt.subplots(2, 2)  # assuming 4 models, change if needed
        fig.set_size_inches([8, 6])
        subplots = subplots.flatten()

        # Plot each model's attention distances:
        for i, mdl in enumerate(model_names):
            ax = subplots[i]
            # Set axis properties:
            ax.set_ylim([0, 140])
            ax.set_title(mdl_label_dict[mdl])
            ax.set_ylabel('Attention distance (px)')
            ax.set_xlabel('Layer')

            for lyr in layer_indices:
                lyr_dists = all_distances[mdl][lyr]
                all_heads_mean_dist = [np.mean(lyr_dists[h]) for h in head_indices]
                if color_heads:
                    for h in head_indices:
                        color = head_color_dict[h]
                        ax.scatter(lyr, all_heads_mean_dist[h], marker='o', color=color)
                else:
                    ax.scatter(np.repeat(lyr, len(all_heads_mean_dist)), all_heads_mean_dist, marker='o', color='b')

        plt.suptitle(f"{img_cat} ({img_lbl}), {len(lyr_dists[h])} images\nInput Blur: {blur}")
        plt.tight_layout()

    else:
        fig = plt.figure()
        used_labels = set()  # for validating that each label is only entered once to the legend.
        for mdl in model_names:
            color = get_color_for_model(mdl)
            for lyr in layer_indices:
                lyr_dists = all_distances[mdl][lyr]
                all_heads_mean_dist = [np.mean(lyr_dists[h]) for h in head_indices]
                label = mdl_label_dict[mdl] if (mdl not in used_labels) else None
                plt.scatter(np.repeat(lyr, len(all_heads_mean_dist)), all_heads_mean_dist,
                            marker='o', color=color, label=label)
                used_labels.add(mdl)

        plt.xticks(head_indices)
        plt.xlabel('Network Depth (Layer)')
        plt.ylabel('Mean Attention Distance (px)')
        plt.title(f"{img_cat} ({img_lbl}), {len(lyr_dists[h])} images\n(model: deit pretrained)")
        plt.legend()
        fig = plt.gcf()
        fig.set_size_inches([7.2, 4.75])
        plt.tight_layout()

    if save_fig:
        plt.savefig(fig_name)

    print('done')


if __name__ == "__main__":
    main()
