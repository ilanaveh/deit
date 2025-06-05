"""
3/6/25
Compute Attention distances for each token (the average spatial distance to all other tokens, weighted by the attention
maps).
Based on Andrey's CVPR paper ("Seeing more with less: human-like representations in vision models")
and code: andreyg\Projects\Variable_Resolution_DETR\Programming\detr_var\EXPERIMENTS\attention_map_graph_generator\
            objects\graph_plotter.py

Currently, distances for all tokens are averaged, as opposed to Andrey who separated between center and periphery.
"""

import numpy as np
import os.path as osp
from helper_functions import load_and_preprocess_img, get_model_with_attn
import torch


layer_indices = np.arange(12)
model_names = ['', 'deit_blur0_tmp_new', 'deit_blur16_tmp_new', 'deit_blur32_tmp_new',
               'deit_blur0-16_tmp_fix_bug', 'deit_blur16-32_tmp', 'deit_blur0-32_tmp_new']
blur = 0  # input blur
n_patches = 14  # property of deit

# Try for specific image:
img_pth = '/home/projects/bagon/shared/imagenet'
img_sub_dir = 'train/n04479046'
img_name = 'n04479046_15'

# Get attention maps, for each layer and each model:
all_models_attn = {layer: {} for layer in layer_indices}

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
    original_image, input_tensor = load_and_preprocess_img(img_full_pth, blur, show_im_with_blur=False)

    # -------------------------------
    # 3. Forward Pass
    # -------------------------------
    with torch.no_grad():
        out = model(input_tensor)

    for lyr in layer_indices:
        # get attention for each layer, mean over heads:
        lyr_attn = np.squeeze(model.blocks[lyr].attn.last_attn).mean(0)

        # Remove [CLS] token (which isn't relevant for distances:
        lyr_attn_no_CLS = lyr_attn[1:, 1:]

        # Insert to dictionary:
        all_models_attn[lyr][mdl or 'original'] = lyr_attn_no_CLS

        # Get distances for each token (mostly based on Andrey (Projects\Variable_Resolution_DETR\Programming\detr_var\
        #   EXPERIMENTS\attention_map_graph_generator\objects\graph_plotter.py):

        # Initialize a tensor to store the sum of all scaled distances
        total_scaled_distances = torch.zeros((1))
        query_points_counted = 0

        for token_idx in range(lyr_attn_no_CLS.shape[0]):  # 0-195 (0 is not [CLS] token, since it was removed)
            patch_attn_grid = lyr_attn_no_CLS[token_idx, :].reshape(n_patches, n_patches).detach().cpu().numpy()

            patch_x, patch_y = divmod(token_idx, n_patches)
            grid_x, grid_y = torch.meshgrid(torch.arange(n_patches), torch.arange(n_patches), indexing='ij')
            distances = torch.sqrt((grid_x - patch_x) ** 2 +
                                   (grid_y - patch_y) ** 2)

            # Scale distances by the pixel values in the map
            scaled_distances = distances * patch_attn_grid

            # Sum the scaled distances and add to total
            total_scaled_distances += scaled_distances.mean()
            query_points_counted += 1

        # Compute the average scaled distance
        avg_scaled_distance = total_scaled_distances / query_points_counted

print('done')
