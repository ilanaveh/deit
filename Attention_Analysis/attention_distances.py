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
import os
import os.path as osp
from helper_functions import load_and_preprocess_img, get_model_with_attn
import torch
import matplotlib.pyplot as plt
import pickle


save_file = True  # whether to save all_models_distances dictionary.
layer_indices = np.arange(12)
model_names = ['', 'deit_blur0_tmp_new', 'deit_blur16_tmp_new', 'deit_blur32_tmp_new',
               'deit_blur0-16_tmp_fix_bug', 'deit_blur16-32_tmp', 'deit_blur0-32_tmp_new']
blur = 0  # input blur
n_patches = 14  # property of deit

# Go over all images in 'trenchcoat' category in validation set:
img_pth = '/home/projects/bagon/shared/imagenet'
img_dataset = 'val'
img_cat = 'n04479046'
img_cat = 'n01443537'
# img_name = 'n04479046_15'

filename = osp.join(f'../Attention_Analysis/all_models_distances_{img_cat}.pkl')

if osp.isfile(filename):
    with open(filename, "rb") as file:
        all_models_distances = pickle.load(file)
else:
    all_models_attn = {layer: {} for layer in layer_indices}

    # for each model and layer - a list of distances for all images:
    all_models_distances = {layer: {mdl: [] for mdl in model_names} for layer in layer_indices}

    # Get attention maps, for each layer and each model:
    for mdl in model_names:
        print(mdl)

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

        for (i, img_name) in enumerate(os.listdir(osp.join(img_pth, img_dataset, img_cat))):
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
                # get attention for each layer, mean over heads:
                lyr_attn = np.squeeze(model.blocks[lyr].attn.last_attn).mean(0)

                # Remove [CLS] token (which isn't relevant for distances:
                lyr_attn_no_CLS = lyr_attn[1:, 1:]

                # Insert to dictionary:
                all_models_attn[lyr][mdl or 'original'] = lyr_attn_no_CLS

                # Get distances for each token (mostly based on Andrey (Projects\Variable_Resolution_DETR\Programming\
                #   detr_var\EXPERIMENTS\attention_map_graph_generator\objects\graph_plotter.py):

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
                avg_scaled_distance = (total_scaled_distances / query_points_counted).numpy()[0]
                all_models_distances[lyr][mdl].append(avg_scaled_distance)

    if save_file:
        with open(filename, 'wb') as file:
            pickle.dump(all_models_distances, file)

# plot:
for model in model_names:
    means = [np.mean(all_models_distances[layer][model]) for layer in layer_indices]
    stds = [np.std(all_models_distances[layer][model]) for layer in layer_indices]
    plt.errorbar(layer_indices, means, yerr=stds, label=model.split('_tmp')[0], marker='o', linestyle='-')

plt.xticks(layer_indices)
plt.xlabel('Layer')
plt.ylabel('Mean Attention Distance')
plt.legend()

print('done')
