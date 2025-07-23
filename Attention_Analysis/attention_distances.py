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
import json


# Dictionary for 'out' folder of each model:
model_out_dict = {
    'deit_blur0_tmp_new': osp.join('out', 'jobs_from_scratch_main_tmp_code'),
    'deit_blur2_tmp_new': osp.join('out', 'jobs_from_scratch_main_tmp_code'),
    'deit_blur4_tmp_new': osp.join('out', 'jobs_from_scratch_main_tmp_code'),
    'deit_blur6_tmp_new': osp.join('out', 'jobs_from_scratch_main_tmp_code'),
    'deit_blur8_tmp_new': osp.join('out', 'jobs_from_scratch_main_tmp_code'),
    'deit_blur32_tmp_new': osp.join('out', 'jobs_from_scratch_main_tmp_code'),
    'deit_blur0-32_tmp_new': osp.join('out', 'jobs_from_scratch_main_tmp_code'),
    'deit_blur6_rep': osp.join('out', 'jobs_from_scratch_main_tmp_code'),
    'deit_blur8_rep': osp.join('out', 'jobs_from_scratch_main_tmp_code'),
    'deit_blur16_tmp_new': osp.join('out', 'jobs_from_scratch_main_tmp_code'),
    'deit_blur0-16_tmp': osp.join('out', 'jobs_from_scratch_main_tmp_code'),
    'deit_blur0-16_tmp_fix_bug': osp.join('out', 'jobs_from_scratch_main_tmp_code'),
    'deit_blur16-32_tmp': osp.join('out', 'jobs_from_scratch_main_tmp_code'),
    'deit_blur0-16_rep': osp.join('out', 'jobs_from_scratch_main_tmp_code'),
    'deit_blur0-32_rep': osp.join('out', 'jobs_from_scratch_main_tmp_code'),
    'deit_blur16-32_rep': osp.join('out', 'jobs_from_scratch_main_tmp_code'),
    'original': 'out',
    'deit_blur4': 'out',
    'deit_blur8': 'out',
    'deit_blur16': 'out',
    'deit_blur32': 'out',
    'deit_blur0-32_tmp': 'out',
    'deit_blur4_rep': 'out'
}

# Create a function to get the appropriate color based on model name (from deit_plot_performance.py)
color_map = {
    'blur0-32': 'cyan',
    'blur0-16': 'orange',
    'blur16-32': 'olive',
    'blur0': 'blue',
    'original': 'blue',
    'blur2': 'green',
    'blur4': 'red',
    'blur6': 'black',
    'blur8': 'purple',
    'blur16': 'pink',
    'blur32': 'brown'

}


def get_color_for_model(model_name):
    for blur_level in color_map.keys():
        if blur_level in model_name:
            return color_map[blur_level]
    return 'gray'  # Default color if no match is found


def main():
    save_file = False  # whether to save all_models_distances dictionary.
    save_fig = False
    start_from_saved_data_when_possible = False
    sep_panels = True  # whether to show each model in its own subplot.
    add_thresh = True

    layer_indices = np.arange(12)
    # model_names = ['', 'deit_blur0_tmp_new', 'deit_blur16_tmp_new', 'deit_blur32_tmp_new',
    #                'deit_blur0-16_tmp_fix_bug', 'deit_blur16-32_tmp', 'deit_blur0-32_tmp_new',
    #                '']

    model_names = [
        # models saved in 'out/jobs_from_scratch_main_tmp_code':
        'deit_blur0_tmp_new', 'deit_blur2_tmp_new', 'deit_blur4_tmp_new', 'deit_blur6_tmp_new', 'deit_blur6_rep',
        'deit_blur8_rep', 'deit_blur16_tmp_new', 'deit_blur32_tmp_new',
        'deit_blur0-16_tmp_fix_bug',  # this is instead 'deit_blur0-16_tmp' which stopped before training ended (5/5/25)
        'deit_blur0-32_tmp_new', 'deit_blur16-32_tmp',
        # Repetitions of the RandBlur jobs:
        'deit_blur0-16_rep', 'deit_blur16-32_rep',
        # models saved in 'out':
        'original', 'deit_blur4', 'deit_blur8', 'deit_blur16', 'deit_blur32', 'deit_blur0-32_tmp', 'deit_blur4_rep'
    ]
    blur = 0  # input blur
    n_patches = 14  # property of deit (14 patches in each row/column -> total 196 patches).
    patch_size = 16

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

    filename = osp.join(f'../Attention_Analysis/from_attention_distances/'
                        f'all_models_distances_{img_cat}_{img_lbl}_{len(model_names)}models.pkl')

    filename = filename.replace('.pkl', f'input_blur{blur}.pkl') if blur else filename

    if not osp.isfile(filename):
        # if osp.isfile(filename.replace(f'_{len(model_names)}models', '')):  # old version, without # of models
        #     # Check if all models are in dictionary:
        #     with open(filename, "rb") as file:
        #         all_models_distances = pickle.load(file)
        #
        # else:
            start_from_saved_data_when_possible = False
    else:
        with open(filename, "rb") as file:
            all_models_distances = pickle.load(file)

    if start_from_saved_data_when_possible:
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
                '/home/projects/bagon/ilanaveh/code/Transformers/deit/', model_out_dict[mdl], mdl) \
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
                if img_name == 'n04479046_15.JPEG':
                    pass
                img_full_pth = osp.join(img_pth, img_dataset, img_cat, img_name)
                original_image, input_tensor = load_and_preprocess_img(img_full_pth, blur, show_im_with_blur=False)

                # -------------------------------
                # 3. Forward Pass
                # -------------------------------
                with torch.no_grad():
                    out = model(input_tensor)

                for lyr in layer_indices:
                    # get attention for each layer, mean over heads:
                    lyr_attn = np.squeeze(model.blocks[lyr].attn.last_attn).mean(0)  # [ntokens=197, ntokens=197]

                    # Remove [CLS] token (which isn't relevant for distances:
                    lyr_attn_no_CLS = lyr_attn[1:, 1:]  # [ntokens=196, ntokens=196]

                    # Insert to dictionary:
                    all_models_attn[lyr][mdl or 'original'] = lyr_attn_no_CLS

                    # Get distances for each token (mostly based on Andrey (Projects\Variable_Resolution_DETR\Programming\
                    #   detr_var\EXPERIMENTS\attention_map_graph_generator\objects\graph_plotter.py):

                    # Initialize a list to store all scaled distances for layer (one for each token)
                    all_scaled_distances = []

                    for token_idx in range(lyr_attn_no_CLS.shape[0]):  # 0-195 (0 is not [CLS] token, since it was removed)
                        if token_idx == 47:  # for image n04479046_15, this is left eye.
                            pass
                        token_attn = lyr_attn_no_CLS[token_idx, :]

                        if add_thresh:
                            token_attn[token_attn < .001] = 0

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
                        all_scaled_distances.append(scaled_distances.sum())  # mean over tokens (use sum since we had softmax)

                    # Compute the average scaled distance
                    avg_scaled_distance = np.mean(all_scaled_distances)
                    all_models_distances[lyr][mdl].append(avg_scaled_distance)

        if save_file:
            with open(filename, 'wb') as file:
                pickle.dump(all_models_distances, file)

    # plot:

    if sep_panels:
        fig, subplots = plt.subplots(2, 5)  # assuming 10 blur levels, change if needed
        fig.set_size_inches([10.67, 6])
        subplots = subplots.flatten()
        # first, get list of blur-levels:
        blur_level_inds = dict()
        i = 0
        for mdl in model_names:
            if 'original' in mdl:
                blur_level = 'blur0'
            else:
                blur_level = next((blur for blur in color_map.keys() if blur in mdl), None)
            if blur_level and (blur_level not in blur_level_inds.keys()):
                blur_level_inds[blur_level] = i
                i += 1

        # Plot each model's attention distances:
        for mdl in model_names:
            means = [np.mean(all_models_distances[layer][mdl]) for layer in layer_indices]
            stds = [np.std(all_models_distances[layer][mdl]) for layer in layer_indices]

            # Get subplot index:
            if 'original' in mdl:
                blur_level = 'blur0'
            else:
                blur_level = next((blur for blur in color_map.keys() if blur in mdl), None)

            mdl_i = blur_level_inds[blur_level]
            ax = subplots[mdl_i]

            ax.errorbar(layer_indices, means, yerr=stds, marker='o', linestyle='-')

        # Set axis properties:
        for blur_level, ax_i in blur_level_inds.items():
            ax = subplots[ax_i]
            ax.set_ylim([40, 130])
            ax.set_title(blur_level)
            if (ax_i != 0) & (ax_i != 5):
                ax.set_yticklabels([])
            else:
                ax.set_ylabel('Attention distance (px)')
            ax.set_xticks(np.arange(0, len(layer_indices), 2))
            if ax_i < 5:
                ax.set_xticklabels([])
            elif ax_i == 7:
                ax.set_xlabel('Layer')
            ax.grid(axis='y')

        plt.suptitle(f"{img_cat} ({img_lbl})\nInput Blur: {blur}")
        plt.tight_layout()

    else:
        # Initialize a set to keep track of which blur levels have been added to the legend
        legend_added = set()
        for mdl in model_names:
            means = [np.mean(all_models_distances[layer][mdl]) for layer in layer_indices]
            stds = [np.std(all_models_distances[layer][mdl]) for layer in layer_indices]
            color = get_color_for_model(mdl)
            if 'original' in mdl:
                blur_level = 'blur0'
            else:
                blur_level = next((blur for blur in color_map.keys() if blur in mdl), None)
            if blur_level and (blur_level not in legend_added):
                plt.plot([], [], color=color, label=blur_level)  # Add empty plot for legend
                legend_added.add(blur_level)

            plt.errorbar(layer_indices, means, yerr=stds,
                         marker='o', linestyle='-', color=color)

        plt.xticks(layer_indices)
        plt.xlabel('Layer')
        plt.ylabel('Attention Distance (px)')
        plt.title(f"{img_cat} ({img_lbl})\nInput Blur: {blur}")
        plt.legend()
        fig = plt.gcf()
        fig.set_size_inches([7.2, 4.75])
        plt.tight_layout()

    if save_fig:
        figname = filename.replace('.pkl', '_sep_panels.png') if sep_panels else filename.replace('pkl', 'png')
        plt.savefig(figname)

    print('done')


if __name__ == "__main__":
    main()
