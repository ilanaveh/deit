"""
17/7/25
Save attention maps in Andrey's format, so I can try Andrey's distance code on them.
Format: One map for each image (and each model), of shape: [lyrs, w, h, w, h]
In my case (deit), w=h=14 (number of patches in each dimension)

Code here is based on attention_distances.py
"""
from attention_distances import imagenet_idx_to_lbl, imagenet_class_to_idx, model_out_dict
import numpy as np
import os
import os.path as osp
from helper_functions import load_and_preprocess_img, get_model_with_attn
import torch

save_maps = True
save_dir = 'from_save_attn_maps'

blur = 0  # input blur
n_patches = 14  # property of deit (14 patches in each row/column -> total 196 patches).

layer_indices = np.arange(12)
# model_names = ['deit_blur0_tmp_new', ]
model_names = ['deit_blur16_tmp_new', 'deit_blur0-16_tmp_fix_bug']

# DEfine model types to correspond to Andrey's terminology:
model_type_dict = {
    'deit_blur0_tmp_new': 'baseline',
    'deit_blur16_tmp_new': 'equiconst',
    'deit_blur0-16_tmp_fix_bug': 'variable'
}

limit_n_ims = 100

# Go over all images in 'trenchcoat' category in validation set:
img_pth = '/home/projects/bagon/shared/imagenet'
img_dataset = 'train'
# img_cat = 'n09472597'  # volcano
# img_cat = 'n01532829'  # house finch
img_cat = 'n04479046'  # trenchcoat
# img_cat = 'n07697313'  # cheeseburger

img_lbl = imagenet_idx_to_lbl[f"{imagenet_class_to_idx[img_cat]}"]
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

        # Initialize tensor for attention maps (shape [l,w,h,w,h] = [12,14,14,14,14]:
        img_attn_maps = torch.zeros((len(layer_indices), n_patches, n_patches, n_patches, n_patches))

        # Name for saving file:
        attention_map_name = img_name.split('.')[0].replace('_', '-') + "_" + model_type_dict[mdl] + ".pt"
        attention_map_save_path = os.path.join(save_dir, attention_map_name)

        for lyr in layer_indices:
            # get attention for each layer, mean over heads:
            lyr_attn = np.squeeze(model.blocks[lyr].attn.last_attn).mean(0)  # [ntokens=197, ntokens=197]

            # Remove [CLS] token (which isn't relevant for distances:
            lyr_attn_no_CLS = lyr_attn[1:, 1:]  # [ntokens=196, ntokens=196]

            # Get distances for each token (mostly based on Andrey (Projects\Variable_Resolution_DETR\Programming\
            #   detr_var\EXPERIMENTS\attention_map_graph_generator\objects\graph_plotter.py):

            # Initialize a list to store all scaled distances for layer (one for each token)
            all_scaled_distances = []

            for token_idx in range(lyr_attn_no_CLS.shape[0]):  # 0-195 (0 is not [CLS] token, since it was removed)
                if token_idx == 47:  # for image n04479046_15, this is left eye.
                    pass
                token_attn = lyr_attn_no_CLS[token_idx, :]

                # Renormalize (normalization was lost, since we removed [CLS] token):
                token_attn = token_attn / token_attn.sum()
                assert np.round(token_attn.sum(), 3) == 1

                # Get attention grid of token. Shape: [14, 14]
                token_attn_grid = token_attn.reshape(n_patches, n_patches).detach().cpu().numpy()

                patch_x, patch_y = divmod(token_idx, n_patches)

                img_attn_maps[lyr, :, :, patch_y, patch_x] = torch.tensor(token_attn_grid)

        if save_maps:
            torch.save(img_attn_maps, attention_map_save_path)



