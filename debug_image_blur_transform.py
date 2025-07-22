import copy

from PIL import Image
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
from types import SimpleNamespace
from datasets import add_blur_transform
import numpy as np
import os
import matplotlib.pyplot as plt
from datasets import GaussianBlur
import pickle
import itertools
from timm.data.auto_augment import RandAugment


save_stats = False
save_ims = False
save_ims_combs = True
perform_trans_before_randAug = False

im_name =  'n03325584_54'  #  'n03325584_54',  'n03124043_49'
im_path = os.path.join('/home/projects/bagon/shared/imagenet/train', im_name.split('_')[0], im_name+'.JPEG')
out_pth = os.path.join('/home/projects/bagon/ilanaveh/code/Transformers/deit/out_from_debug_image_blur_transform',
                        im_name)
out_after_trans_pth = os.path.join(out_pth, 'ims_after_trans_diff_blurs')
out_combs_pth = os.path.join(out_pth, 'ims_diff_op_combs')

# In the pickle file - list of operations that RandAugment chooses from - saved self.ops from
# timm.data.auto_augment.RandAugment.__call__ :
pkl_pth = 'out_from_debug_image_blur_transform/save_ops_list/ops_list.pkl'

# From torchvision.datasets.folder.pil_loader:
with open(im_path, "rb") as f:
    img = Image.open(f)
    sample = img.convert("RGB")

range_original = [np.min(np.array(sample)), np.max(np.array(sample))]
if save_ims:
    plt.imsave(os.path.join(out_after_trans_pth, 'original_image.png'), np.array(sample))

# Create 'args' similar to args in main_tmp.py, for build_transform:
args = SimpleNamespace()
args.input_size = 224
args.color_jitter = .3
args.aa = 'rand-m9-mstd0.5-inc1'
args.train_interpolation = 'bicubic'
args.reprob = .25
args.remode = 'pixel'
args.recount = 1
args.eval_crop_ratio = .875


def get_RandAug_combs(pkl_pth):
    with open(pkl_pth, 'rb') as f:
        ops_list = pickle.load(f)

    all_op_pairs = list(itertools.combinations(ops_list, 2))
    all_possible_op_combs = []
    for op_pair in all_op_pairs:
        for op in op_pair:
            op.prob = 1
        all_possible_op_combs.append(op_pair)
        if not (op_pair[0],) in all_possible_op_combs:
            all_possible_op_combs.append((op_pair[0],))
        if not (op_pair[1],) in all_possible_op_combs:
            all_possible_op_combs.append((op_pair[1],))

    return all_possible_op_combs


def build_transform(is_train, args):
    "Based on deit.datasets.build_transform"
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int(args.input_size / args.eval_crop_ratio)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)

# From deit.datasets.build_dataset_blur:
is_train = True
transform = build_transform(is_train, args)

blurs = [0, 4, 8, 16, None]
niters = 20

if save_stats:
    with open(os.path.join(out_after_trans_pth, "log.txt"), 'a') as f:
        f.write('Original image value range:\n')
        f.write(f'{range_original}\n\n')

for blur in blurs:
    if blur is not None:
        blur_trans = GaussianBlur(blur)

    for i in range(niters):
        im_save_nm = f'im_trans_blur{blur}_{i}.png'
        # Based on code in deit.datasets.BlurDataset.__getitem__ (but break down transform appliance for logging):

        # Perform Blur transform:
        if blur is not None:
            sample_trans = blur_trans(sample)
            range_after_blur = [np.min(np.array(sample_trans)), np.max(np.array(sample_trans))]

            # Perform all other transforms:
            sample_trans = transform(sample_trans)
        else:
            sample_trans = transform(sample)

        range_final = [np.min(np.array(sample_trans)), np.max(np.array(sample_trans))]

        if save_ims:
            sample_trans_norm = (sample_trans - sample_trans.min()) / (sample_trans.max() - sample_trans.min()) * 255
            sample_trans_numpy = np.array(sample_trans_norm.permute(1, 2, 0)).astype('uint8')
            plt.imsave(os.path.join(out_after_trans_pth, im_save_nm), sample_trans_numpy)

        if save_stats:
            with open(os.path.join(out_after_trans_pth, "log.txt"), 'a') as f:
                f.write(f'blur{blur}_{i}:\n')
                if blur is not None:
                    f.write('After blur (before regular transforms):\n')
                    f.write(f'{range_after_blur}\n')

                f.write('After all transforms:\n')
                f.write(f'{range_final}\n')

                f.write('\n')

    if save_stats:
        with open(os.path.join(out_after_trans_pth, "log.txt"), 'a') as f:
            f.write('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n')


# Save images after each transform, with different combinations of the RandAugment transform:
for blur in blurs:
    sample_new = copy.deepcopy(sample)

    # Get all possible tuples of operations that can be performed by RandAugmentation:
    all_op_combs = get_RandAug_combs(pkl_pth)

    # Create directory for output images:
    if save_ims_combs:
        if perform_trans_before_randAug:
            blur_out_dir = os.path.join(out_combs_pth, f'Blur_{blur}')
        else:
            blur_out_dir = os.path.join(out_combs_pth, 'only_blur_and_randaug', 'new', f'Blur_{blur}')

        if not os.path.isdir(blur_out_dir):
            os.mkdir(blur_out_dir)

    # Perform blur transform
    if blur is not None:
        blur_trans = GaussianBlur(blur)
        sample_new = blur_trans(sample_new)

        if save_ims_combs:
            im_save_nm = f'Blur_{blur}_after_blur_transform.png'
            sample_new.save(os.path.join(blur_out_dir, im_save_nm))

    # Perform all other transforms:
    for t in transform.transforms:
        if not isinstance(t, RandAugment):
            if perform_trans_before_randAug:
                aug_name = t.__repr__().split('(')[0]
                sample_new = t(sample_new)

                if save_ims_combs:
                    im_save_nm = f'Blur_{blur}_after_{aug_name}.png'
                    sample_new.save(os.path.join(blur_out_dir, im_save_nm))
            else:
                continue

        else:
            for op_tpl in all_op_combs:
                sample_with_ops = copy.deepcopy(sample_new)
                op_tpl_repr = ''
                for op in op_tpl:
                    op_tpl_repr += '_'
                    op_tpl_repr += op.name
                    sample_with_ops = op(sample_with_ops)

                if save_ims_combs:
                    im_save_nm = f'Blur_{blur}_after_RandAugment{op_tpl_repr}.png'
                    sample_with_ops.save(os.path.join(blur_out_dir, im_save_nm))

            break





