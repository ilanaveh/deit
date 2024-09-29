# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

from PIL import ImageFilter  # for GaussianBlur
import random  # for GaussianBlurRand
import matplotlib.pyplot as plt  # for saving example images.
import numpy as np  # for saving example images.

class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes

    return dataset, nb_classes


def build_transform(is_train, args):
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


class GaussianBlur(object):
    """
    Apply Gaussian blur filter with the given sigma to the input PIL Image.
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
        img = img.filter(ImageFilter.GaussianBlur(
            radius=self.sigma))

        return img

    def __repr__(self):
        return self.__class__.__name__ + '(sigma={0})'.format(self.sigma)


class GaussianBlurRand(object):
    """
    Apply Gaussian blur filter to the input PIL Image, with a rondom choice between self.sigma_min-self.sigma_max.
    if no sigma_max is given (or if sigma_min = sigma_max) -> same as regular GaussianBlur.
    Taken from: DeepLabv3FineTuning-disClasses/pretraining_resnet/pretrain_resnet_var_blurs.py.
    Args:
        sigma_min (int): Desired Gaussian blur level sigma / lower bound
        sigma_max (int; optional): Upper bound.
   """

    def __init__(self, sigma_min=0, sigma_max=None):
        assert isinstance(sigma_min, int)
        self.is_range = bool(sigma_max) & (sigma_min != sigma_max)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, img, return_blur=False):
        """
        Args:
            img (PIL Image): Image to be scaled.
            return_blur (bool): Whether to return the chosen blur sigma.
        Returns:
            PIL Image: Rescaled image.
            if return_blur=True: also return the chosen blur sigma.
        """

        radius = random.randint(self.sigma_min, self.sigma_max) if self.is_range else self.sigma_min
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))
        if return_blur:
            return img, radius
        else:
            return img

    def __repr__(self):
        if self.is_range:
            return self.__class__.__name__ + '(sigma={}-{})'.format(self.sigma_min, self.sigma_max)
        else:
            return self.__class__.__name__ + '(sigma={})'.format(self.sigma_min)


def add_blur_transform(ori_transforms, blur, blur_max=None, use_custom_compose=False):
    """
    Add GaussianBlur / GaussianBlurRand transform to the Transforms Compose object.
    :param ori_transforms: Compose object with original sequence of transforms
    :param blur: Blur sigma
    :param blur_max: (optional) use GaussianBlurRand with a range of sigmas to choose from (from blur to blur_max).
    :param use_custom_compose: (bool) whether to use CustomCompose (that logs blurs) instead of regular Compose.
    :return: New Compose object, with the blur-transform added to beginning.
    """

    # Original list of transforms (extract from Compose)
    transform_list = ori_transforms.transforms

    # Prepend the blur transform
    if blur_max:
        updated_transform_list = [GaussianBlurRand(blur, blur_max)] + transform_list
    else:
        updated_transform_list = [GaussianBlur(blur)] + transform_list

    # Create a new Compose object with the updated list
    if use_custom_compose:
        return CustomCompose(updated_transform_list)
    else:
        return transforms.Compose(updated_transform_list)


class BlurDataset(ImageFolder):
    """
    Based on original ImageFolder, but return the applied blur level in addition to the image.
    """
    def __init__(
            self,
            root,
            transform,
            return_blur,
            chosen_imgs_lst=[],
            save_imgs_pth=''):

        super().__init__(root, transform=transform)
        self.return_blur = return_blur
        self.chosen_imgs_lst = chosen_imgs_lst
        self.save_imgs_pth = save_imgs_pth

    # Override the __getitem__ method, s.t. the blur level applied for each image is returned:
    def __getitem__(self, index):

        # 1. Copy the original ImageFolder getitem method, but add applied_blur as output if GaussianBlurRand is used:
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            if self.return_blur:
                sample, applied_blur = self.transform(sample)
            else:
                sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        # Save example images:
        im_nm = path.split('/')[-1].split('.JPEG')[0]
        if self.chosen_imgs_lst and (im_nm in self.chosen_imgs_lst) and self.save_imgs_pth:
            im_save_nm = '{}_blur{}.png'.format(im_nm, applied_blur) if self.return_blur else '{}.png'.format(im_nm)
            if not os.path.isfile(os.path.join(self.save_imgs_pth, im_save_nm)):
                sample_norm = (sample-sample.min()) / (sample.max()-sample.min()) * 255
                sample_numpy = np.array(sample_norm.permute(1, 2, 0)).astype('uint8')
                plt.imsave(os.path.join(self.save_imgs_pth, im_save_nm), sample_numpy)
                # For creating image with only Blur transform:
                if isinstance(self.transform.transforms[0], GaussianBlurRand) or \
                        isinstance(self.transform.transforms[0], GaussianBlur):
                    sample_for_blur = self.loader(path)
                    if self.return_blur:
                        blur_trans = GaussianBlur(applied_blur)
                    else:
                        blur_trans = self.transform.transforms[0]
                    sample_blurred = blur_trans(sample_for_blur)
                    sample_blur_numpy = np.array(sample_blurred)
                    plt.imsave(os.path.join(self.save_imgs_pth, im_save_nm.replace('.png', '_onlyBlur.png')),
                               sample_blur_numpy)

        # 2. return blur level, in addition to sample & target.
        if self.return_blur:
            return sample, target, applied_blur
        else:
            return sample, target


def build_dataset_blur(is_train, args, return_blur=False):
    """
    based on 'build_dataset', but calls BlurDataset instead of ImageFolder.
    """
    transform = build_transform(is_train, args)

    assert args.data_set == 'IMNET'  # assume using imagenet.

    root = os.path.join(args.data_path, 'train' if is_train else 'val')

    if args.chosen_imgs_pth and os.path.isfile(args.chosen_imgs_pth):
        with open(args.chosen_imgs_pth, "r") as file:
            list_json = file.read()
        chosen_imgs_lst = json.loads(list_json)
    else:
        chosen_imgs_lst = []
    dataset = BlurDataset(root, transform=transform, return_blur=return_blur, chosen_imgs_lst=chosen_imgs_lst,
                          save_imgs_pth=os.path.join(args.output_dir, args.model_name))
    nb_classes = 1000

    return dataset, nb_classes


class CustomCompose:
    """
    Based on torch's Compose (torchvision.transforms.transforms.Compose), but change __call__, s.t. it can receive the
    actual blur level used in GaussianBlurRand.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            if isinstance(t, GaussianBlurRand):
                img, applied_blur = t(img, return_blur=True)
            else:
                img = t(img)

        if any(isinstance(t, GaussianBlurRand) for t in self.transforms):
            return img, applied_blur
        else:
            return img

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string

