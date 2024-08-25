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

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.
        Returns:
            PIL Image: Rescaled image.
        """

        radius = random.randint(self.sigma_min, self.sigma_max) if self.is_range else self.sigma_min
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))

        return img


def add_blur_transform(ori_transforms, blur, blur_max=None):
    """
    Add GaussianBlur / GaussianBlurRand transform to the Transforms Compose object.
    :param ori_transforms: Compose object with original sequence of transforms
    :param blur: Blur sigma
    :param blur_max: (optional) use GaussianBlurRand with a range of sigmas to choose from (from blur to blur_max).
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
    return transforms.Compose(updated_transform_list)
