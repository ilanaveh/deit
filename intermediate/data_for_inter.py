import os
import os.path as osp
import pickle
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
from skimage import io


def get_data_from_pkl(pkl_pth):
    all_face_images, all_imagepath, all_emotion = pickle.load(open(pkl_pth, "rb"))

    valid_img_inds = set([idx for idx, element in enumerate(all_emotion) if (element != 'N')])
    gt_labels = np.asarray([all_emotion[i] for i in valid_img_inds]) == 'H'

    return all_face_images, all_imagepath, gt_labels


def get_sample_ims(dataset, n_ims_each_ds, transform=None, start_i=0):
    """
    Used by:
     - evaluate_models/visualize_bbox.py
     - intermediate/plot_activation_maps.py
    :param datasets: string, from: ['coco', 'affectnet', 'imagenet256', 'imagenet512']
    :param n_ims_each_ds: how many images from each dataset (int)
    :param start_i: index to start from when choosing images (useful when more exmaples are needed)
    :return: ims_dict, ori_ims_dict, im_names_dict
    4/4/24 Changed input to single dataset, and output from dicts with multiple datasets to lists of a specific dataset.
    """

    # standard PyTorch mean-std input image normalization (skip resize for now, since affectnet images are small).
    if not transform:
        transform = T.Compose([
            T.Resize(800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    # images from coco & affectnet:
    paths = {'coco': '/home/projects/bagon/shared/coco/val2017',
             'affectnet': '/home/projects/bagon/ilanaveh/data/AffectNet/val_set/images'}

    if 'imagenet' not in dataset:
        im_names = os.listdir(paths[dataset])[: n_ims_each_ds]

        # Original images (before transform):
        ori_ims = [Image.open(osp.join(paths[dataset], im_names[i])) for i in range(n_ims_each_ds)]

    else:  # imagenet is different because there are sub-directories within the general dir.
        imagenet_pth = f'/home/projects/bagon/shared/{dataset}/val'
        imagnet_all_pths_lst = [osp.join(imagenet_pth, os.listdir(imagenet_pth)[i])
                                for i in range(start_i, start_i+n_ims_each_ds)]
        im_names = [os.listdir(imagnet_all_pths_lst[i])[0] for i in range(n_ims_each_ds)]
        ori_ims = [Image.open(osp.join(imagnet_all_pths_lst[i], im_names[i])) for i in range(n_ims_each_ds)]

    # mean-std normalize the input image (batch-size: 1 --> unsqueeze first dimension)
    ims = [transform(im).unsqueeze(0) for im in ori_ims]

    return ims, ori_ims, im_names


class AttsDataset(Dataset):
    """
    From DeepLabv3FineTuning-disClasses/intermediate/train_on_atts/Data.py
    """

    def __init__(self, csv_file, root_dir, transform=None, inds=pd.DataFrame(), remove_inds=True, split_ratio=1,
                 return_im_name=False, condition='emotion'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            inds: inds of dataframe to use/ remove for current dataset.
            remove_inds: whether the above inds should be used or removed (e.g. for valset, remove the inds of trainset)
            split_ratio:
                if <= 1: what proportion of the images to use for current dataset.
                if > 1: number of images to use for current dataset.
            return_im_name: whether the sample returned by get_item should include the image name (or only image,target)
            condition: choose between 'emotion' / 'age' / 'gender'
        """
        self.anns_df = pd.read_csv(csv_file, index_col=0)
        self.root_dir = root_dir
        self.transform = transform
        self.return_im_name = return_im_name
        self.condition = condition

        # leave only relevant condition:
        self.anns_df = self.anns_df[['im_name', self.condition]]

        #     Convert labels to indexes:
        cls = np.unique(self.anns_df[self.condition])
        for (i, c) in enumerate(cls):
            self.anns_df[self.condition] = self.anns_df[self.condition].replace([c], i)

        if not inds.empty:
            if remove_inds:
                self.anns_df = self.anns_df.drop(inds)
            else:
                self.anns_df = self.anns_df.iloc[inds]

        else:
            if split_ratio < 1:
                self.anns_df = self.anns_df.sample(frac=split_ratio)
            elif split_ratio > 1:
                self.anns_df = self.anns_df.sample(frac=(split_ratio / self.anns_df.shape[0]))

        self.anns_df = self.anns_df.sample(frac=1)  # shuffle row order.

    def __len__(self):
        return len(self.anns_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.anns_df.iloc[idx, 0])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        anns = self.anns_df.iloc[idx, 1]
        # anns = np.array([anns])
        if self.return_im_name:
            sample = [image, anns, self.anns_df.iloc[idx, 0]]
        else:
            sample = [image, anns]

        return sample


class AttsDatasetFixed(Dataset):
    """
    From DeepLabv3FineTuning-disClasses/intermediate/train_on_atts/Data.py
    """

    def __init__(self, csv_file, root_dir, transform=None, phase='train', return_im_name=False, condition='emotion'):
        """
        Fixed division to 'train' and 'val', according to info in the csv file (there should be a 'phase' column).

        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            return_im_name: whether the sample returned by get_item should include the image name (or only image,target)
            condition: choose between 'emotion' / 'age' / 'gender'
        """
        self.anns_df = pd.read_csv(csv_file, index_col=0)
        self.root_dir = root_dir
        self.transform = transform
        self.return_im_name = return_im_name
        self.condition = condition
        self.phase = phase

        # leave only relevant condition:
        self.anns_df = self.anns_df[['im_name', self.condition, 'phase']]

        # Convert labels to indexes:
        cls = np.unique(self.anns_df[self.condition])
        for (i, c) in enumerate(cls):
            self.anns_df[self.condition] = self.anns_df[self.condition].replace([c], i)

        # Take only the images of the current phase (train/val):
        self.anns_df = self.anns_df.loc[self.anns_df['phase'] == self.phase]

        # shuffle row order:
        self.anns_df = self.anns_df.sample(frac=1)

    def __len__(self):
        return len(self.anns_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.anns_df.iloc[idx, 0])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        anns = self.anns_df.iloc[idx, 1]
        # anns = np.array([anns])
        if self.return_im_name:
            sample = [image, anns, self.anns_df.iloc[idx, 0]]
        else:
            sample = [image, anns]

        return sample


if __name__ == '__main__':
    data_pth = '/home/labs/waic/ilanaveh/data'
    affectnet_pth = osp.join(data_pth, 'AffectNet', 'val_set')
    pkl_pth = osp.join(affectnet_pth, 'AffectNet_dataset_896.pkl')

    ims, im_pth, lbls = get_data_from_pkl(pkl_pth)
