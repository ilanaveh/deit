"""
4/11/25
Evaluate performance of downstream-training models:
    1. Plot accuracies
    2. Breakdown of performance for different classes
"""
from deit.datasets import build_dataset
from deit.datasets import add_blur_transform
from deit.engine import evaluate
from types import SimpleNamespace
import torch
from timm.models import create_model
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

save_figs = False
save_dir = '/home/projects/bagon/ilanaveh/code/Transformers/deit/downstream_training/figures/from_downstream_eval'
model_blurs = ['0', '4', '8', '16', '0-4', '0-8', '0-16']
test_blurs = {'0': [0], '4': [4], '8': [8], '16': [16], '0-4': [4], '0-8': [8], '0-16': [16]}
args = SimpleNamespace(data_set="Affectnet", data_path='/home/projects/bagon/ilanaveh/data/AffectNet',
                       desired_classes=[0, 1, 2, 3, 4, 5, 6, 7], balance_clss=False, debug=False,
                       input_size=224, eval_crop_ratio=.875, device='cuda')
args.nb_classes = len(args.desired_classes)
class_label_dict = {0: 'Neutral', 1: 'Happy', 2: 'Sad', 3: 'Surprise',
                    4: 'Fear', 5: 'Disgust', 6: 'Anger', 7: 'Contempt'}
device = torch.device(args.device)
model_dir = '/home/projects/bagon/ilanaveh/code/Transformers/deit/downstream_training/out'
batch_size = 128

class_accs = {mdl: {c: None for c in args.desired_classes} for mdl in model_blurs}

for model_blur in model_blurs:
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    model_name = f'finetune_deit_model_blur{model_blur}_affectnet_blur{model_blur}_{args.nb_classes}cls_unbalanced'
    print(model_name)
    model = create_model(
        'deit_base_patch16_224',
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=0,
        drop_path_rate=0.1,
        drop_block_rate=None,
        img_size=args.input_size
    )

    model.to(device)
    checkpoint = torch.load(os.path.join(model_dir, model_name, 'checkpoint.pth'), map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    for test_blur in test_blurs[model_blur]:
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Dataloader ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        dataset_val, _ = build_dataset(is_train=False, args=args)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=int(1.5 * batch_size),
            num_workers=10,
            pin_memory=False,
            drop_last=False
        )

        if test_blur:
            data_loader_val.dataset.transform = add_blur_transform(data_loader_val.dataset.transform, test_blur)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Evaluate ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        test_stats, breakdown = evaluate(data_loader_val, model, device,
                                         return_breakdown=True, des_classes=args.desired_classes)
        print(f"Accuracy of the network on the {len(dataset_val)} test images with blur "
              f"{test_blur}: {test_stats['acc1']:.1f}%")

        for c in args.desired_classes:
            tot_samples_in_class = np.sum([s for s in breakdown[c].values()])
            class_accs[model_blur][c] = breakdown[c][c] / tot_samples_in_class

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Plot ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        conf_matrix = pd.DataFrame(breakdown).T  # .T transposes so rows = true labels, cols = predicted

        # rename both rows (index) and columns using the mapping
        conf_matrix = conf_matrix.rename(index=class_label_dict, columns=class_label_dict)

        # plot
        ttl = f"Model Trained with Blur {model_blur}, Test Blur: {test_blur}"

        f = plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(ttl)

        if save_figs:
            f.savefig(os.path.join(save_dir, f"Conf mat {ttl.replace(':', '')}.jpg"))

print('done')
