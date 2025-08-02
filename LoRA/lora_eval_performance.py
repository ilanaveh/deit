"""
02/08/25
For meeting with Udi and Danny.
Evaluate performance of different models with different test-blurs.
Based on main_lora.py, with changes from deit_eval_performance.py
"""
import sys
sys.path.append("/home/projects/bagon/ilanaveh/code/Transformers")

import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import deit.utils as utils
from timm.models import create_model
import numpy as np
from pathlib import Path
from timm.data import Mixup
from timm.utils import NativeScaler, get_state_dict, ModelEma
from timm.optim import create_optimizer
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from deit.datasets import build_dataset, add_blur_transform, build_dataset_blur
from Code_from_Liel.modeling_finetune import inject_lora_vit
import loralib as lora
import time
from deit.engine import train_one_epoch, evaluate
import json
import datetime
from tensorboardX import SummaryWriter
from collections import Counter


def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=150, type=int)  # Change to 30, as in Liel's code.
    parser.add_argument('--bce-loss', action='store_true')
    parser.add_argument('--unscale-lr', action='store_true')

    parser.add_argument('--debug', default=False, type=bool, help='build small dataset if debugging.')

    # Model parameters
    parser.add_argument('--model', default='deit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--load_pretrained', action='store_true', help='whether to load pretrained deit model.')

    # Instead of deit_model, add lora_model params - for starting from a trained lora model:
    parser.add_argument('--lora_model_dir', default='out', type=str, help='directory (within "deit/LoRA") of lora mdl')
    parser.add_argument('--lora_model_name', default=None, type=str,
                        help='model name, or None for original (pretrained or not, according to args.load_pretrained)')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # * --- This is from Liel, currently unused in my code:
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
            weight decay. We use a cosine schedule for WD and using a larger decay by
            the end of training improves performance for ViTs.""")
    # * ---
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    # Removed arguments related to LR decay (they don't exist in Liel's code)

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.3, metavar='PCT',
                        help='Color jitter factor (default: 0.3)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    parser.add_argument('--train-mode', action='store_true')
    parser.add_argument('--no-train-mode', action='store_false', dest='train_mode')
    parser.set_defaults(train_mode=True)

    parser.add_argument('--ThreeAugment', action='store_true')  # 3augment

    parser.add_argument('--src', action='store_true')  # simple random crop

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
                        help='Name of teacher model to train (default: "regnety_160"')
    parser.add_argument('--teacher-path', type=str, default='')
    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")

    # * Cosub params
    parser.add_argument('--cosub', action='store_true')

    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--attn-only', action='store_true')

    # Dataset parameters
    parser.add_argument('--data-path', default='/home/projects/bagon/ilanaveh/data/AffectNet', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='Affectnet', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19', 'Affectnet'],
                        type=str, help='dataset path, changed to affectnet for finetuning')
    parser.add_argument('--desired_classes', default=[1, 6], type=int, nargs='+',
                        help='0: Neutral, 1: Happiness, 2: Sadness, 3: Surprise, 4: Fear, 5: Disgust, 6: Anger, '
                             '7: Contempt, 8: None, 9: Uncertain, 10: No-Face.ToDo: decide which classes I want.')
    parser.add_argument('--balance_clss', default=True, type=bool,
                        help='whether to take the same number of images from each class (relevant for Affectnet)')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='/home/projects/bagon/ilanaveh/code/Transformers/deit/LoRA/out',
                        help='path where to save, empty for no saving')
    parser.add_argument('--model_name', default='deit_lora',
                        help='sub-directory for saving checkpoint')

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='/home/projects/bagon/ilanaveh/code/Transformers/deit/LoRA/out',
                        help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--eval-crop-ratio', default=0.875, type=float, help="Crop ratio for evaluation")
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--distributed', action='store_true', default=False, help='Enabling distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # blur parameter
    parser.add_argument('--blur', default=0, type=int, help='Sigma of the Gaussian blur')
    parser.add_argument('--blur_max', default=None, type=int, help='For Variable-Blur training: max sigma')
    parser.add_argument('--blur_for_tb_log', default=None, type=int,
                        help='For Variable-Blur training: blur to show in TB logging (validation). '
                             'Default (if None): blur_max')

    parser.add_argument('--chosen-imgs-pth',
                        default='/home/projects/bagon/ilanaveh/code/Transformers/deit/ims2save_for_logging/ims2save.txt',
                        type=str, help='path to text file with list of images to save as examples from each run.')

    # suffix for model name
    parser.add_argument('--suf', default='', type=str, help='suffix for model name (would be added with "_"')
    return parser


def main(args):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)
    args.debug = False

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # Change args.desired_classes according to lora_model_name:
    args.desired_classes = [1, 2, 3, 6] if '_4cls' in args.lora_model_name else [1, 6]

    n_cls = len(args.desired_classes)

    print(f"Creating dataset: {args.data_set}, with {n_cls} classes: {args.desired_classes}")
    print("Validation Dataset:")
    dataset_val, args.nb_classes = build_dataset(is_train=False, args=args)

    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    if args.blur:
        # Add blur transform to val dataloader:
        data_loader_val.dataset.transform = add_blur_transform(data_loader_val.dataset.transform, args.blur)

    crt_mdl_msg = f"Creating model: {args.model}"
    crt_mdl_msg = crt_mdl_msg + " (with pretrained weights)" if args.load_pretrained else crt_mdl_msg + " (untrained)"
    print(crt_mdl_msg)

    model = create_model(
        args.model,
        pretrained=args.load_pretrained,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        img_size=args.input_size
    )

    model.to(device)  # need this for model_ema (so it would be on cuda). Later, add again to move lora layers to cuda.

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    # * --- Add code for LoRA (Based on Liel's 'run_class_finetuning.py', with some changes)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total: {total_params:,} | Trainable: {trainable_params:,}")

    # inject_lora_attention(model) * --- from Liel
    model = inject_lora_vit(model, r=64, alpha=32)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)  # ToDo: should update this also later?

    # * --- Make only lora trainable (from git):
    lora.mark_only_lora_as_trainable(model)

    # Enable norm & bias layers * --- from Liel
    for name, param in model.named_parameters():
        if 'norm' in name:
            param.requires_grad = True

    # Enable head (classification layer) * --- from Liel
    for name, param in model.named_parameters():
        # if 'head' in name or 'fc' in name:
        if name in ['head.weight', 'head.bias']:
            param.requires_grad = True

    # Validate number of params decreased * --- from Liel
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total: {total_params:,} | Trainable: {trainable_params:,}")

    model.to(device)  # add second time, for moving lora params to cuda.

    if args.lora_model_name:
        lora_mdl_pth = os.path.join('/home/projects/bagon/ilanaveh/code/Transformers/deit/LoRA',
                                    args.lora_model_dir, args.lora_model_name, 'checkpoint.pth')
        if os.path.isfile(lora_mdl_pth):
            checkpoint = torch.load(lora_mdl_pth, map_location='cpu')
            print(f"Loading model: {args.lora_model_name} (epoch: {checkpoint['epoch']})")
            model.load_state_dict(checkpoint['model'])
        else:
            print(f"No such model {args.lora_model_name}")
            return None

    test_stats = evaluate(data_loader_val, model, device)
    print(f"Accuracy of the model {args.lora_model_name} on the {len(dataset_val)} test images with blur "
          f"{args.blur}: {test_stats['acc1']:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    # Add lora arguments:
    parser.add_argument('--use_lora', default=True, type=bool, help='use lora finetuning')

    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)

