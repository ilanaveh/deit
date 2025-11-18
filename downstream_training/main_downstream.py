"""
22/10/25
Downstream training of deit model on Affectnet.
Based on deit/main_tmp.py

Changes:
    - Dataset: Affectnet
"""


# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import sys
sys.path.append("/home/projects/bagon/ilanaveh/code/Transformers")

import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from deit.datasets import build_dataset
from deit.engine import train_one_epoch, evaluate
from deit.losses import DistillationLoss
from deit.samplers import RASampler
from deit.augment import new_data_aug_generator
from deit.datasets import add_blur_transform

import deit.models
import deit.models_v2

import deit.utils as utils
import os
from tensorboardX import SummaryWriter


def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--bce-loss', action='store_true')
    parser.add_argument('--unscale-lr', action='store_true')

    # Model parameters
    parser.add_argument('--model', default='deit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--load_pretrained', action='store_true', help='whether to load pretrained deit model.')

    parser.add_argument('--deit_model_dir', default='out', type=str, help='directory (within "deit") of deit model')
    parser.add_argument('--deit_model_name', default=None, type=str,
                        help='model name, or None for original (pretrained or not, according to args.load_pretrained)')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
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

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

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
    # 18/11/25: disable mixup
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0. (default: 0)')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Cosub params
    parser.add_argument('--cosub', action='store_true')

    # * Finetuning params
    parser.add_argument('--attn-only', action='store_true')

    # Dataset parameters
    parser.add_argument('--data-path', default='/home/projects/bagon/ilanaveh/data/AffectNet', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='Affectnet', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19', 'Affectnet'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--desired_classes', default=[0, 1, 2, 3, 4, 5, 6, 7], type=int, nargs='+',
                        help='0: Neutral, 1: Happiness, 2: Sadness, 3: Surprise, 4: Fear, 5: Disgust, 6: Anger, '
                             '7: Contempt, 8: None, 9: Uncertain, 10: No-Face.ToDo: decide which classes I want.')
    parser.add_argument('--balance_clss', action='store_true',
                        help='whether to take the same number of images from each class (relevant for Affectnet)')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir',
                        default='/home/projects/bagon/ilanaveh/code/Transformers/deit/downstream_training/out',
                        help='path where to save, empty for no saving')
    parser.add_argument('--model_name', default='deit_downstream',
                        help='sub-directory for saving checkpoint')

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume',
                        default='/home/projects/bagon/ilanaveh/code/Transformers/deit/downstream_training/out',
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
                             'Default (if None): blur_max. '
                             'Currently only works with blur / blur_max (not other blurs in range)')

    # suffix for model name
    parser.add_argument('--suf', default='', type=str, help='suffix for model name (would be added with "_"')

    # For logging example images
    parser.add_argument('--ims2save_pth', default='/home/projects/bagon/ilanaveh/code/Transformers/deit/'
                                                  'ims2save_for_logging/ims2save.txt',
                        type=str, help='path to text file with list of images to save.')

    # For masking patches, according to facial landmarks:
    parser.add_argument('--select_patches', default=[], type=str, nargs='+',
                        help="Which facial-landmarks to use for including patches."
                             "Options: eyes, nose, mouth, eyebrows, outline. "
                             "Default: empty list -> no mask (use all patches)")
    parser.add_argument('--debug_mask', action='store_true', help='option to visualize patches that remain after mask')
    return parser


def main(args):
    utils.init_distributed_mode(args)

    print(args)

    device = torch.device(args.device)
    args.debug = torch.cuda.device_count() == 1

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    n_cls = len(args.desired_classes)
    n_ptch = len(args.select_patches)

    args.get_landmarks = bool(args.select_patches)  # for build_dataset
    # Change model name to format:
    #   "finetune_deit_model_blur{deit_model_training_blur}_affectnet_blur{affectnet_training_blur}_{suf}"
    # If starting from original pretrained deit (i.e. args.deit_model_name=None):
    #   "finetune_deit_model_original_affectnet_blur{affectnet_training_blur}_{suf}"
    # * If using more than 2 classes from affectnet, add "_{n}cls" before suf.
    deit_model_blur = args.deit_model_name.split('blur')[1].split('_')[0] if args.deit_model_name else ''
    args.model_name = f"finetune_deit_model_blur{deit_model_blur}" \
        if args.deit_model_name else "finetune_deit_model_original"
    args.model_name = args.model_name + '_affectnet_blur{}'.format(args.blur)
    args.model_name = args.model_name + '-{}'.format(args.blur_max) if args.blur_max else args.model_name
    args.model_name = args.model_name + '_{}cls'.format(n_cls) if (n_cls > 2) else args.model_name
    args.model_name = args.model_name + '_unbalanced' if not args.balance_clss else args.model_name
    args.model_name = args.model_name + '_{}ptch'.format(n_ptch) if bool(args.select_patches) else args.model_name
    args.model_name = args.model_name + '_{}'.format(args.suf) if args.suf else args.model_name
    args.model_name = args.model_name + '_db' if (torch.cuda.device_count() == 1) else args.model_name
    print(f"~~~\n{args.model_name}\n~~~")

    output_dir = Path(args.output_dir) / args.model_name
    output_dir.mkdir(parents=False, exist_ok=True)  # create output_dir if doesn't exist, alert if parent doesn't exist.

    # For adding mask to patches (according to facial landmarks):
    landmarks_dict = {
        'outline': list(range(0, 17)),  # 1-17 in illustration, inds: 0-16
        'eyebrows': list(range(17, 27)),  # 18-27 in illustration, inds: 17-26
        'nose': list(range(27, 36)),  # 28-36 in illustration, inds: 27-35
        'eyes': list(range(36, 48)),  # 37-48 in illustration, inds: 36-47
        'mouth': list(range(48, 68))  # 49-68 in illustration, inds: 48-67
    }

    args.desired_landmark_inds = [ind for ptch in args.select_patches for ind in landmarks_dict[ptch]]
    if bool(args.select_patches):
        print(f"Selected patches: {args.select_patches}")

    print(f"=> Creating dataset: {args.data_set}, with {n_cls} classes: {args.desired_classes}")
    print("Train Dataset:")
    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)

    print("Validation Dataset:")
    dataset_val, _ = build_dataset(is_train=False, args=args)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    if args.blur or args.blur_max:  # if blur > 0 or blur_max was given (and then transform is added even for blur=0).
        data_loader_train.dataset.transform = \
            add_blur_transform(data_loader_train.dataset.transform, args.blur, blur_max=args.blur_max)
        data_loader_val.dataset.transform = add_blur_transform(data_loader_val.dataset.transform, args.blur)

    if args.blur_max:
        dataset_val_blur_max, _ = build_dataset(is_train=False, args=args)

        data_loader_val_blur_max = torch.utils.data.DataLoader(
            dataset_val_blur_max, sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )

        data_loader_val_blur_max.dataset.transform = \
            add_blur_transform(data_loader_val_blur_max.dataset.transform, args.blur_max)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ For creating Tensorboard log: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if utils.is_main_process():
        tb_dir = os.path.join(args.output_dir.replace('out', 'board'),
                              "{}_epochs/{}_classes/{}".format(args.epochs, n_cls, args.model_name))
        print(f'=> Creating Tensorboard directory: {tb_dir}')
        writer_tb = SummaryWriter(log_dir=tb_dir)

        if args.blur_max and not args.blur_for_tb_log:
            args.blur_for_tb_log = args.blur_max
    else:
        writer_tb = None

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    crt_mdl_msg = f"=> Creating model: {args.model}"
    crt_mdl_msg = crt_mdl_msg + " (with pretrained weights)" if args.load_pretrained else crt_mdl_msg + " (untrained)"
    print(crt_mdl_msg)

    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        img_size=args.input_size
    )

    if args.attn_only:
        for name_p, p in model.named_parameters():
            if '.attn.' in name_p:
                p.requires_grad = True
            else:
                p.requires_grad = False
        try:
            model.head.weight.requires_grad = True
            model.head.bias.requires_grad = True
        except:
            model.fc.weight.requires_grad = True
            model.fc.bias.requires_grad = True
        try:
            model.pos_embed.requires_grad = True
        except:
            print('no position encoding')
        try:
            for p in model.patch_embed.parameters():
                p.requires_grad = False
        except:
            print('no patch embed')

    model.to(device)

    # Load trained checkpoint:
    if args.deit_model_name:
        deit_model_path = os.path.join('/home/projects/bagon/ilanaveh/code/Transformers/deit', args.deit_model_dir,
                                       args.deit_model_name)

        deit_checkpoint = torch.load(os.path.join(deit_model_path, 'best_checkpoint.pth'), map_location='cpu')

        # Remove the classification-head weights from deit checkpoint:
        deit_checkpoint_no_head = {k: v for k, v in deit_checkpoint['model'].items() if not k.startswith('head.')}

        missing, unexpected = model.load_state_dict(deit_checkpoint_no_head, strict=False)
        assert missing == ['head.weight', 'head.bias']
        assert unexpected == []

        print(f"=> Starting from deit model: '{deit_model_path}', epoch: {deit_checkpoint['epoch']}")

        with (output_dir / "log.txt").open("a") as f:
            f.write(f"Starting from deit model: '{deit_model_path}', epoch: {deit_checkpoint['epoch']}" + "\n")

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    if not args.unscale_lr:
        linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
        args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()

    lr_scheduler, _ = create_scheduler(args, optimizer)

    criterion = LabelSmoothingCrossEntropy()

    if mixup_active:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if args.bce_loss:
        criterion = torch.nn.BCEWithLogitsLoss()

    if args.resume:
        if args.resume.startswith('https'):
            resume_ok = True
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        elif os.path.isfile(os.path.join(args.resume, args.model_name, 'checkpoint.pth')):
            resume_ok = True
            checkpoint = torch.load(os.path.join(args.resume, args.model_name, 'checkpoint.pth'), map_location='cpu')
            print(f"Continuing from: {os.path.join(args.resume, args.model_name)}, epoch {checkpoint['epoch'] + 1}")
        else:
            resume_ok = False

        if resume_ok:
            model_without_ddp.load_state_dict(checkpoint['model'])
            if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                args.start_epoch = checkpoint['epoch'] + 1
                if args.model_ema:
                    utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
                if 'scaler' in checkpoint:
                    loss_scaler.load_state_dict(checkpoint['scaler'])
            lr_scheduler.step(args.start_epoch)
    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images with blur "
              f"{args.blur}: {test_stats['acc1']:.1f}%")

        if args.blur_max:
            test_stats_blur_max = evaluate(data_loader_val_blur_max, model, device)
            print(f"Accuracy of the network on the {len(dataset_val)} test images with maximal blur "
                  f"({args.blur_max}): {test_stats_blur_max['acc1']:.1f}%")

        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        if args.blur_max:
            train_stats, applied_blurs_all = train_one_epoch(
                model, criterion, data_loader_train,
                optimizer, device, epoch, loss_scaler,
                args.clip_grad, model_ema, mixup_fn,
                set_training_mode=args.train_mode,
                # keep in eval mode for deit finetuning / train mode for training and deit III finetuning
                args=args
            )

        else:
            train_stats = train_one_epoch(
                model, criterion, data_loader_train,
                optimizer, device, epoch, loss_scaler,
                args.clip_grad, model_ema, mixup_fn,
                set_training_mode=args.train_mode,
                # keep in eval mode for deit finetuning / train mode for training and deit III finetuning
                args=args,
            )

        lr_scheduler.step(epoch)
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'model_ema': get_state_dict(model_ema),
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, checkpoint_path)

        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images with blur "
              f"{args.blur}: {test_stats['acc1']:.1f}%")

        if args.blur_max:
            test_stats_blur_max = evaluate(data_loader_val_blur_max, model, device)
            print(f"Accuracy of the network on the {len(dataset_val)} test images with maximal blur "
                  f"({args.blur_max}): {test_stats_blur_max['acc1']:.1f}%")

        if max_accuracy < test_stats["acc1"]:
            max_accuracy = test_stats["acc1"]
            if args.output_dir:
                checkpoint_paths = [output_dir / 'best_checkpoint.pth']
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'model_ema': get_state_dict(model_ema),
                        'scaler': loss_scaler.state_dict(),
                        'args': args,
                    }, checkpoint_path)

        print(f'Max accuracy: {max_accuracy:.2f}%')

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.blur_max:
            log_stats = {**log_stats,
                         **{f'test_blur_max_{k}': v for k, v in test_stats_blur_max.items()}}

        if args.output_dir and utils.is_main_process():
            print(f"Saving epoch {epoch} stats to log file at: {output_dir}")
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            if args.blur_max:
                with (output_dir / "applied_blurs.txt").open("a") as f:
                    f.write(f"The blurs applied in epoch {epoch}:\n{json.dumps(applied_blurs_all)}\n\n")
        else:
            print("Not saving log.")

        if writer_tb is not None:
            print('Writing TB Train, epoch {}'.format(epoch))
            writer_tb.add_scalar('Loss/Train_Loss', train_stats['loss'], epoch)
            writer_tb.add_scalar('Accuracy/Train_Acc', train_stats['acc1'], epoch)

            print(f'Writing TB Val, epoch {epoch}')
            if args.blur_max:
                if args.blur_for_tb_log == args.blur:
                    print(f"Logging performance for minimal blur in range: {args.blur_for_tb_log}")
                    val_loss_for_tb = test_stats['loss']
                    val_acc1_for_tb = test_stats['acc1']
                elif args.blur_for_tb_log == args.blur_max:
                    print(f"Logging performance for maximal blur in range: {args.blur_for_tb_log}")
                    val_loss_for_tb = test_stats_blur_max['loss']
                    val_acc1_for_tb = test_stats_blur_max['acc1']
                else:
                    print(f"args.blur_for_tb_log should be equal to args.blur ({args.blur}) or args.blur_max ("
                          f"{args.blur_max}), but got {args.blur_for_tb_log} => not logging.")
                    val_loss_for_tb = None
                    val_acc1_for_tb = None
            else:
                print(f"Logging performance for blur {args.blur}")
                val_loss_for_tb = test_stats['loss']
                val_acc1_for_tb = test_stats['acc1']

            writer_tb.add_scalar('Loss/Val_Loss', val_loss_for_tb, epoch)
            writer_tb.add_scalar('Accuracy/Val_Acc', val_acc1_for_tb, epoch)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
