"""
13/08/2024
Probe intermediate layers from deit, and train on AffectNet.
"""
import os
from pathlib import Path

from timm.models import create_model
import torch
import torch.nn as nn
from data_for_inter import AttsDatasetFixed
import os.path as osp
import torchvision.transforms as T
import numpy as np
import json
import shutil
import argparse


def create_model_probed(block_ind, num_classes=2, model_path=None):
    """
    create model based on a pretrained deit, with a FC layer after the chosen block from deit.
    Freeze all parameters
    except for the new FC layer.
    :param model_path: path to trained deit checkpoint, from which we want to start.
    :param block_ind: number of block we want to probe from (its output) - between 0-11.
    :param num_classes: # of output classes (2 for binary classification).
    :return: probed model
    """

    # Create deit model with parameters according to those given in main.py:
    model = create_model(
        'deit_base_patch16_224',
        pretrained=False,
        num_classes=1000,
        drop_rate=0,
        drop_path_rate=0.1,
        drop_block_rate=None,
        img_size=224
    )

    model.cuda()
    model.eval()

    # Load trained checkpoint:
    if model_path:
        deit_checkpoint = torch.load(os.path.join(model_path, 'checkpoint.pth'), map_location='cpu')
        model.load_state_dict(deit_checkpoint['model'])

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Remove all blocks downstream to chosen block: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    inds2rmv = range(len(model.blocks)-1, block_ind, -1)

    for i in inds2rmv:
        del model.blocks[i]

    # ~~~~~~~~~~~~~~~~~~~~~~~ Replace last FC layer with a new one, for binary classification: ~~~~~~~~~~~~~~~~~~~~~~~~~
    model.head = nn.Linear(model.embed_dim, num_classes)  # embed_dim = 768 (don't change, only change num_classes).

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Freeze weights of all layers, except for last FC: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    for param in model.parameters():
        param.requires_grad = False

    model.head.bias.requires_grad = True
    model.head.weight.requires_grad = True

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=32, type=int, metavar='N', help='mini-batch size (default: 2)')
    # parser.add_argument('--blur', default=0, type=int, help='blur sigma of the model and inputs for the atts training')
    parser.add_argument('--epochs', default=30, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--lr', default=None, type=float, metavar='LR', help='initial learning rate, None for debug')
    parser.add_argument('--block_ind', default=11, type=int, help='probe output of this block')
    parser.add_argument('--deit_model_name', default=None, type=str, help='model name, or None for untrained detr')

    args = parser.parse_args()

    # When debugging - change model sufix, and set normal LR:
    db_suf = '' if args.lr else '_db'
    args.lr = args.lr if args.lr else .01

    if args.deit_model_name:
        deit_model_path = osp.join('/home/projects/bagon/ilanaveh/code/Transformers/deit/out', args.deit_model_name)
        model_name = f'{args.deit_model_name}_block{args.block_ind}_lr{args.lr}'
    else:
        model_name = f'untrained_block{args.block_ind}_lr{args.lr}'

    model_name = model_name + db_suf

    home_dir = '/home/projects/bagon/ilanaveh'
    dataset_path = osp.join(home_dir, 'data/AffectNet/train_set')
    data_dir = osp.join(dataset_path, 'images')

    output_dir = Path(home_dir) / 'code/Transformers/deit/intermediate/out' / model_name
    output_dir.mkdir(parents=False, exist_ok=True)  # create output_dir if doesn't exist, alert if parent doesn't exist.

    cuda = torch.cuda.is_available()

    if args.deit_model_name:
        model = create_model_probed(args.block_ind, 2, deit_model_path)
    else:
        model = create_model_probed(args.block_ind, 2)

    model.eval()

    cont_from_resume = osp.isfile(osp.join(output_dir, 'checkpoint.pth.tar'))

    if cont_from_resume:
        inter_checkpoint = torch.load(osp.join(output_dir, 'checkpoint.pth.tar'))
        print('>> Using saved checkpoint from {}'.format(osp.join(output_dir, 'checkpoint.pth.tar')))
        model.load_state_dict(inter_checkpoint['model_state_dict'])
        start_epoch = inter_checkpoint['epoch'] + 1
        best_acc = inter_checkpoint['best_acc']
        print('>> Loaded checkpoint, continuing from epoch {}'.format(start_epoch))
    else:
        print('>> file {} not found, starting from scratch'.format(osp.join(output_dir, 'checkpoint.pth.tar')))
        start_epoch = 1
        best_acc = 0

    criterion = nn.CrossEntropyLoss()

    if cuda:
        model.cuda()
        criterion = criterion.cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    # --------------------------------------------- Datasets and Dataloaders: ------------------------------------------
    mean_rgb = [0.485, 0.456, 0.406]
    std_rgb = [0.229, 0.224, 0.225]

    transforms = T.Compose([T.ToTensor(), T.Normalize(mean=mean_rgb, std=std_rgb)])

    train_dataset = AttsDatasetFixed(
        csv_file=osp.join(dataset_path, 'AffectNet_lbls_phase.csv'),
        root_dir=data_dir,
        transform=transforms,
        phase='train',
        return_im_name=True)

    val_dataset = AttsDatasetFixed(
        csv_file=osp.join(dataset_path, 'AffectNet_lbls_phase.csv'),
        root_dir=data_dir,
        transform=transforms,
        phase='val',
        return_im_name=True)

    assert train_dataset.anns_df.merge(val_dataset.anns_df, on=['im_name']).empty  # validate train and val are distinct

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Get stats for model before training: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    val_acc, val_loss = validate(val_loader, model, criterion)

    best_acc = max(val_acc, best_acc)

    log_stats = {'epoch': 0,
                 'test_acc': val_acc,
                 'test_loss': val_loss.item()}

    with (output_dir / "log.txt").open("a") as f:
        f.write(json.dumps(log_stats) + "\n")

    save_checkpoint({
        'epoch': 0,
        'model_state_dict': model.state_dict(),
        'optim_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
        'trainset_inds': train_dataset.anns_df.index
    }, is_best=False, filedir=output_dir)

    print(f'Epoch 0 (before beginning training): Checkpoint Saved.')

    for ep in range(start_epoch, args.epochs):
        print('\nepoch', ep)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~         Train        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        tr_acc, tr_loss = train(train_loader, model, criterion, optimizer)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~         Val:       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        val_acc, val_loss = validate(val_loader, model, criterion)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Check if best: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        is_best = val_acc > best_acc

        if is_best:
            print('\nnew best!\n'
                  'end of epoch: {}\n'
                  'new best prec1: {}\n'.format(ep, val_acc))

        best_acc = max(val_acc, best_acc)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Log stats : ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        log_stats = {'epoch': ep,
                     'train_acc': tr_acc,
                     'train_loss': tr_loss.item(),
                     'test_acc': val_acc,
                     'test_loss': val_loss.item(),
                     'is_best': int(is_best)}

        with (output_dir / "log.txt").open("a") as f:
            f.write(json.dumps(log_stats) + "\n")

        save_checkpoint({
            'epoch': ep,
            'model_state_dict': model.state_dict(),
            'optim_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
            'trainset_inds': train_dataset.anns_df.index
        }, is_best=is_best, filedir=output_dir)

        print(f'Epoch {ep}: Checkpoint Saved.')


def save_checkpoint(state, is_best, filedir='.'):
    filename = os.path.join(filedir, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(filedir, 'best_checkpoint.pth.tar'))


def accuracy(output, target):
    """
    Computes the precision@k for the specified values of k
    Copy-paste from DeepLabv3FineTuning-disClasses/intermediate/train_on_atts/teach/train_atts_teach.py
    """
    batch_size = target.size(0)

    pred = np.argmax(output.cpu(), axis=1)
    correct = pred.eq(target.cpu()).numpy().astype(int)

    res = np.sum(correct) * 100 / batch_size
    return res


def train(train_loader, model, criterion, optimizer):
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    for i, sample in enumerate(train_loader):
        if (i % 10) == 0:
            print(f'{i} / {len(train_loader)}')

        input, target, im_name = sample

        input = input.cuda()
        target = target.cuda()  # IN 4/4/24 - removed " async=True".

        # compute output
        output = model(input)

        # Compute loss:
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec = accuracy(output.data, target)
        losses.update(loss.data, input.size(0))
        acc.update(prec, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        # print devices of model, input and labels:
        optimizer.step()

    return acc.avg, losses.avg


def validate(val_loader, model, criterion):
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, sample in enumerate(val_loader):

        if (i % 10) == 0:
            print(f'{i} / {len(val_loader)}')

        input, target, im_name = sample

        input = input.cuda()
        target = target.cuda()

        # compute output
        output = model(input)

        # Compute loss:
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec = accuracy(output.data, target)
        losses.update(loss.data, input.size(0))
        acc.update(prec, input.size(0))

    return acc.avg, losses.avg


class AverageMeter(object):
    """
    Computes and stores the average and current value (copy-paste from
    DeepLabv3FineTuning-disClasses/intermediate/train_on_atts/teach/train_atts_teach_loss.py)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
