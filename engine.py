# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main_downstream.py
"""
import math
import sys
sys.path.append("/home/projects/bagon/ilanaveh/code/Transformers/deit")  # for importing losses, utils

from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils
import numpy as np

def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None):
    # IN 9/9/25: add cosine_sim for lora feature consistency loss:
    cosine_sim = torch.nn.CosineSimilarity(dim=-1)

    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    
    if args.cosub:
        criterion = torch.nn.BCEWithLogitsLoss()

    if bool(args.blur_max):
        applied_blurs_all = []

    for sample in metric_logger.log_every(data_loader, print_freq, header):
        # 9/11/25: Add option to get landmarks from sample (relevant for downstream training)
        apply_mask = False
        if len(sample) == 2:
            samples, targets = sample
        elif len(sample) == 3:
            if bool(args.blur_max):
                # 3rd argument is 'applied_blurs' (in main_tmp, if bool(args.blur_max) => CustomCompose is used)
                samples, targets, applied_blurs = sample
                applied_blurs_all += applied_blurs.tolist()
            else:
                # 3rd argument is 'landmarks' (relevant only for downstream training)
                samples, targets, landmarks = sample  # landmarks: tensor of shape [B, n_lnd, XY] = [B, 68, 2]
                apply_mask = True
        elif len(sample) == 4:
            apply_mask = True
            if bool(args.blur_max):
                # applied_blurs + landmarks are returned (entails we're in downstream training + variable-blur.
                samples, targets, applied_blurs, landmarks = sample
            else:
                # landmarks and im_id are returned:
                samples, targets, landmarks, im_id = sample
        else:  # len(sample) = 5
            # applied_blurs + landmarks + im_id are returned (entails we're in downstream training + variable-blur).
            samples, targets, applied_blurs, landmarks, im_id = sample
            applied_blurs_all += applied_blurs.tolist()
            apply_mask = True

        # IN 21/08/25: add option to get separate sample for teacher (without blur - implemented in BlurDataset):
        if isinstance(samples, dict):
            sep_smpl_tchr = True
            samples_tchr = samples['teacher']
            samples_tchr = samples_tchr.to(device, non_blocking=True)
            samples = samples['student']
        else:
            sep_smpl_tchr = False

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # IN 29/10/24: log accuracy before using mixup function:
        with torch.cuda.amp.autocast():
            output = model(samples)
            # IN 04/08/25: For distillation, a tuple is outputted, so get mean
            # [based on deit/models/DistilledVisionTransformer > forward (line 59)]
            if not isinstance(output, torch.Tensor):
                output = (output[0] + output[1]) / 2
            acc1, acc5 = accuracy(output, targets, topk=(1, 5))

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
            
        if args.cosub:
            samples = torch.cat((samples,samples),dim=0)
            
        if args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)
         
        with torch.cuda.amp.autocast():
            if sep_smpl_tchr and not isinstance(criterion, DistillationLoss):
                # 9/9/25: LoRA with feature consistency
                outputs_high, feats_high = model(samples_tchr, return_features=True)
                outputs_blur, feats_blur = model(samples, return_features=True)
            else:
                if apply_mask:
                    # Create mask:
                    patch_mask = utils.build_patch_mask(landmarks, bb_size=args.landmark_bb_size,
                                                        thresh_jaccard=args.thresh_jaccard_index)
                    if args.debug_mask:
                        for i in range(len(samples)):
                            utils.visualize_patch_mask(img=samples[i], landmarks=landmarks[i], patch_mask=patch_mask[i],
                                                       save_fig=True, im_id=im_id[i],
                                                       suf=f'after_transform_jaccard{args.thresh_jaccard_index}')
                    # Pass mask to model, to drop all other patches:
                    outputs = model(samples, patch_mask)
                else:
                    outputs = model(samples)
            if not args.cosub:
                if isinstance(criterion, DistillationLoss):
                    if sep_smpl_tchr:
                        loss = criterion(samples, outputs, targets, inputs_tchr=samples_tchr)
                    else:
                        loss = criterion(samples, outputs, targets)
                else:
                    if sep_smpl_tchr:
                        # 9/9/25: LoRA with feature consistency
                        loss_cls = criterion(outputs_blur, targets)  # Main loss: classify from blurred inputs
                        loss_feat = 1 - cosine_sim(feats_high, feats_blur).mean()  # Consistency loss on CLS tokens
                        loss = loss_cls + args.feat_const_lamda * loss_feat  # λ = 0.2, tune between 0.1–0.5
                    else:
                        loss = criterion(outputs, targets)
            else:
                outputs = torch.split(outputs, outputs.shape[0]//2, dim=0)
                loss = 0.25 * criterion(outputs[0], targets) 
                loss = loss + 0.25 * criterion(outputs[1], targets) 
                loss = loss + 0.25 * criterion(outputs[0], outputs[1].detach().sigmoid())
                loss = loss + 0.25 * criterion(outputs[1], outputs[0].detach().sigmoid()) 

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        batch_size = samples.shape[0]
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        #  IN 29/10/24: add train accuracy to logger:
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if bool(args.blur_max):
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, applied_blurs_all
    else:
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, return_breakdown=False, des_classes=None, args=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    # Create empty breakdown dict, for adding each batch's breakdown to:
    if return_breakdown:
        if not des_classes:
            print("Classes not given --> not returning breakdown")
            return_breakdown = False
        else:
            breakdown = {c_tar: {c_pred: 0 for c_pred in des_classes} for c_tar in des_classes}

    for sample in metric_logger.log_every(data_loader, 10, header):
        # 9/11/25: Add option to get landmarks from sample (relevant for downstream training)
        apply_mask = False
        if len(sample) == 2:
            images, target = sample
        else:
            if len(sample) == 3:
                # 3rd argument is 'landmarks' (relevant only for downstream training)
                images, target, landmarks = sample  # landmarks: tensor of shape [B, n_lnd, XY] = [B, 68, 2]

            elif len(sample) == 4:
                # landmarks and im_id are returned:
                images, target, landmarks, im_id = sample

            apply_mask = True
            landmarks = landmarks.to(device)

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            if apply_mask:
                # Create mask:
                patch_mask = utils.build_patch_mask(landmarks, bb_size=args.landmark_bb_size,
                                                    thresh_jaccard=args.thresh_jaccard_index)
                if args.debug_mask:
                    for i in range(len(images)):
                        utils.visualize_patch_mask(img=images[i], landmarks=landmarks[i], patch_mask=patch_mask[i],
                                                   save_fig=True, im_id=im_id[i],
                                                   suf=f'after_transform_jaccard{args.thresh_jaccard_index}')
                # Pass mask to model, to drop all other patches:
                output = model(images, patch_mask)
            else:
                output = model(images)
            loss = criterion(output, target)
        if return_breakdown:
            acc_list, breakdown = accuracy_with_class_breakdown(output, target, topk=(1, 5), return_breakdown=True,
                                                                prev_breakdown=breakdown)
            acc1, acc5 = acc_list
        else:
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    if return_breakdown:
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, breakdown
    else:
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def accuracy_with_class_breakdown(output, target, topk=(1,), return_breakdown=False, prev_breakdown={}):
    """
    Based on timm.utils.accuracy, but added breakdown of per-class accuracy.
    prev_breakdown should be given if return_breakdown=True.
    :return:
    """
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    correct_for_topk = [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]

    if return_breakdown:
        pred_top1 = np.array(pred[0].cpu())
        for i, t in enumerate(np.array(target.cpu())):
            prev_breakdown[t][pred_top1[i]] += 1
        return correct_for_topk, prev_breakdown

    else:
        return correct_for_topk
