import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def sem_scal_loss(pred, ssc_target):

    # Get softmax probabilities
    pred = F.softmax(pred, dim=1)
    loss = 0
    count = 0

    n_classes = pred.shape[1]
    mask = ssc_target != 17

    for i in range(0, n_classes):

        # Get probability of class i
        p = pred[:, i]

        # Remove unknown voxels
        target_ori = ssc_target
        p = p[mask]
        target = ssc_target[mask]

        completion_target = torch.ones_like(target)
        completion_target[target != i] = 0

        completion_target_ori = torch.ones_like(target_ori).float()
        completion_target_ori[target_ori != i] = 0

        if torch.sum(completion_target) > 0:
            count += 1.0
            nominator = torch.sum(p * completion_target)
            loss_class = 0

            if torch.sum(p) > 0:
                precision = nominator / (torch.sum(p))
                loss_precision = F.binary_cross_entropy(
                    precision, torch.ones_like(precision)
                )
                loss_class += loss_precision
            if torch.sum(completion_target) > 0:
                recall = nominator / (torch.sum(completion_target))
                loss_recall = F.binary_cross_entropy(recall, torch.ones_like(recall))
                loss_class += loss_recall

            if torch.sum(1 - completion_target) > 0:
                specificity = torch.sum((1 - p) * (1 - completion_target)) / (
                    torch.sum(1 - completion_target)
                )
                loss_specificity = F.binary_cross_entropy(
                    specificity, torch.ones_like(specificity)
                )
                loss_class += loss_specificity

            loss += loss_class

    return loss / count


def geo_scal_loss(pred, ssc_target):

    pred = F.softmax(pred, dim=1)

    # Compute empty and nonempty probabilities
    empty_probs = pred[:, 0]
    nonempty_probs = 1 - empty_probs

    mask = ssc_target != 17
    nonempty_target = ssc_target != 0
    nonempty_target = nonempty_target[mask].float()

    nonempty_probs = nonempty_probs[mask]
    empty_probs = empty_probs[mask]


    intersection = (nonempty_target * nonempty_probs).sum()
    precision = intersection / nonempty_probs.sum()
    recall = intersection / nonempty_target.sum()
    spec = ((1 - nonempty_target) * (empty_probs)).sum() / (1 - nonempty_target).sum()

    geo_loss = F.binary_cross_entropy(precision, torch.ones_like(precision)) + \
               F.binary_cross_entropy(recall, torch.ones_like(recall)) + F.binary_cross_entropy(spec, torch.ones_like(spec))


    return geo_loss


def CE_loss(pred, target, class_fre):

    class_weights = 1 / torch.log(class_fre + 0.001)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights, ignore_index = 17, reduction="mean"
    )

    v_loss = criterion(pred, target.long())

    return v_loss


def T_Loss_3D(target, grid_pre, class_fre):

    B, C, X, Y, Z = grid_pre.shape

    Shape = B * X * Y * Z
    target = target.reshape(Shape)
    grid_pre = grid_pre.reshape(Shape, -1)

    mask = target != 18
    target = target[mask]

    grid_pre = grid_pre[mask,:]

    v_loss = CE_loss(grid_pre, target, class_fre)

    geo_loss = geo_scal_loss(grid_pre, target)

    sem_loss = sem_scal_loss(grid_pre, target)

    total_loss = (v_loss + sem_loss + geo_loss) / 3

    return total_loss,  v_loss,  sem_loss, geo_loss