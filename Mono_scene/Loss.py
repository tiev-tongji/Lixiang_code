import torch
import torch.nn as nn
import torch.nn.functional as F


def sem_scal_loss(pred, ssc_target):
    # Get softmax probabilities
    pred = F.softmax(pred, dim=1)
    loss = 0
    count = 0

    # mask = ssc_target != 255

    n_classes = pred.shape[1]
    for i in range(0, n_classes):

        # Get probability of class i
        p = pred[:, i]

        # Remove unknown voxels
        target_ori = ssc_target
        target = ssc_target



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

    # Get softmax probabilities
    # bs = ssc_target[0]
    pred = F.softmax(pred, dim=1)


    # Compute empty and nonempty probabilities
    empty_probs = pred[:, 0]
    nonempty_probs = 1 - empty_probs


    nonempty_target = ssc_target != 0
    nonempty_target = nonempty_target.float()


    intersection = (nonempty_target * nonempty_probs).sum()
    precision = intersection / nonempty_probs.sum()
    recall = intersection / nonempty_target.sum()
    spec = ((1 - nonempty_target) * (empty_probs)).sum() / (1 - nonempty_target).sum()

    geo_loss = F.binary_cross_entropy(precision, torch.ones_like(precision)) + \
               F.binary_cross_entropy(recall, torch.ones_like(recall)) + F.binary_cross_entropy(spec, torch.ones_like(spec))


    return geo_loss


def O_loss(pred, target):

    index = torch.nonzero(target, as_tuple=True)
    o_index = index[0]

    pre = pred[o_index,:]
    true = target[o_index,:]
    true = true.squeeze(1)

    criterion = nn.CrossEntropyLoss()
    o_loss = criterion(pre, true.long())

    return o_loss


def CE_loss(pred, target):

    batch, class_num, x, y, z = pred.shape
    pred = pred.reshape(batch * x * y * z, class_num)
    target = target.reshape(batch * x * y * z, 1)

    pre = pred
    true = target

    o_loss = O_loss(pre, true)
    true = true.squeeze(1)

    criterion = nn.CrossEntropyLoss()
    v_loss = criterion(pre, true.long())

    return v_loss,o_loss, pre, true


def T_Loss_3D(voxels, grid_pre):

    v_loss, o_loss, pre, true= CE_loss(grid_pre, voxels)

    geo_loss = geo_scal_loss(pre, true)

    total_loss = (v_loss  + o_loss + geo_loss) / 3

    return total_loss, v_loss, o_loss, geo_loss