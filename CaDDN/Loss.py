import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
from Utils import Upsample, bin_depths


def geo_scal_loss(pred, ssc_target):
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
               F.binary_cross_entropy(recall, torch.ones_like(recall)) + F.binary_cross_entropy(spec,
                                                                                                torch.ones_like(spec))

    # geo_loss = 0
    return geo_loss


def O_loss(pred, target):
    index = torch.nonzero(target, as_tuple=True)
    o_index = index[0]

    pre = pred[o_index, :]
    true = target[o_index, :]
    true = true.squeeze(1)

    criterion = nn.CrossEntropyLoss()
    o_loss = criterion(pre, true.long())

    # o_loss = 0
    return o_loss


def CE_loss(pred, target, m_index):
    batch, x, y, z = target.shape
    pred = pred.reshape(batch * x * y * z, 2)
    target = target.reshape(batch * x * y * z, 1)

    pre = pred
    true = target

    o_loss = O_loss(pre, true)
    true = true.squeeze(1)

    criterion = nn.CrossEntropyLoss()
    v_loss = criterion(pre, true.long())

    return v_loss, o_loss, pre, true


def Depth_loss(pred, target):
    bin = 40  # depth bin 的索引数

    N, B, D, H, W = pred.shape  # 预测出的深度图像shape
    p_depth = pred.view(B * N, D, H, W)

    b, n, h, w = target.shape  # 原始深度图像shape
    t_depth = target.view(b * n, h, w)

    t_depth = Upsample(t_depth, H, W)  # 对原始深度图像进行下采样

    # Bin depth map to create target
    t_target = bin_depths(t_depth, 'LID', 2.0, 46.80, bin - 1, target=True)

    t_target = torch.as_tensor(t_target, dtype=torch.long)
    # p_depth = torch.as_tensor(p_depth, dtype=torch.int64)

    # 计算 Depth loss
    d_loss = kornia.losses.focal_loss(p_depth, t_target, alpha=0.25, gamma=2.0, reduction="mean")

    return d_loss


def T_Loss_3D(voxels, grid_pre, depth, depth_pre, m_index, l_weight, v_weight):

    d_loss = Depth_loss(depth_pre, depth)

    v_loss, o_loss, pre, true = CE_loss(grid_pre, voxels, m_index)




    total_loss = (d_loss * l_weight) + (o_loss + (v_loss * v_weight))

    return total_loss / 3, d_loss, v_loss, o_loss

