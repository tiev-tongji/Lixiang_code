import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F



def One_Hot(pre_voxel):

    num_class = pre_voxel.shape[-1]
    pre_voxel_max = torch.argmax(pre_voxel, dim = -1)

    pre_voxel_max = np.array(pre_voxel_max.cpu())

    oh_arr = torch.eye(2)[pre_voxel_max]

    return oh_arr


def Occupancy_Accuracy(true, pre, flag):

    pre = F.softmax(pre, dim=1)
    device = true.device

    b,x,y,z = true.shape
    sum = b*x*y*z


    pre = pre.permute(0, 2, 3, 4, 1)
    pre = torch.argmax(pre, dim=-1)

    p_num = torch.count_nonzero(pre)
    t_num = torch.count_nonzero(true)
    t_bili = t_num / sum
    o_bili = p_num / t_num
    if flag == 1:
        print('        t_scale & o_scale is     :', t_bili.item(), o_bili.item())


    v_acc = torch.eq(pre.to(device), true.to(device)).float().mean()  # 计算Occupancy 的整体准确率


    pt = true.to(device) * pre.to(device)

    p_num = torch.count_nonzero(pt)
    o_acc = p_num / t_num            # 计算Occupancy 的正样本的正确率


    return v_acc, o_acc


def mask_fun(B):

    mask = np.load("/home/ps/Downloads/mwj_data/Mask_tensor.npy", allow_pickle=True)
    mask = torch.tensor(mask, dtype=torch.float32)
    x,y,z = mask.shape
    mask = mask.reshape(1,x,y,z)
    Mask = mask.repeat_interleave(repeats = B, dim=0)

    return Mask


def mask_index(Mask):

    B,X,Y,Z = Mask.shape
    mask = Mask.reshape(B*X*Y*Z, 1)
    mask = torch.mean(input = mask, dim=1, keepdim=True)
    mask = torch.nonzero(mask, as_tuple=True)
    m_index = mask[0]

    return m_index


def Upsample(depth_img, H, W):

    depth_img = depth_img.unsqueeze(1)
    #print(depth_img.shape,H,W)

    up = nn.Upsample(size=(H, W), mode='bilinear', align_corners=True)
    img_resize = up(depth_img)
    img_resize = img_resize.squeeze(1)
    return img_resize


def bin_depths(depth_map, mode, depth_min, depth_max, num_bins, target=False):

    if mode == "UD":
        bin_size = (depth_max - depth_min) / num_bins
        indices = ((depth_map - depth_min) / bin_size)
    elif mode == "LID":
        bin_size = 2 * (depth_max - depth_min) / (num_bins * (1 + num_bins))
        indices = -0.5 + 0.5 * torch.sqrt(1 + 8 * (depth_map - depth_min) / bin_size)
    elif mode == "SID":
        indices = num_bins * (torch.log(1 + depth_map) - math.log(1 + depth_min)) / \
                  (math.log(1 + depth_max) - math.log(1 + depth_min))
    else:
        raise NotImplementedError

    if target:
        # Remove indicies outside of bounds
        mask = (indices < 0) | (indices > num_bins) | (~torch.isfinite(indices))
        indices[mask] = num_bins

        # Convert to integer
        indices = indices.type(torch.int16)

    return indices


def Upsample_3D(Occ_3D, C, H, W):


    up = nn.Upsample(size=(C, H, W), mode='trilinear', align_corners=True)
    img_resize = up(Occ_3D)


    return img_resize


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=15, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss