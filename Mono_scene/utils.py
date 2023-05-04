import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Resize


def Occupancy_Accuracy(true, pre, flag):

    pre = F.softmax(pre, dim=1)
    device = true.device

    b, x, y, z = true.shape
    sum = b*x*y*z



    pre = pre.permute(0, 2, 3, 4, 1)
    pre = torch.argmax(pre, dim = -1)

    p_num = torch.count_nonzero(pre)
    t_num = torch.count_nonzero(true)
    t_bili = t_num / sum
    o_bili = p_num / t_num
    if flag == 1:
        print('        t_scale & o_scale is     :', t_bili.item(), o_bili.item())

    v_acc = torch.eq(pre.to(device), true.to(device)).float().mean()  # 计算Occupancy 的整体准确率


    index = true != 0
    o_true = true[index]
    o_pre = pre[index]
    o_acc = torch.eq(o_pre.to(device), o_true.to(device)).float().mean()  # 计算Occupancy 的整体准确率

    return v_acc, o_acc


def expand_batch_projected_pix(B, projected_pix):

    cam, N, pix = projected_pix.shape
    projected_pix = projected_pix.reshape(1, cam, N, pix)
    projected_pix = projected_pix.repeat_interleave(repeats = B, dim=0)

    return projected_pix


def expand_batch_fov_mask(B, fov_mask):

    cam, N = fov_mask.shape
    fov_mask = fov_mask.reshape(1, cam, N)
    fov_mask = fov_mask.repeat_interleave(repeats = B, dim=0)

    return fov_mask

def expand_batch_pts(B, pts):

    print('pts.shape:', pts.shape)

    cam, N, x = pts.shape
    pts = pts.reshape(1, cam, N, x)
    pts = pts.repeat_interleave(repeats = B, dim=0)

    return pts


def mask_fun(B):

    mask = np.load("/home/ps/Downloads/nu_tensor/project_xy_mask/resize/fov_mask_3D.npy", allow_pickle=True)
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


def Upsample_3D(Occ_3D, C, H, W):


    up = nn.Upsample(size=(C, H, W), mode='trilinear', align_corners=True)
    img_resize = up(Occ_3D)


    return img_resize


def Resize_RGB_1(Img):
    torch_resize = Resize([225, 400])
    img = torch_resize(Img)

    return img

def Resize_RGB_2(Img):
    torch_resize = Resize([900, 1600])
    img = torch_resize(Img)

    return img



