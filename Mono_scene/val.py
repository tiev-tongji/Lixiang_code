import os
import torch
from Loss import T_Loss_3D
from utils import Occupancy_Accuracy
from utils import expand_batch_fov_mask, expand_batch_projected_pix




os.environ["CUDA_VISIBLE_DEVICES"]='0'
device = torch.device("cuda")


def get_val_info(model, val_loader, batch, projected_pix, fov_mask):
    N = 0

    t_l = 0
    v_l = 0

    geo_l = 0
    o_l = 0

    v_acc = 0
    o_acc = 0

    model.eval()
    print('**********************************************Running Eval**************************************************')


    with torch.no_grad():  # 不记录梯度
        for batchi, (imgs, rots, trans, intrins, voxels)in enumerate(val_loader):

            pix = expand_batch_projected_pix(B = batch, projected_pix = projected_pix)
            mask = expand_batch_fov_mask(B = batch, fov_mask = fov_mask)


            grid_pre = model(imgs.to(device), pix.to(device), mask.to(device), )

            # loss
            total_loss, v_loss, o_loss, geo_loss = T_Loss_3D(voxels.to(device), grid_pre.to(device))


            t_l = t_l + total_loss
            v_l = v_l + v_loss
            o_l = o_l + o_loss
            geo_l = geo_l + geo_loss

            v_accuracy, o_accuracy = Occupancy_Accuracy(voxels, grid_pre, 0)

            v_acc = v_acc + v_accuracy
            o_acc = o_acc + o_accuracy

            N = batchi

    model.train()


    return t_l / (N + 1), v_l / (N + 1), o_l / (N + 1), geo_l / (N + 1), v_acc / (N + 1), o_acc / (N + 1)