import os
import torch
from params import class_frequencies
from utils import Occupancy_Accuracy
from total_loss import T_Loss_3D

os.environ["CUDA_VISIBLE_DEVICES"]='1,2'
device = torch.device("cuda")

def get_val_info(model, val_loader):
    N = 0

    t_l = 0
    v_l = 0
    sem_l = 0
    geo_l = 0

    v_acc = 0
    o_acc = 0

    class_fre = torch.as_tensor(class_fre, dtype=torch.float32)

    model.eval()
    print('**********************************************Running Eval**************************************************')


    with torch.no_grad():  # 不记录梯度
        for batchi, (img, target, fov_mask, projected_pix, ) in enumerate(val_loader):

            # 得到预测出的3D occ_grid（180, 140, 50, 2）
            grid_pre = model(img.to(device), projected_pix.to(device), fov_mask.to(device),)

            # loss
            total_loss, v_loss, sem_loss, geo_loss = T_Loss_3D(target.to(device), grid_pre.to(device), class_fre.to(device))

            t_l = t_l + total_loss
            v_l = v_l + v_loss

            sem_l = sem_l + sem_loss
            geo_l = geo_l + geo_loss

            v_accuracy, o_accuracy = Occupancy_Accuracy(target, grid_pre, 0)

            v_acc = v_acc + v_accuracy
            o_acc = o_acc + o_accuracy

            N = batchi

    model.train()

    return t_l / (N + 1), v_l / (N + 1), sem_l / (N + 1), geo_l / (N + 1), v_acc / (N + 1), o_acc / (N + 1)