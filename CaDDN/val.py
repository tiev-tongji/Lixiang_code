import os
import torch
from Utils import Occupancy_Accuracy
from Loss import T_Loss_3D


os.environ["CUDA_VISIBLE_DEVICES"]='3,4'
device = torch.device("cuda")


def get_test_info(model, val_loader):
    model.eval()
    N = 0
    v_acc = 0
    o_acc = 0
    counter = 0
    print('**********************************************Running Test**************************************************')


    with torch.no_grad():  # 不记录梯度
        for batchi, (imgs, voxels, depths, RT, trans, DoF) in enumerate(val_loader):

            counter = counter + 1


            # 得到预测出的3D occ_grid（180, 140, 50, 2）
            grid_pre, depth_pre = model(imgs.to(device), RT.to(device), trans.to(device), DoF.to(device))


            v_accuracy, o_accuracy = Occupancy_Accuracy(voxels, grid_pre, 0)

            v_acc = v_acc + v_accuracy
            o_acc = o_acc + o_accuracy
            N = batchi


    model.train()

    return v_acc/(N+1), o_acc/(N+1)

def get_val_info(model, val_loader, m_index, l_weight, v_weight):
    N = 0

    t_l = 0
    d_l = 0
    v_l = 0
    o_l = 0
    geo_l = 0

    v_acc = 0
    o_acc = 0

    model.eval()
    print('**********************************************Running Eval**************************************************')


    with torch.no_grad():  # 不记录梯度
        for batchi, (imgs, voxels, depths, RT, intrins, DoF) in enumerate(val_loader):


            # 得到预测出的3D occ_grid（180, 140, 50, 2）
            grid_pre, depth_pre = model(imgs.to(device), RT.to(device), trans.to(device), DoF.to(device))

            # loss
            total_loss, d_loss, v_loss, o_loss = T_Loss_3D(voxels.to(device), grid_pre.to(device),
                                                                     depths.to(device), depth_pre.to(device), m_index, l_weight, v_weight)

            t_l = t_l + total_loss
            v_l = v_l + v_loss
            o_l = o_l + o_loss
            d_l = d_l + d_loss
            # geo_l = geo_l + geo_loss
            v_accuracy, o_accuracy = Occupancy_Accuracy(voxels, grid_pre, 0)

            v_acc = v_acc + v_accuracy
            o_acc = o_acc + o_accuracy

            N = batchi

    model.train()

    return t_l / (N + 1), d_l / (N + 1), v_l / (N + 1), o_l / (N + 1), v_acc / (N + 1), o_acc/(N+1)
    # return t_l/(N+1), d_l/(N+1), v_l/(N+1), o_l/(N+1), geo_l/(N+1), v_acc/(N+1), o_acc/(N+1)