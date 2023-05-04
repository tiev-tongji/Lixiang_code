import os
import yaml
import torch
import numpy as np
from time import time
from Loss import T_Loss_3D
import torch.nn.functional as F
from CaDDN.Mul_Frame.Model import compile_model
from CaDDN.Mul_Frame.Data_Loader import compile_data
from tensorboardX import SummaryWriter
from Utils import  Occupancy_Accuracy, EarlyStopping
from val import get_test_info, get_val_info


os.environ["CUDA_VISIBLE_DEVICES"]='1'
device = torch.device("cuda")


def train(
          logdir='/...../Loss',  # 日志的输出文件
          Batch = 2,
          nepochs = 24,  # 训练最大的epoch数
          lr = 1e-4,  # 学习率
          weight_decay = 1e-6,  # 权重衰减系数
          max_grad_norm = 5.0,
          stop = 1,
          val_step = 30,  # 每隔多少个iter验证一次
          baocun = 1,    # 是否保存可视化数据
          l_weight = 0.8,

          v_weight = 0.6,
          config= 'example.yaml' # 数据文件夹地址
          ): # 配置参数文件夹地址


    cfg = yaml.load(open(config, 'r'), Loader=yaml.Loader)



    print("*************正在加载train_data & val_data & test_data***************")

    train_loader, val_loader, test_loader = compile_data(batch = Batch)  # 获取训练数据和测试数据

    print('加载tesnor数据所需内存: %.2f M' % (torch.cuda.memory_allocated() / 1e6))
    print('加载数据总共所需内存: %.2f M' % (torch.cuda.memory_reserved() / 1e6))
    print("***********************训练数据和测试数据加载成功***********************")

    early_stopping = EarlyStopping()
    model = compile_model(cfg)    # 获取模型
    model = torch.nn.DataParallel(model)
    model.to(device)
    print('模型初始化tensor所需内存: %.2f M' % (torch.cuda.memory_allocated()/ 1e6 ))
    print('模型初始化总共需要内存: %.2f M' % (torch.cuda.memory_reserved() / 1e6))

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)  # 使用Adam优化器
    writer = SummaryWriter(logdir=logdir)  # 记录训练过程

    print('++++++++++++++++++++++++++++++Training Start+++++++++++++++++++++++++++++++++++++++')
    model.train()
    counter = 0
    T0 = time()

    for epoch in range(nepochs):
        np.random.seed()

        print('**********************************Training for {} epoch************************************'.format(epoch))

        for batchi, (imgs, voxels, depths, RT, intrins, DoF) in enumerate(train_loader):
            opt.zero_grad()

            # 得到预测出的3D occ_grid（batch, 180, 140, 50, 2）
            grid_pre, depth_pre = model(imgs.to(device), RT.to(device), intrins.to(device), DoF.to(device))

            # 进行 loss 运算
            total_loss, d_loss, v_loss, o_loss = T_Loss_3D(voxels.to(device), grid_pre.to(device),
                                                                     depths.to(device), depth_pre.to(device),l_weight, v_weight)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # 梯度裁剪

            opt.step()
            counter += 1

            if counter % 5 == 0:  # 每5个iter打印并记录一次loss
                writer.add_scalar('train/total_loss', total_loss, counter)
                writer.add_scalar('train/dep_loss', d_loss, counter)
                writer.add_scalar('train/v_loss', v_loss, counter)
                writer.add_scalar('train/o_loss', o_loss, counter)


                v_accuracy, o_accuracy = Occupancy_Accuracy(voxels, grid_pre, 1)

                print('---------counter{} & total_loss:{} & depth_loss:{} & v_loss:{} & o_loss:{} ------------- '.format(counter, total_loss.item(), d_loss.item(), v_loss.item(), o_loss.item()))

                print( '//////////////////counter{} & train/v_accuracy:{} & train/o_accuracy:{}/////////////////////// \n'.format(counter, v_accuracy.item(), o_accuracy.item()))

                writer.add_scalar('train/v_accuracy', v_accuracy, counter)
                writer.add_scalar('train/o_accuracy', o_accuracy, counter)

                if counter % 50 == 0:
                    if baocun == 1:

                        grid_pre = F.softmax(grid_pre, dim=1)
                        grid_pre = grid_pre.permute(0,2,3,4,1)

                        p_dir = "/..../p"
                        t_dir = "/..../t"

                        name1 = str(counter) + '_true'
                        name2 = str(counter) + '_pre'

                        t_path = os.path.join(t_dir, name1)
                        p_path = os.path.join(p_dir, name2)

                        p_voxel = grid_pre.cpu().detach().numpy()
                        np.save(t_path, voxels)
                        np.save(p_path, p_voxel)
                        print('         保存第{}批次的数据     \n'.format(counter))

            if counter % val_step == 0:  # 验证1次，记录loss
                v_total_loss, v_d_loss, v_v_loss, v_o_loss, v_acc, o_acc = get_val_info(model, val_loader, l_weight, v_weight)

                print('     val counter{} / total_loss:{} & dep_loss:{} & v_loss: {} & o_loss: {} '.format(counter, v_total_loss.item(), v_d_loss.item(), v_v_loss.item(), v_o_loss.item()))

                print('     val /  v_accuracy:{} & o_accuracy:{}      '.format(v_acc.item(), o_acc.item()))
                writer.add_scalar('val/total_loss', v_total_loss, counter)
                writer.add_scalar('val/dep_loss', v_d_loss, counter)
                writer.add_scalar('val/v_loss', v_v_loss, counter)
                writer.add_scalar('val/o_loss', v_o_loss, counter)


                writer.add_scalar('val/v_accuracy', v_acc, counter)
                writer.add_scalar('val/o_accuracy', o_acc, counter)
                print(
                    '**********************************************Ending Eval**************************************************\n')

                early_stopping(v_total_loss, model)
                if early_stopping.early_stop:
                    stop = 0
                    break

        if stop == 0:
            print("Early stopping")  # 结束模型训练
            break

    test_v_acc, test_o_acc = get_test_info(model, test_loader)
    print('             test / v_accuracy:{} & o_accuracy:{}               '.format(test_v_acc.item(), test_o_acc.item()))
    print('整个训练所需最大内存： %.2f M' % (torch.cuda.max_memory_allocated() / 1e6))

    T1 = time()
    print('整个训练所需时间 {} S'.format((T1 - T0)))
    print('++++++++++++++++++++++++++++++Training End+++++++++++++++++++++++++++++++++++++++')



if __name__ == '__main__':
    train()

