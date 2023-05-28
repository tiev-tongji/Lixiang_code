import os
import torch
import numpy as np
from time import time
from loss import T_Loss_3D
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from val import get_val_info
from data_process.data_pro import compile_data
from mono_net import compile_model
from early_stop import EarlyStopping
from project_mask import load_project_mask
from utils import mask_fun, mask_index, Occupancy_Accuracy
from utils import expand_batch_fov_mask, expand_batch_projected_pix
from params import class_frequencies

os.environ["CUDA_VISIBLE_DEVICES"]='0,1'
device = torch.device("cuda")


def train(
          logdir='/...../Loss',  # 日志的输出文件
          Batch = 2,
          nepochs = 25,  # 训练最大的epoch数
          lr = 1e-4,  # 学习率
          weight_decay = 1e-5,  # 权重衰减系数
          max_grad_norm = 5.0,
          stop = 1,
          val_step = 500,  # 每隔多少个iter验证一次
          baocun = 1,    # 是否保存可视化数据
          ): # 配置参数文件夹地址


    print("*************正在加载train_data & val_data & test_data***************")
    projected_pix, fov_mask, pts = load_project_mask()
    train_loader, val_loader = compile_data(batch = Batch)  # 获取训练数据和测试数据

    class_fre = class_frequencies
    class_fre = torch.as_tensor(class_fre, dtype=torch.float32)

    print('加载tesnor数据所需内存: %.2f M' % (torch.cuda.memory_allocated() / 1e6))
    print('加载数据总共所需内存: %.2f M' % (torch.cuda.memory_reserved() / 1e6))
    print("***********************训练数据和测试数据加载成功***********************")

    early_stopping = EarlyStopping()
    model = compile_model()    # 获取模型


    model = torch.nn.DataParallel(model)
    model.to(device)


    print('模型初始化tensor所需内存: %.2f M' % (torch.cuda.memory_allocated()/ 1e6 ))
    print('模型初始化总共需要内存: %.2f M' % (torch.cuda.memory_reserved() / 1e6))

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)  # 使用Adam优化器
    writer = SummaryWriter(logdir=logdir)  # 记录训练过程

    Mask = mask_fun(B = Batch)
    m_index = mask_index(Mask)
    print("加载 mask_3D: ", Mask.shape)

    print('++++++++++++++++++++++++++++++Training Start+++++++++++++++++++++++++++++++++++++++')
    model.train()
    counter = 0
    T0 = time()

    for epoch in range(nepochs):
        np.random.seed()

        print( '**********************************Training for {} epoch************************************'.format(epoch))

        for batchi, (img, target, fov_mask, projected_pix) in enumerate(train_loader):

            # print('imgs.shape:', img.shape)
            # print('target.shape:', target.shape)
            # print('fov_mask.shape:', fov_mask.shape)
            # print('projected_pix.shape:', projected_pix.shape)

            opt.zero_grad()

            # 得到预测出的3D occ_grid（180, 140, 50, 2）
            grid_pre = model(img.to(device), projected_pix.to(device), fov_mask.to(device),)

            target = target.to(device)
            # 进行 loss 运算
            total_loss, v_loss, sem_loss, geo_loss = T_Loss_3D(target.to(device), grid_pre.to(device), class_fre.to(device))

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # 梯度裁剪

            opt.step()
            counter += 1

            if counter % 5 == 0:  # 每5个iter打印并记录一次loss
                writer.add_scalar('train/total_loss', total_loss, counter)
                writer.add_scalar('train/v_loss', v_loss, counter)
                writer.add_scalar('train/sem_loss', sem_loss, counter)
                writer.add_scalar('train/geo_loss', geo_loss, counter)

                v_accuracy, o_accuracy = Occupancy_Accuracy(target, grid_pre, 1)

                print(
                    '---------counter{} & total_loss:{} &  v_loss:{} & sem_loss:{} & geo_loss:{} ------------- '.format(counter, total_loss.item(),
                        v_loss.item(), sem_loss.item(), geo_loss.item()))


                print(
                    '//////////////////counter{} & train/v_accuracy:{} & train/o_accuracy:{}/////////////////////// \n'.format(
                        counter, v_accuracy.item(), o_accuracy.item()))

                writer.add_scalar('train/v_accuracy', v_accuracy, counter)
                writer.add_scalar('train/o_accuracy', o_accuracy, counter)

                if counter % 100 == 0:
                    if baocun == 1:

                        grid_pre = F.softmax(grid_pre, dim=1)
                        grid_pre = grid_pre.permute(0, 2, 3, 4, 1)
                        grid_pre = torch.argmax(grid_pre, dim = -1)
                        fov_mask = fov_mask.reshape(Batch,6,200,200,16)

                        p_dir = "/.../p"
                        t_dir = "/.../t"
                        m_dir = "/.../m"

                        name1 = str(counter) + '_true'
                        name2 = str(counter) + '_pre'
                        name3 = str(counter) + '_mask'
                        t_path = os.path.join(t_dir, name1)
                        p_path = os.path.join(p_dir, name2)
                        m_path = os.path.join(m_dir, name3)

                        p_voxel = grid_pre.cpu().detach().numpy()
                        ture = target.cpu().detach().numpy()
                        mask = fov_mask.cpu().detach().numpy()

                        np.save(t_path, ture)
                        np.save(p_path, p_voxel)
                        np.save(m_path, mask)

                        print('         保存第{}批次的数据     \n'.format(counter))


            if counter % val_step == 0:  # 验证1次，记录loss
                v_total_loss, v_v_loss, v_sem_loss, v_geo_loss, v_acc, o_acc = get_val_info(model, val_loader)

                print('     val counter{} / total_loss:{} &  v_loss: {} & sem_loss: {} & geo_loss:{}       '.format(counter, v_total_loss.item(),
                                                                                                                    v_v_loss.item(), v_sem_loss.item(), v_geo_loss.item()))

                print('     val /  v_accuracy:{} & o_accuracy:{}      '.format(v_acc.item(), o_acc.item()))

                writer.add_scalar('val/total_loss', v_total_loss, counter)
                writer.add_scalar('val/v_loss', v_v_loss, counter)
                writer.add_scalar('val/sem_loss', v_sem_loss, counter)
                writer.add_scalar('val/geo_loss', v_geo_loss, counter)

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


    T1 = time()
    print('整个训练所需最大内存： %.2f M' % (torch.cuda.max_memory_allocated() / 1e6))
    print('整个训练所需时间 {} S'.format((T1 - T0)))
    print('++++++++++++++++++++++++++++++Training End+++++++++++++++++++++++++++++++++++++++')

if __name__ == '__main__':
    train()