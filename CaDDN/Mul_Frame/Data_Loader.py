import torch
import numpy as np
import torch.utils.data

def compile_data(batch):

    N = 1600
    depths = np.load("/home..../.npy", allow_pickle=True)
    depths = torch.tensor(depths, dtype=torch.float32)
    print('加载深度图', depths.shape)

    voxels = np.load("/home..../.npy", allow_pickle=True)
    voxels = torch.tensor(voxels, dtype=torch.int16)
    print('加载voexl', voxels.shape)

    imgs = np.load("/home..../.npy", allow_pickle=True)
    imgs = torch.tensor(imgs, dtype=torch.float32)
    print('加载imgs', imgs.shape)

    RT = np.load("/home..../.npy", allow_pickle=True)
    RT = torch.tensor(RT, dtype=torch.float32)
    print('加载外参', RT.shape)

    intrins = np.load("/home..../.npy", allow_pickle=True)
    intrins = torch.tensor(intrins, dtype=torch.float32)
    print('加载内参', intrins.shape)

    DoF = np.load("/home..../.npy", allow_pickle=True)
    DoF = torch.tensor(DoF, dtype=torch.float32)
    print('加载pose', DoF.shape)

    # 生出多帧图像数据
    x1 = imgs[N-1,:].unsqueeze(0)       # 第1600帧
    x3 = imgs[0,:].unsqueeze(0)         # 第1帧

    t1 = imgs[0:N-1,:]                  # 1-1599帧
    t3 = imgs[1:N,:]                    # 2-1600帧

    imgs_1 = torch.cat((x3, t1),  0)     # t-1
    imgs_3 = torch.cat((t3, x1),  0)     # t+1

    Imgs = torch.stack((imgs_1, imgs, imgs_3), 0)
    Imgs =  Imgs.permute(1, 0, 2, 3, 4, 5)        # size: 1600*3*6*3*240*360




    # 生出多帧外参RT
    X1 = RT[N - 1, :].unsqueeze(0)         # 第1600帧
    X3 = RT[0, :].unsqueeze(0)             # 第1帧

    T1 = RT[0:N - 1, :]                    # 1-1599帧
    T3 = RT[1:N, :]                        # 2-1600帧

    RT_1 = torch.cat((X3, T1), 0)          # t-1
    RT_3 = torch.cat((T3, X1), 0)          # t+1

    RT = torch.stack((RT_1, RT, RT_3), 0)
    RT = RT.permute(1, 0, 2, 3, 4)          # size: 1600*3*6*4*4




    Datas = torch.utils.data.TensorDataset(Imgs, voxels, depths, RT, intrins, DoF)
    print('Datas:',len(Datas))

    traindata, valdata, tsetdata = torch.utils.data.random_split(Datas, [0.7, 0.2, 0.1], generator=torch.manual_seed(0))


    print('train data:',len(traindata))
    print('val data:', len(valdata))
    print('tset data:', len(tsetdata))

    b_size = batch
    trainloader = torch.utils.data.DataLoader(traindata, batch_size = b_size,
                                              shuffle = True,
                                              num_workers = 4,
                                              drop_last=False)  # 给每个线程设置随机种子


    valloadar = torch.utils.data.DataLoader(valdata, batch_size = b_size,
                                            shuffle = True,
                                            drop_last = False)

    tsetloadar = torch.utils.data.DataLoader(tsetdata, batch_size = b_size,
                                            shuffle = True,
                                            drop_last=False)
    print('trainloader:',len(trainloader))
    print('valloadar:', len(valloadar))
    print('tsetloadar:', len(tsetloadar))


    return trainloader, valloadar, tsetloadar
