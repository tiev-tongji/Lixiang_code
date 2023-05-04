import torch
import numpy as np
import torch.utils.data


def compile_data(batch):

    # 加载1600
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



    Datas = torch.utils.data.TensorDataset(imgs, voxels, depths, torch.tensor(RT, dtype=torch.float32),torch.tensor(intrins, dtype=torch.float32))
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
    #print('加载训练集所需内存: %.2f M' % (torch.cuda.memory_allocated() / 1e6))
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



