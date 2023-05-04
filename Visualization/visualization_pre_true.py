import numpy
import os
import numpy as np
import torch


p_dir = '/..../p'
t_dir = '/..../t'


voxels = np.load('/..../true.npy', allow_pickle=True)
p_voxel = np.load('/..../pre.npy', allow_pickle=True)

print('p_voxel.shape:', p_voxel.shape)
print('voxel.shape:', voxels.shape)
b,x,y,z = np.shape(voxels)


voxels = torch.tensor(voxels)
print('voxel.shape:', voxels.shape)


p_voxel = torch.tensor(p_voxel)
p_voxel_max = torch.argmax(p_voxel, dim = -1)
print('p_voxel_max.shape:', p_voxel_max.shape)

# 转为 One_Hot
def One_Hot(pre_voxel):

    pre_voxel_max = torch.argmax(pre_voxel, dim = -1)
    pre_voxel_max = np.array(pre_voxel_max.cpu())

    oh_arr = np.eye(2)[pre_voxel_max]
    oh_arr = torch.tensor(oh_arr)

    return oh_arr

# 由 One_Hot 转为 单值
def Inv_OneHot(voxel):

    voxel_max = torch.argmax(voxel, dim = -1)

    return np.array(voxel_max, dtype = numpy.int16)



for i in range(b):

    true_name = 'True_' + str(i) + '.bin'
    pre_name = 'Pre_' + str(i) + '.bin'
    true_path = os.path.join(t_dir, true_name)
    pre_path = os.path.join(p_dir, pre_name)

    Voxel = voxels[i,:]
    P_voxel = p_voxel[i,:]

    p_voxel_max = Inv_OneHot(P_voxel)


    Voxel = np.array(Voxel, dtype=numpy.int16)

    Voxel.tofile(true_path)
    p_voxel_max.tofile(pre_path)

