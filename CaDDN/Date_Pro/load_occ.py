import kornia
import numpy as np
import torch
import os
import re
import gzip
import shutil


'''
# 解压occ数据
dir = '/..../voxel_gz_new_0302'
Dir = '/..../un_voxel_gz_new_0302'


for name in os.listdir(dir):
    print('name:', name)
    path = os.path.join(dir, name)
    print('path:', path)
    N = name.rsplit('.')
    new_name = N[0] + '.bin'
    print('new name:',new_name)
    Path = os.path.join(Dir, new_name)
    print('Ptah:',Path)
    with gzip.open(path, 'rb') as f_in:
        with open(Path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

'''



Dir = '/..../un_voxel_gz_new_0302'
Voxels = []
for i in range(1600):
    i = i+1
    name = 'voxel_' + str(i) + '.bin'
    path = os.path.join(Dir,name)
    print('path:',path)
    voxel = np.fromfile(path, dtype=np.int16, count = -1 )

    voxel = voxel / 1000

    voxel[voxel > 0.9] = 1
    voxel[voxel < 0.9] = 0

    print(np.shape(voxel))
    print(voxel)
    V = voxel.reshape([180, 140, 50])

    print(V.shape)
    Voxels.append(V)


Voxels = torch.tensor(np.array(Voxels),dtype=torch.int8)
np.save("/..../V.npy", Voxels)









