import os
import torch
import shutil
import numpy as np

val_class_weights = np.zeros(17)
train_class_weights = np.zeros(17)

val_file = '/.../nuscenes_occ/val/nuscenes_occ/samples'
train_file = '/.../nuscenes_occ/train/nuscenes_occ/samples'

val_names = os.listdir(val_file)
train_names = os.listdir(train_file)


for name in val_names:
    file_name = os.path.join(val_file, name)
    data = np.load(file_name)


    N,n = data.shape
    for i in range(N):
        index = data[i,3]
        val_class_weights[index] = val_class_weights[index] + 1

    print('val_class_weights:', val_class_weights)


for names in train_names:
    file_names = os.path.join(train_file, names)
    datas = np.load(file_names)


    NN,nn = datas.shape
    for i in range(NN):
        indexs = datas[i,3]
        train_class_weights[indexs] = train_class_weights[indexs] + 1

    print('train_class_weights:', train_class_weights)



sum = val_class_weights + train_class_weights
print()
print('Sum_train_class_weights:', train_class_weights)
print('Sum_val_class_weights:', val_class_weights)

for j in range(17):
    print(sum[j])

np.save("/.../class_weights.npy", sum)