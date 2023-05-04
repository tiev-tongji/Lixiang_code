import os
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

Dir = '/home/....cam_pose'

z_front_RT = []
Sum_post = []
last_line = [0,0,0,1]
last_line = np.reshape(last_line, [1,4])

cover = 'z_front_pose.txt'
path = os.path.join(Dir, cover)
print(path)
iter = 0

with open(path, encoding='utf-8') as file:
    content = file.readlines()
for line in content:

    a = line.split(' ')
    iter = iter + 1
    print('iter:', iter)

    b1 = a[1:4]
    num = map(eval, b1)
    c1 = []

    for i in num:
        c1.append(i)
    T = np.array(c1)
    T = np.reshape(T, [3, 1])

    b2 = a[4:]
    n = map(eval, b2)
    c2 = []
    for j in n:
        c2.append(j)
    r = R.from_quat(c2)
    r = r.as_matrix()

    RT = np.concatenate((r, T), axis=-1)
    RT = np.concatenate((RT, last_line), axis=0)
    z_front_RT.append(RT)

    if iter == 1600 :
        break


z_front_RT = np.array(z_front_RT)
print(np.shape(z_front_RT))

Pose = []
for i in range(1600):
    print('i:', i)

    if i == 0 :

        ttt = z_front_RT[i + 1, :, :]
        tt = z_front_RT[i, :, :]
        t = tt
        inv = np.linalg.inv(tt)
        TTT = np.matmul(inv, ttt)
        TT = np.matmul(inv, tt)
        T = np.matmul(inv, t)
        Sum_T = np.array([TTT, TT, T])
        Pose.append(Sum_T)
        continue

    if i == 1600-1 :

        tt = z_front_RT[i, :, :]
        t = z_front_RT[i - 1, :, :]
        ttt = tt
        inv = np.linalg.inv(tt)
        TTT = np.matmul(inv, ttt)
        TT = np.matmul(inv, tt)
        T = np.matmul(inv, t)
        Sum_T = np.array([TTT, TT, T])
        Pose.append(Sum_T)
        break

    ttt =  z_front_RT[i+1, :, :]     # t+1
    tt = z_front_RT[i, :, :]       # t
    t = z_front_RT[i-1, :, :]    # t-1

    inv = np.linalg.inv(tt)
    TTT = np.matmul(inv, ttt)
    TT = np.matmul(inv, tt)
    T = np.matmul(inv, t)
    Sum_T = np.array([TTT, TT, T])
    Pose.append(Sum_T)


Pose = torch.tensor(np.array(Pose), dtype=torch.float32)
np.save("/home/...../Pose.npy", Pose)








