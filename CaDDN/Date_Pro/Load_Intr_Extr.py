import os
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R


tran = [[336.1, 0, 480.0, 0], [0, 336.1, 320, 0], [0, 0, 1, 0]]
trans = []
for i in range(1600):
    trans.append(tran)


Trans = torch.tensor(np.array(trans), dtype=torch.float32)
np.save(".../Intrinsics.npy", Trans)


Dir = '/.../cam_pose2'

b_left_RT = []
b_front_RT = []
b_rear_RT = []
b_right_RT = []
z_front_RT = []
z_rear_RT = []

last_line = [0,0,0,1]
last_line = np.reshape(last_line, [1,4])
print(np.shape(last_line))

cover = ['b_left_pose.txt','b_front_pose.txt','b_rear_pose.txt','b_right_pose.txt','z_front_pose.txt','z_rear_pose.txt']
for s in cover:
    path = os.path.join(Dir, s)
    print(path)
    iter = 0

    with open(path, encoding='utf-8') as file:
        content = file.readlines()
    for line in content:
        a = line.split(' ')
        iter = iter +1
        print('iter:',iter)

        # 定义 3*1 的 T
        b1 = a[1:4]
        num = map(eval, b1)
        c1 = []
        for i in num:
            c1.append(i)


        T = np.array(c1)
        T = np.reshape(T,[3,1])

        # 定义 3*3 的 R
        b2 = a[4:]
        b2[0] = a[5]
        b2[1] = a[6]
        b2[2] = a[7]
        b2[3] = a[4]
        n = map(eval, b2)
        c2 = []
        for j in n:
            c2.append(j)
        r = R.from_quat(c2)
        r = r.as_matrix()


        RT = np.concatenate((r,T),axis=-1)
        RT = np.concatenate((RT,last_line),axis=0)


        if s == 'b_left_pose.txt':
            b_left_RT.append(RT)
        if s == 'b_front_pose.txt':
            b_front_RT.append(RT)
        if s == 'b_rear_pose.txt':
            b_rear_RT.append(RT)
        if s == 'b_right_pose.txt':
            b_right_RT.append(RT)
        if s == 'z_front_pose.txt':
            z_front_RT.append(RT)
        if s == 'z_rear_pose.txt':
            z_rear_RT.append(RT)

        if iter == 1600:
            break

b_left_RT = np.array(b_left_RT)
b_front_RT = np.array(b_front_RT)
b_rear_RT = np.array(b_rear_RT)
b_right_RT = np.array(b_right_RT)
z_front_RT = np.array(z_front_RT)
z_rear_RT = np.array(z_rear_RT)


z_front_inv = np.linalg.inv(z_front_RT)
b_left = np.matmul(z_front_inv, b_left_RT)
b_front = np.matmul(z_front_inv, b_front_RT)
b_rear  = np.matmul(z_front_inv, b_rear_RT)
b_right = np.matmul(z_front_inv, b_right_RT)
z_front = np.matmul(z_front_inv, z_front_RT)
z_rear = np.matmul(z_front_inv, z_rear_RT)


b_left = np.linalg.inv(b_left )
b_front = np.linalg.inv(b_front)
b_rear  = np.linalg.inv(b_rear)
b_right = np.linalg.inv(b_right)
z_front = np.linalg.inv(z_front)
z_rear = np.linalg.inv(z_rear)



Sum_RT = np.array([b_left, b_front , b_rear, b_right, z_front, z_rear])
Sum_RT = torch.tensor(Sum_RT, dtype=torch.float32)
Sum_RT = Sum_RT.permute(1,0,2,3)


np.save("/,,,,,/Extrinsics.npy", Sum_RT)