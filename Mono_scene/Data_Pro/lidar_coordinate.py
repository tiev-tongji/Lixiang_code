import torch
import numpy as np






def Lidar_coordinate(rots, trans, lidar_R, lidar_T):

    cam_r = rots
    cam_t = trans
    lid_r = lidar_R
    lid_t = lidar_T


    last_line = [0, 0, 0, 1]
    last_line = np.reshape(last_line, [1, 4])

    lid_t = np.reshape(lid_t, [3, 1])
    lid_rt = np.concatenate((lid_r, lid_t), axis=-1)
    lid_rt = np.concatenate((lid_rt, last_line), axis=0)

    # print("lid_tr.shape:", lid_rt.shape)

    cam_r_0 = cam_r[0, :]
    cam_r_1 = cam_r[1, :]
    cam_r_2 = cam_r[2, :]
    cam_r_3 = cam_r[3, :]
    cam_r_4 = cam_r[4, :]
    cam_r_5 = cam_r[5, :]

    cam_t_0 = cam_t[0, :]
    cam_t_1 = cam_t[1, :]
    cam_t_2 = cam_t[2, :]
    cam_t_3 = cam_t[3, :]
    cam_t_4 = cam_t[4, :]
    cam_t_5 = cam_t[5, :]

    cam_rt_1 = 0
    cam_rt_2 = 0
    cam_rt_3 = 0
    cam_rt_4 = 0
    cam_rt_5 = 0
    cam_rt_0 = 0

    for i in range(6):

        if i == 0:
            t_0 = np.reshape(cam_t_0, [3, 1])
            rt_0 = np.concatenate((cam_r_0, t_0), axis=-1)
            cam_rt_0 = np.concatenate((rt_0, last_line), axis=0)
            # print("cam_rt_0 .shape:", cam_rt_0.shape)

        if i == 1:
            t_1 = np.reshape(cam_t_1, [3, 1])
            rt_1 = np.concatenate((cam_r_1, t_1), axis=-1)
            cam_rt_1 = np.concatenate((rt_1, last_line), axis=0)
            # print("cam_rt_1 .shape:", cam_rt_1.shape)

        if i == 2:
            t_2 = np.reshape(cam_t_2, [3, 1])
            rt_2 = np.concatenate((cam_r_2, t_2), axis=-1)
            cam_rt_2 = np.concatenate((rt_2, last_line), axis=0)
            # print("cam_rt_2 .shape:", cam_rt_2.shape)

        if i == 3:
            t_3 = np.reshape(cam_t_3, [3, 1])
            rt_3 = np.concatenate((cam_r_3, t_3), axis=-1)
            cam_rt_3 = np.concatenate((rt_3, last_line), axis=0)
            # print("cam_rt_3 .shape:", cam_rt_3.shape)

        if i == 4:
            t_4 = np.reshape(cam_t_4, [3, 1])
            rt_4 = np.concatenate((cam_r_4, t_4), axis=-1)
            cam_rt_4 = np.concatenate((rt_4, last_line), axis=0)
            # print("cam_rt_4 .shape:", cam_rt_4.shape)

        if i == 5:
            t_5 = np.reshape(cam_t_5, [3, 1])
            rt_5 = np.concatenate((cam_r_5, t_5), axis=-1)
            cam_rt_5 = np.concatenate((rt_5, last_line), axis=0)
            # print("cam_rt_5 .shape:", cam_rt_5.shape)

    lid_rt_inv = np.linalg.inv(lid_rt)
    # print('lid_tr_inv：',lid_tr_inv)
    rt_0 = np.matmul(lid_rt_inv, cam_rt_0)
    rt_1 = np.matmul(lid_rt_inv, cam_rt_1)
    rt_2 = np.matmul(lid_rt_inv, cam_rt_2)
    rt_3 = np.matmul(lid_rt_inv, cam_rt_3)
    rt_4 = np.matmul(lid_rt_inv, cam_rt_4)
    rt_5 = np.matmul(lid_rt_inv, cam_rt_5)

    Cam_rt = np.array([rt_0, rt_1, rt_2, rt_3, rt_4, rt_5])
    Sum_RT = torch.tensor(Cam_rt)

    Sum_R = Sum_RT[:, 0:3, 0:3]
    Sum_T = Sum_RT[:, 0:3, -1]


    return Sum_R, Sum_T