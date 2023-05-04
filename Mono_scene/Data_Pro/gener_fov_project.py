import torch
import numpy as np
from torch import nn
from numba import prange

def Intrinsics(i):

    intrins = np.load("/.../I.npy", allow_pickle=True)
    intrins = intrins[0, i, :]

    return intrins


def Extrinsics(i):

    last_line = [0, 0, 0, 1]
    last_line = np.reshape(last_line, [1, 4])

    R = np.load("/..../lidar_R.npy", allow_pickle=True)
    T = np.load("/..../lidar_T.npy", allow_pickle=True)


    if i == 0:
        r_0 = R[0, :]
        t_0 = T[0, :]
        T_0 = np.reshape(t_0, [3, 1])

        RT_0 = np.concatenate((r_0, T_0), axis=-1)
        RT_0 = np.concatenate((RT_0, last_line), axis=0)

        return r_0, t_0, RT_0

    if i == 1:
        r_1 = R[1, :]
        t_1 = T[1, :]
        T_1 = np.reshape(t_1, [3, 1])

        RT_1 = np.concatenate((r_1, T_1), axis=-1)
        RT_1 = np.concatenate((RT_1, last_line), axis=0)

        return r_1, t_1, RT_1

    if i == 2:
        r_2 = R[2, :]
        t_2 = T[2, :]
        T_2 = np.reshape(t_2, [3, 1])

        RT_2 = np.concatenate((r_2, T_2), axis=-1)
        RT_2 = np.concatenate((RT_2, last_line), axis=0)

        return r_2, t_2, RT_2

    if i == 3:
        r_3 = R[3, :]
        t_3 = T[3, :]
        T_3 = np.reshape(t_3, [3, 1])

        RT_3 = np.concatenate((r_3, T_3), axis=-1)
        RT_3 = np.concatenate((RT_3, last_line), axis=0)

        return r_3, t_3, RT_3

    if i == 4:
        r_4 = R[4, :]
        t_4 = T[4, :]
        T_4 = np.reshape(t_4, [3, 1])

        RT_4 = np.concatenate((r_4, T_4), axis=-1)
        RT_4 = np.concatenate((RT_4, last_line), axis=0)

        return r_4, t_4, RT_4

    if i == 5:
        r_5 = R[5, :]
        t_5 = T[5, :]
        T_5 = np.reshape(t_5, [3, 1])

        RT_5 = np.concatenate((r_5, T_5), axis=-1)
        RT_5 = np.concatenate((RT_5, last_line), axis=0)

        return r_5, t_5, RT_5


def ego_to_cam(point, intrins, rot, trans):

    """
    Transform points (3 x N) from ego frame into a pinhole camera
    """

    point = point.T
    rot = torch.tensor(rot, dtype=torch.float32)
    trans = torch.tensor(trans, dtype=torch.float32)
    point = torch.tensor(point, dtype=torch.float32)
    intrins = torch.tensor(intrins, dtype=torch.float32)

    point = point - trans.unsqueeze(1)
    point = rot.permute(1, 0).matmul(point)
    point = intrins.matmul(point)
    point[:2] /= point[2:3]

    point = point.numpy()
    points = point.T


    return points


def vox2world(vol_origin, vox_coords, vox_size, offsets=(0.5, 0.5, 0.5)):

    vol_origin = vol_origin.astype(np.float32)
    vox_coords = vox_coords.astype(np.float32)
    cam_pts = np.empty_like(vox_coords, dtype=np.float32)

    for i in prange(vox_coords.shape[0]):
        for j in range(3):
            cam_pts[i, j] = (
                    vol_origin[j]
                    + (vox_size * vox_coords[i, j])
                    + vox_size * offsets[j]
            )
    return cam_pts


def vox2pix(cam_E, cam_k, vox_origin, voxel_size, W, H, scene_size, R, T):

    # Compute the x, y, z bounding of the scene in meter

    vol_bnds = np.zeros((3, 2))
    vol_bnds[:, 0] = vox_origin
    vol_bnds[:, 1] = vox_origin + np.array(scene_size)


    # Compute the voxels centroids in lidar cooridnates
    vol_dim = np.ceil((vol_bnds[:, 1] - vol_bnds[:, 0]) / voxel_size).copy(order='C').astype(int)

    xv, yv, zv = np.meshgrid(
        range(vol_dim[0]),
        range(vol_dim[1]),
        range(vol_dim[2]),
        indexing='ij')

    vox_coords = np.concatenate([
        xv.reshape(1, -1),
        yv.reshape(1, -1),
        zv.reshape(1, -1)], axis=0).astype(int).T
    print('vox_coords.shape:', np.shape(vox_coords))
    print('vox_coords:',vox_coords)



    cam_pts = vox2world(vox_origin, vox_coords, voxel_size)

    pts = ego_to_cam(cam_pts, cam_k, R, T)

    projected_pix = pts[:, 0:2]

    fov_mask = (pts[:,2] > 0) & (pts[:,0] >=0) & (pts[:,0] < W ) & (pts[:,1] >= 0) & (pts[:,1] < H )

    return fov_mask, projected_pix


class Dataset_Pro(nn.Module):
    def __init__(self,  frustum_size = 4):
        super().__init__()

        self.frustum_size = frustum_size

        self.vox_origin = np.array([-50.0, -50.0, -5.0])
        self.scene_size = (100.0, 100.0, 8.0)
        self.voxel_size = 0.5


        self.img_W = 400
        self.img_H = 225
        self.scale = 0.25


    def forward(self, Intrinsic,  R, T):    # Intrinsic 内参  Extrinsic 外参

        data = {}

        # compute the 3D-2D mapping
        print('R:', np.shape(R))
        print('T:', np.shape(T))
        print('Intrinsic:', np.shape(Intrinsic))


        print('原始内参：', Intrinsic)
        Intrinsic = Intrinsic * self.scale
        Intrinsic[2,2] = 1.0
        print('缩放后内参：', Intrinsic)

        fov_mask, projected_pix = vox2pix(cam_k = Intrinsic, vox_origin = self.vox_origin, voxel_size = self.voxel_size,
                                                 W = self.img_W, H = self.img_H, scene_size = self.scene_size, R = R, T = T )

        data["projected_pix"] = projected_pix
        data["fov_mask"] = fov_mask


        return data



num_cam = 6
emp_pix = []
emp_mask = []
Data_P = Dataset_Pro()

for p in range(num_cam):

    Intr = Intrinsics(i=p)                 # shape( 3 * 3 )
    R, T, RT = Extrinsics(i = p)           # shape( 4 * 4 )

    Data_set = Data_P(Intrinsic = Intr[:,0:3], R = R, T = T)  # Intrinsic 内参  Extrinsic 外参

    fov_mask = torch.tensor(Data_set["fov_mask"], dtype=torch.int32)
    projected_pix = torch.tensor(Data_set["projected_pix"], dtype=torch.int32)


    emp_mask.append(fov_mask)
    emp_pix.append(projected_pix)

    print('   完成第{}个相机的处理  '.format(p+1))



projected_pix = torch.stack(emp_pix, dim=0)
fov_mask = torch.stack(emp_mask, dim=0)


# fov_mask = fov_mask.reshape(-1,200,200,16)
# projected_pix = projected_pix.reshape(-1,200,200,16,2)

print('fov_mask.shape:  ',fov_mask.shape)
print('projected_pix.shape:  ',projected_pix.shape)


# np.save("/..../projected_pix.npy", projected_pix)
# np.save("/..../fov_mask.npy", fov_mask)


# print('++++++++++++++++++++++++++++++开始转换+++++++++++++++++++++++++++++++++++++++')
# fov_mask = fov_mask.reshape(-1,200,200,16)
# print('加载fov_mask', fov_mask.shape)
#
# occ = torch.zeros([200, 200, 16], dtype=torch.int16)
#
# for p in range(6):
#     mask = fov_mask[p, :]
#
#     for x in range(0, 200, 1):
#         for y in range(0, 200, 1):
#             for z in range(0, 16, 1):
#
#                 if mask[x, y, z] == 1:
#                     occ[x, y, z] = 1
#
#     print('   完成第{}个相机的处理  '.format(p + 1))
#
#
# np.save("/...../fov_mask_3D.npy", occ)



