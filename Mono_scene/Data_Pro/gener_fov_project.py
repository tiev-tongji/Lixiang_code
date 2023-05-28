import torch
import numpy as np
from torch import nn
from numba import prange




def ego_to_cam(point, intrins, rot, trans):

    """
    Transform points (3 x N) from ego frame into a pinhole camera
    """

    point = point.T
    rot = torch.as_tensor(rot, dtype=torch.float32)
    trans = torch.as_tensor(trans, dtype=torch.float32)
    point = torch.as_tensor(point, dtype=torch.float32)
    intrins = torch.as_tensor(intrins, dtype=torch.float32)

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


def vox2pix(cam_k, vox_origin, voxel_size, W, H, scene_size, R, T):

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

        self.img_W = 1600 * 0.25
        self.img_H = 900 * 0.25

        # self.scale = 0.25


    def forward(self, Intrinsic,  R, T):    # Intrinsic 内参  Extrinsic 外参

        data = {}


        fov_mask, projected_pix = vox2pix(cam_k = Intrinsic, vox_origin = self.vox_origin, voxel_size = self.voxel_size,
                                                 W = self.img_W, H = self.img_H, scene_size = self.scene_size, R = R, T = T )

        data["projected_pix"] = projected_pix
        data["fov_mask"] = fov_mask


        return data





def Gener_fov_project(intrins, R, T):

    num_cam = 6
    emp_pix = []
    emp_mask = []
    Data_P = Dataset_Pro()

    for p in range(num_cam):

        Data_set = Data_P(Intrinsic = intrins[p, :, 0:3], R = R[p, :], T =T[p,:])  # Intrinsic 内参  Extrinsic 外参

        fov_mask = torch.as_tensor(Data_set["fov_mask"], dtype=torch.bool)
        projected_pix = torch.tensor(Data_set["projected_pix"], dtype=torch.int32)

        emp_mask.append(fov_mask)
        emp_pix.append(projected_pix)


    projected_pix = torch.stack(emp_pix, dim=0)
    fov_mask = torch.stack(emp_mask, dim=0)



    return fov_mask, projected_pix










