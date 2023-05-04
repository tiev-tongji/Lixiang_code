import torch
import torch.nn as nn
import numpy as np
from .Frustum_to_Voxel.frustum_grid_generator import FrustumGridGenerator
from .Frustum_to_Voxel.frustum_sampler import Sampler
from .Frustum_to_Voxel.Conv3d_Collapse import Conv3DCollapse


class FrustumToVoxel(nn.Module):

    def __init__(self, model_cfg):

        super().__init__()

        self.mod_cfg = model_cfg

        self.disc_cfg = self.mod_cfg['frustum_to_voxel']
        self.grid_size = np.array(self.mod_cfg['grid_size'])
        self.pc_range = np.array(self.mod_cfg['pc_range'])
        self.grid_generator = FrustumGridGenerator(grid_size=self.grid_size,
                                                   pc_range=self.pc_range,
                                                   disc_cfg=self.disc_cfg)
        self.sampler = Sampler()
        self.Conv3DCollapse = Conv3DCollapse()

    def forward(self, frustum_features, RT, intrins):

        emp = []
        batch_feature = frustum_features.permute(1, 0, 2, 3, 4, 5)
        RT = RT.permute(1, 0, 2, 3) # 将B和N的位置进行调换


        for i in range(6):

            X = batch_feature[i,]
            rt = RT[i,]

            grid = self.grid_generator(intrins = intrins, RT = rt)  # (B, X, Y, Z, 3)

            # Sample frustum volume to generate voxel volume
            voxel_features = self.sampler(input_features = X, grid = grid)  # (B, C, X, Y, Z)

            emp.append(voxel_features)


        cc = torch.stack(emp, dim=0)

        # (N, B, C, X, Z, Y) -> (B, N, C, X, Y, Z)
        voxel_features = cc.permute(1, 0, 2, 3, -1, 4)   # 池化 将 (B, N, C, X, Y, Z) -> (B, C, X, Y, Z)
        B, N, C, X, Y, Z = voxel_features.shape


        Nprime = N * C
        x = voxel_features.reshape(B, Nprime, X, Y, Z)

        voxel_features = self.Conv3DCollapse(voxel_features = x)


        return voxel_features


