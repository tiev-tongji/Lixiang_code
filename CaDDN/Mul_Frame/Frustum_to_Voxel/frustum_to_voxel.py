import torch
import numpy as np
import torch.nn as nn
from .utils import cumulative_warp_features
from .frustum_sampler import Sampler
from .Conv3d_Collapse import Conv3DCollapse
from .frustum_grid_generator import FrustumGridGenerator

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
        self.Conv3DCollapse_N = Conv3DCollapse(in_channels=6, out_channels=40)
        self.Conv3DCollapse_T = Conv3DCollapse(in_channels=3, out_channels=40)

    def forward(self, frustum_features, RT, intrins, DoF):

        # frustum_features size: B*3*6*c*d*h*w
        # trans size: B*3*4
        # Rt size:  B*3*6*4*4
        # DoF size:  B*3*4*4

        batch_feature = frustum_features.permute(1, 2, 0, 3, 4, 5, 6)  # B*3*6*c*d*h*w  --> 3*6*B*c*d*h*w
        rt = RT.permute(1, 2, 0, 3, 4)  # 将B 与 N和T的位置进行调换

        all_time = []

        for j in range(3):

            emp = []
            batch_fea = batch_feature[j, ]
            RT = rt[j, ]

            for i in range(6):

                X = batch_fea[i, ]
                rt = RT[i, ]

                grid = self.grid_generator(intrins=intrins, RT=rt)  # (B, X, Y, Z, 3)

                # Sample frustum volume to generate voxel volume
                voxel_features = self.sampler(input_features=X, grid=grid)  # (B, C, X, Y, Z)
                emp.append(voxel_features)

            cc = torch.stack(emp, dim=0)

            # (N, B, C, X, Y, Z) -> (B, N, C, X, Y, Z)
            voxel_features = cc.permute(1, 0, 2, 3, -1, 4)

            # 池化 将 (B, N, C, X, Y, Z) -> (B, C, X, Y, Z)
            B, N, C, X, Y, Z = voxel_features.shape
            x = voxel_features.reshape(B, N*C, X, Y, Z)
            voxel_features = self.Conv3DCollapse_N(x)

            all_time.append(voxel_features)

        All = torch.stack(all_time, dim=0)
        All = All.permute(1, 0, 2, 3, 4, 5)

        DoF = DoF[:, :, 0:3, :]
        voxel_feature = cumulative_warp_features(All, DoF, mode='bilinear')

        b, t, c, x, y, z = voxel_feature.shape
        voxel_feature =voxel_feature.reshape(b, t*c, x, y, z)
        voxel_features = self.Conv3DCollapse_T(voxel_feature)


        return voxel_features