import torch.nn as nn
from .Frustum_Feature_Network.depth_ffn import DepthFFE
from .Frustum_to_Voxel.frustum_to_voxel import FrustumToVoxel
from .Voxel3d_Unet import Unet3D



class Modeler(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.depth_ffe = DepthFFE(self.cfg['model'])

        self.frustum_to_voxel = FrustumToVoxel(self.cfg['model'])

        self.U_3D = Unet3D()


    def forward(self, imgs, RT, intrins):

        frustum_features, p_depth = self.depth_ffe(imgs)    # 生成frustum

        voxel = self.frustum_to_voxel(frustum_features, RT, intrins)

        p_voxel = self.U_3D(voxel)             # 对Occupancy voxel进行3D卷积

        return p_voxel, p_depth


def compile_model(cfg):
    return Modeler(cfg)




