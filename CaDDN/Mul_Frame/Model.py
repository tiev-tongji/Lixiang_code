import torch.nn as nn
from CaDDN.Mul_Frame.Frustum_Feature_Network.Depth_ffn import DepthFFE
from CaDDN.Mul_Frame.Frustum_to_Voxel.frustum_to_voxel import FrustumToVoxel
from CaDDN.Mul_Frame.Frustum_to_Voxel.Conv3d_Collapse import Conv3DCollapse
from Voxel3d_Unet import Unet3D


class Modeler(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.depth_ffe = DepthFFE(self.cfg['model'])

        self.frustum_to_voxel = FrustumToVoxel(self.cfg['model'])
        self.U_3D = Unet3D()
        self.Conv3DCollapse_T = Conv3DCollapse(in_channels=3, out_channels=6)


    def forward(self, imgs,  RT, intrins, DoF):
        # imgs size: B*3*6*3*240*360
        # trans size: B*3*4
        # Rt size:  B*3*6*4*4
        # DoF size:  B*3*4*4

        B, T, N, C, H, W = imgs.shape
        imgs = imgs.view(B * T, N, C, H, W)
        frustum_features, p_depth = self.depth_ffe(imgs)  # 生成frustum size: (B*T)*n*c*d*h*d   p_depth size: (B*T)*n*c*h*d

        b, n, c, d, h, w = frustum_features.shape
        frustum_features = frustum_features.view(B, T, n, c, d, h, w)
        p_depth = p_depth.view(B, T, n, c, h, w)
        p_depth = p_depth.view(B, T * n, c, h, w)
        p_depth = self.Conv3DCollapse_T(p_depth)

        voxel = self.frustum_to_voxel(frustum_features, RT, intrins, DoF)  # 生成Occupancy voxel

        p_voxel = self.U_3D(voxel)  # 对Occupancy voxel进行3D卷积


        return p_voxel, p_depth




def compile_model(cfg):
        return Modeler(cfg)





















