import torch
import torch.nn as nn
from Unet_2D import UNet2D
from Unet_3D import UNet3D
from FLoSP import FLoSP
from Conv3D_Collapse import Conv3DCollapse


class Mono_Net(nn.Module):

    def __init__(self):
        super().__init__()

        self.projects = {}
        self.n_classes = 2
        self.project_scale = 1                             # 3D occ_grid 的缩放系数
        self.scale_2ds = [1, 2, 4, 8]                      # 2D scales  2DUnet处理后的分辨率
        self.full_scene_size = [200,200,16]                # occ_grid 的大小
        self.project_res = ["1","2","4","8"]

        self.C_2D = UNet2D.build(out_feature = 104 , use_decoder=True)      # 定义输出特征为105，其中深度特征为40，图像特征为64

        self.C_3D = UNet3D(class_num = self.n_classes, norm_layer = nn.BatchNorm3d, project_scale = self.project_scale,
                    feature = 104, full_scene_size = self.full_scene_size)

        self.projects_1 = FLoSP(self.full_scene_size, project_scale=self.project_scale)
        self.projects_2 = FLoSP(self.full_scene_size, project_scale=self.project_scale)
        self.projects_4 = FLoSP(self.full_scene_size, project_scale=self.project_scale)
        self.projects_8 = FLoSP(self.full_scene_size, project_scale=self.project_scale)



        self.Conv3DCollapse = Conv3DCollapse()

    def forward(self, imgs, projected_pix, fov_mask,):   #  imgs（bacth * 6 * 3 * 900 * 16000）
                                                        #  projected_pix (bacth * 6 * 640000 * 2)
                                                        #  fov_mask (bacth * 6 * 640000 )

        imgs = imgs.permute(1, 0, 2, 3, 4)              #（6 * bacth * 3 * 240 * 360)
        cam_num, batch, C, H, W = imgs.shape

        projected_pix = projected_pix.permute(1, 0, 2, 3)        # (6 * bacth * 1260000 * 2)
        fov_mask = fov_mask.permute(1, 0, 2)                     # (6 * bacth * 1260000)


        # 定义存储 占据预测 的空数组
        cam_emp = []

        for i in range(cam_num):

            emp = []                                     # 定义存储处理 FLoSP（将多个分辨率的3维 occ_grid 相加）的空数组
            one_cam = imgs[i,:]                          # 每次取出一个相机的 imags 数据           (bacth * 3 * 240 * 360)
            pix = projected_pix[i,:]                     # 每次取出一个相机的 projected_pix 数据   (bacth * 1260000 * 2)
            mask = fov_mask[i,:]                         # 每次取出一个相机的 fov_mask 数据        (bacth * 1260000)



            x_rgb = self.C_2D(one_cam)                         # 将一个相机的 imags 送出 2D Unet 进行处理
                                                               # 返回多分辨率的2维图像 图像大小于原图一直 {"1_1":（batch * 105 * 240 * 360）, "1_2":..., "1_4":..., "1_8":... }


            for j in range(batch):

                x3d = None
                project = pix[j, :]
                fov = mask[j, :]

                for scale_2d in self.project_res:

                    # project features at each 2D scale to target 3D scale
                    scale = int(scale_2d)

                    # Sum all the 3D features
                    if scale == 1:
                        x3d = self.projects_1(x_rgb["1_" + scale_2d][j,:],  project,  fov,  )

                    if scale == 2:
                        x3d  = x3d + self.projects_2(x_rgb["1_" + scale_2d][j,:], project, fov, )

                    if scale == 4:
                        x3d  = x3d + self.projects_4(x_rgb["1_" + scale_2d][j,:], project, fov, )

                    if scale == 8:
                        x3d  = x3d + self.projects_8(x_rgb["1_" + scale_2d][j,:], project, fov, )


                emp.append(x3d)

            emp_stack = torch.stack(emp, dim=0)


            cam_emp.append(emp_stack)


        voxel_features = torch.stack(cam_emp, dim=0)                  # voxel_features: （6 * batch * 180 * 440 * 50）

        voxel_features = voxel_features.permute(1, 0, 2, 3, 4, 5)     # voxel_features: （batch * 6 * 180 * 440 * 50）

        B, N, C, X, Y, Z = voxel_features.shape
        x = voxel_features.reshape(B, N * C, X, Y, Z)
        voxel_features = self.Conv3DCollapse(voxel_features = x)

        Voxel_pre = self.C_3D(voxel_features)


        return Voxel_pre

def compile_model():
    return Mono_Net()