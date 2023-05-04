import torch
import torch.nn as nn
from utils import Upsample



class FLoSP(nn.Module):
    def __init__(self, scene_size, project_scale):
        super().__init__()
        self.scene_size = scene_size             # shape:(180 * 140 * 50)
        self.project_scale = project_scale       # 3D occ_grid 的缩放系数  为1


    def forward(self, x2d, pix, mask,):

        x2d = Upsample(x2d, 225, 400)



        projected_pix = pix
        fov_mask = mask

        c, h, w = x2d.shape
        hw = h * w
        hw = torch.as_tensor(hw, dtype=torch.float32)

        src = x2d.view(c, -1)
        zeros_vec = torch.zeros(c, 1).type_as(src)
        src = torch.cat([src, zeros_vec], 1)


        pix_x, pix_y = projected_pix[:, 0], projected_pix[:, 1]
        img_indices = pix_y * w + pix_x
        img_indices = torch.as_tensor(img_indices, dtype=torch.int32)


        f_m = ~ fov_mask
        f_m = torch.as_tensor(f_m, dtype=torch.bool)


        img_indices[f_m] = hw


        img_indices = img_indices.expand(c, -1).long()  # c, HWD


        src_feature = torch.gather(src, 1, img_indices)


        x3d = src_feature.reshape( c,
            self.scene_size[0] // self.project_scale,
            self.scene_size[1] // self.project_scale,
            self.scene_size[2] // self.project_scale,
        )

        return x3d

