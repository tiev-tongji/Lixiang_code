import torch.nn as nn
from Frustum_Feature_Network.DDN import CamEncode

class DepthFFE(nn.Module):
    def __init__(self, model_cfg):
        super(DepthFFE, self).__init__()

        # self.model_cfg = model_cfg['depth_ffe']
        # self.disc_cfg = self.model_cfg['DISCRETIZE']

        self.ddn = CamEncode()


    def forward(self, imags):

        b, n, c, h, w = imags.shape

        img = imags.view(b * n, c, h, w)  # 将batch维度和相机维度结合起来 [torch.Tensor(24 x 3 x 640 x 960)]

        frustum_features, depth_logits = self.ddn(img)  # 进行图像编 并返回frustum features和 depth预测
                                                        # frustum_features：[torch.Tensor(24 x 64 x 80 x 40 x 60)]
                                                        # depth_logits：[torch.Tensor(24 x 80 x 60 x 40)]

        B, C, D, H, W = frustum_features.shape
        frustum_features = frustum_features.view(b, n, C, D, H, W)   # frustum_features：[torch.Tensor(4 x 6 x 64 x 80 x 40 x 60)]
        depth_logits = depth_logits.view(b, n, D, H, W)      # depth_logits：[torch.Tensor(4 x 6 x 80 x 40x 60)]

        return frustum_features, depth_logits




