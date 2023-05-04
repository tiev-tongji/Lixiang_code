import torch
from torch import nn
from efficientnet_pytorch.model import EfficientNet



class Up(nn.Module):
    def __init__(self, in_channels = 320+112, out_channels = 512):
        super().__init__()

        self.up = nn.Upsample(size=(15, 22), mode='bilinear',
                              align_corners=True)  # 进行上采样 得到相同的特征大小


        self.conv = nn.Sequential(  # 两个3x3卷积
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),  # inplace=True使用原地操作，节省内存
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)  # 对x1进行上采样
        x1 = torch.cat([x2, x1], dim=1)  # 将x1和x2 concat 在一起
        return self.conv(x1)


class CamEncode(nn.Module):  # 提取图像特征进行图像编码
    def __init__(self):
        super(CamEncode,self).__init__()

        self.D = 40  # 深度bin 80
        self.C = 40  # 特征向量的维度 64

        self.trunk = EfficientNet.from_pretrained("efficientnet-b0")  # 使用 efficientnet 提取特征
        self.up1 = Up(320+112, 512)  # 上采样模块，输入输出通道分别为320+112和512
        self.depthnet = nn.Conv2d(512, self.D + self.C, kernel_size=1, padding=0)  # 1x1卷积，变换维度

    def get_depth_dist(self, x, eps=1e-20):  # 对深度维进行softmax，得到每个像素不同深度的概率
        return x.softmax(dim=1)

    def get_depth_feat(self, x):
        x = self.get_eff_depth(x)  # 使用efficientnet提取特征  x: 24 x 512 x 40 x 60
        # Depth
        x = self.depthnet(x)  # 1x1卷积变换维度  x: 24 x 105(C+D) x  40 x 60

        depth_logits = self.get_depth_dist(x[:, :self.D])  # 第二个维度的前D个作为深度维，进行softmax  depth: 24 x 41 x  60 x 40

        frustum_features = depth_logits.unsqueeze(1) * x[:, self.D:(self.D + self.C)].unsqueeze(2)  # 将特征通道维和通道维利用广播机制相乘

        return frustum_features, depth_logits

    def get_eff_depth(self, x):  # 使用efficientnet提取特征

        endpoints = dict()
        x = torch.tensor(x, dtype=torch.float)
        # Stem
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints)+1)] = prev_x
            prev_x = x

        # Head
        endpoints['reduction_{}'.format(len(endpoints)+1)] = x
        x = self.up1(endpoints['reduction_5'], endpoints['reduction_4'])
        return

    def forward(self, imags):
        frustum_features, depth_logits = self.get_depth_feat(imags)  # depth_logits: B*N x D x fH x fW (24 x 80 x 40 x 60)
                                                                     # frustum_features : B*N x C x D x fH x fW (24 x 64 x 80 x 40 x 60)

        return frustum_features, depth_logits