import torch.nn as nn


class Conv3DCollapse(nn.Module):

    def __init__(self):
        super().__init__()

        self.num_cam = 6
        self.num_bev_features = 104
        self.block = BasicBlock3D(in_channels=self.num_bev_features * self.num_cam,
                                  out_channels=self.num_bev_features)

    def forward(self, voxel_features):

        x = voxel_features
        voxel_features = self.block(features = x)  # (B, N*C, Y, X, Z) -> (B, C, Y, X, Z)

        return voxel_features


class BasicBlock3D(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv3d(in_channels = in_channels,
                              out_channels = out_channels,
                              kernel_size = (3,3,3),
                              stride = (1,1,1),
                              padding = (1,1,1))

        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.Pooling = nn.AvgPool3d((3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

        self.Dropout = nn.Dropout(0.5)

    def forward(self, features):

        x = self.conv(features)
        #x = self.Pooling(x)

        x = self.bn(x)
        x = self.relu(x)
        #x = self.Dropout(x)

        return x