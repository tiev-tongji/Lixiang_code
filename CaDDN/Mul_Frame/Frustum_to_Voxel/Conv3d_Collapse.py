import torch.nn as nn



class Conv3DCollapse(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = BasicBlock3D(in_channels=self.in_channels * self.out_channels,
                                  out_channels=self.out_channels)

    def forward(self, depth_features):

        x = depth_features
        depth_feature = self.block(features = x)  # (B, T*N, Y, X, Z) -> (B, N, Y, X, Z)

        return depth_feature




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

    def forward(self, features):

        x = self.conv(features)
        x = self.bn(x)
        x = self.relu(x)

        return x