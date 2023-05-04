import torch.nn as nn
from torch import cat
import torch.nn.functional as F

class Conv3d(nn.Module):

    def __init__(self, in_channels, out_channels, k, s, p, batch_norm=True):
        super(Conv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=k, stride=s, padding=p)
        if batch_norm:
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            self.bn = None


    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)

        return F.relu(x, inplace=True)


class unet3dEncoder(nn.Module):

    def __init__(self, in_channels, out_channels, k, s, p):
        super(unet3dEncoder, self).__init__()

        mid_channels = out_channels

        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=k, stride=s, padding=p)
        self.conv2 = nn.Conv3d(mid_channels, out_channels, kernel_size=k, stride=s, padding=p)

        self.bn1 = nn.BatchNorm3d(mid_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool3d((2,2,2), stride=(1,1,1), padding=(1,1,1))

    def forward(self, x):

        #print('  ######## x0 szie:', x.shape)
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)

        #print('  **********x1 szie:', x1.shape)

        x2 = self.conv2(x1)
        x2 = self.bn2(x2)
        x2 = self.relu(x2)
        #print(' ////////// x2 szie:', x2.shape)

        return x2, self.avgpool(x2)

class unet3dUp(nn.Module):

    def __init__(self, in_channels, out_channels, k, s, p):
        super(unet3dUp, self).__init__()

        mid_channels = out_channels
        self.sample1 = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=(2,2,2), stride=(1,1,1), padding=(1,1,1))
        self.sample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        in_chan = in_channels + out_channels
        self.conv1 = nn.Conv3d(in_chan, mid_channels, kernel_size=k, stride=s, padding=p)
        self.conv2 = nn.Conv3d(mid_channels, out_channels, kernel_size=k, stride=s, padding=p)

        self.bn1 = nn.BatchNorm3d(mid_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, x1):

        x = self.sample1(x)
        x = cat((x, x1), dim=1)


        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)

        x2 = self.conv2(x1)
        x2 = self.bn2(x2)
        x2 = self.relu(x2)

        return x2

class OutConv(nn.Module):
    def __init__(self, ):
        super(OutConv, self).__init__()
        #self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=k, stride=s, padding=p)

        self.conv3d_1 = Conv3d(64, 32, 3, s=(1, 1, 1), p=(1, 1, 1))
        self.conv3d_2 = Conv3d(32, 16, 3, s=(1, 1, 1), p=(1, 1, 1))
        self.conv3d_3 = Conv3d(16, 2, 3, s=(1, 1, 1), p=(1, 1, 1))

    def forward(self, x):

        x = self.conv3d_1(x)
        x = self.conv3d_2(x)
        x = self.conv3d_3(x)

        return x

class Unet3D(nn.Module):
    def __init__(self):
        super(Unet3D, self).__init__()

        init_channels = 40

        self.en1 = unet3dEncoder(init_channels, 64, 3, s=(1, 1, 1), p=(1, 1, 1))
        self.en2 = unet3dEncoder(64, 128, 3, s=(1, 1, 1), p=(1, 1, 1))
        self.en3 = unet3dEncoder(128, 256, 3, s=(1, 1, 1), p=(1, 1, 1))
        # self.en4 = unet3dEncoder(256, 512, 3, s=(1, 1, 1), p=(1, 1, 1))

        # self.up3 = unet3dUp(512, 256, 3, s=(1, 1, 1), p=(1, 1, 1))
        self.up2 = unet3dUp(256, 128,3, s=(1, 1, 1), p=(1, 1, 1))
        self.up1 = unet3dUp(128, 64, 3, s=(1, 1, 1), p=(1, 1, 1))

        self.con_last = OutConv()

    def forward(self, x):

        x1,y1 = self.en1(x)
        x2,y2 = self.en2(y1)
        x3,y3 = self.en3(y2)
        # x4,y4 = self.en4(y3)


        # x5 = self.up3(x4, x3)
        # x6 = self.up2(x5, x2)
        # x7 = self.up1(x6, x1)
        # x = self.con_last(x7)


        x4 = self.up2(x3, x2)
        x3 = self.up1(x4, x1)
        x = self.con_last(x3)


        return x