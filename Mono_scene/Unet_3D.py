import torch.nn as nn
from utils import Upsample_3D
from modules import SegmentationHead
from modules import Process, Upsample, Downsample


class UNet3D(nn.Module):
    def __init__(
        self,
        class_num,
        norm_layer,
        full_scene_size,
        feature,
        project_scale,
        bn_momentum = 0.1,
    ):
        super(UNet3D, self).__init__()

        self.project_scale = project_scale
        self.full_scene_size = full_scene_size
        self.feature = feature
        self.dilations = [1, 2, 3]


        self.process_l1 = nn.Sequential(
            Process(self.feature, norm_layer, bn_momentum, dilations=self.dilations),
            Downsample(self.feature, norm_layer, bn_momentum),
        )
        self.process_l2 = nn.Sequential(
            Process(self.feature * 2, norm_layer, bn_momentum, dilations=self.dilations),
            Downsample(self.feature * 2, norm_layer, bn_momentum),
        )

        self.up_13_l2 = Upsample(
            self.feature * 4, self.feature * 2, norm_layer, bn_momentum
        )
        self.up_12_l1 = Upsample(
            self.feature * 2, self.feature, norm_layer, bn_momentum
        )
        self.up_l1_full = Upsample(
            self.feature, self.feature // 2, norm_layer, bn_momentum
        )

        self.ssc_head = SegmentationHead(
            self.feature , self.feature , class_num, self.dilations
        )


    def forward(self, input_dict):

        # x3d_l1 = input_dict
        # print('x3d_l1.shape:', x3d_l1.shape)
        #
        # x3d_l2 = self.process_l1(x3d_l1)
        # print('x3d_l2.shape:', x3d_l2.shape)
        #
        # x3d_l3 = self.process_l2(x3d_l2)
        # print('x3d_l3.shape:', x3d_l3.shape)
        #
        # B, C, D, H, W = x3d_l2.shape
        # x3d_up_l2 = Upsample_3D(self.up_13_l2(x3d_l3), D, H, W) + x3d_l2
        # print('x3d_up_l2.shape:', x3d_up_l2.shape)
        #
        # x3d_up_l1 = self.up_12_l1(x3d_up_l2) + x3d_l1
        # print('x3d_up_l1.shape:', x3d_up_l1.shape)
        #
        # x3d_up_full = self.up_l1_full(x3d_up_l1)
        # print('x3d_up_full.shape:', x3d_up_full.shape)
        #
        # ssc_logit_full = self.ssc_head(x3d_up_l1)
        # #print('ssc_logit_full.shape:', ssc_logit_full.shape)


        x3d_l1 = input_dict

        x3d_l2 = self.process_l1(x3d_l1)

        x3d_l3 = self.process_l2(x3d_l2)

        B, C, D, H, W = x3d_l2.shape
        x3d_up_l2 = Upsample_3D(self.up_13_l2(x3d_l3), D, H, W) + x3d_l2

        x3d_up_l1 = self.up_12_l1(x3d_up_l2) + x3d_l1

        # x3d_up_full = self.up_l1_full(x3d_up_l1)

        ssc_logit_full = self.ssc_head(x3d_up_l1)

        return ssc_logit_full

