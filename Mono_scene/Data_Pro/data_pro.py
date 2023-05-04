import os
import torch
import torchvision
import numpy as np
from PIL import Image
from nuscenes import NuScenes
from pyquaternion import Quaternion
from torch.utils.data import Dataset
from torchvision.transforms import Resize
from nuscenes.utils.splits import create_splits_scenes


class NuscData(Dataset):
    def __init__(self, nusc, is_train=True, update_intrinsics=True):

        self.nusc = nusc
        self.is_train = is_train
        self.update_intrinsics = update_intrinsics
        self.scenes = self.get_scenes()
        self.ixes = self.prepro()

        self.normalise_image = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def get_scenes(self):
        # filter by scene split

        split = {
            'v1.0-trainval': {True: 'train', False: 'val'},
            'v1.0-mini': {True: 'mini_train', False: 'mini_val'},
        }[self.nusc.version][self.is_train]

        scenes = create_splits_scenes()[split]

        return scenes

    def prepro(self):
        samples = [samp for samp in self.nusc.sample]

        # remove samples that aren't in this split
        samples = [samp for samp in samples if
                   self.nusc.get('scene', samp['scene_token'])['name'] in self.scenes]

        # sort by scene, timestamp (only to make chronological viz easier)
        samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

        return  samples



    def get_occ_data(self, rec):

        if self.is_train:
            dataroot = '/.../nuscenes_occ'
            sum = 28130

        else:
            dataroot = '/.../nuscenes_occ'
            sum = 6019

        occ = []
        Occ = []
        sensor = 'LIDAR_TOP'
        sensor_sample = self.nusc.get('sample_data', rec['data'][sensor])
        occ_name = sensor_sample['filename']
        occ_name = occ_name + '.npy'
        file_name = os.path.join(dataroot, occ_name)
        occ.append(np.load(file_name))


        for o in occ:

            occ_zero = torch.zeros([200, 200, 16], dtype=torch.int8)
            N = np.shape(o)[0]

            for i in range(N):
                # print(o[i, 0], o[i, 1], o[i, 2], o[i, 3])
                x = o[i, 0]
                y = o[i, 1]
                z = o[i, 2]

                if o[i, 3] > 0 & o[i, 3] <= 16:
                    occ_zero[x, y, z] = 1



            Occ.append(occ_zero)

        occ_ = torch.stack(Occ)
        occ_ = torch.squeeze(occ_,0)



        return occ_

    def get_lidar_RT(self, rec):

        if self.is_train:
            dataroot = '/.../nuscenes_occ'
            sum = 28130

        else:
            dataroot = '/.../nuscenes_occ'
            sum = 6019

        sensor = 'LIDAR_TOP'
        sensor_sample = self.nusc.get('sample_data', rec['data'][sensor])
        sensor = self.nusc.get('calibrated_sensor', sensor_sample['calibrated_sensor_token'])
        tran = torch.Tensor(sensor['translation'])
        rot = torch.Tensor(Quaternion(sensor['rotation']).rotation_matrix)

        lidar_R = rot
        lidar_T = tran

        return lidar_R, lidar_T



    def get_image_data(self, rec, cams):

        def Resize_RGB(Img):
            torch_resize = Resize([225, 400])
            img = torch_resize(Img)

            return img

        imgs = []
        rots = []
        trans = []
        intrins = []
        for cam in cams:
            cam_sample = self.nusc.get('sample_data', rec['data'][cam])


            imgname = os.path.join(self.nusc.dataroot, cam_sample['filename'])
            img = Image.open(imgname)

            cam = self.nusc.get('calibrated_sensor', cam_sample['calibrated_sensor_token'])

            intrin = torch.Tensor(cam['camera_intrinsic'])
            rot = torch.Tensor(Quaternion(cam['rotation']).rotation_matrix)
            tran = torch.Tensor(cam['translation'])

            resize_img = Resize_RGB(Img = img)
            intrin = intrin * (1/4)
            intrin[2, 2] = 1.0
            normalise_img = self.normalise_image(resize_img) # 标准化: ToTensor, Normalize 3,128,352

            # normalise_img = self.normalise_image(img)
            imgs.append(normalise_img)
            intrins.append(intrin)  # 3,3
            rots.append(rot)  # 3,3
            trans.append(tran)  # 3,

        imgs = torch.stack(imgs)
        rots = torch.stack(rots)
        trans = torch.stack(trans)
        intrins = torch.stack(intrins)


        return imgs, rots, trans, intrins


class SegmentationData(NuscData):
    def __init__(self, *args, **kwargs):
        super(SegmentationData, self).__init__(*args, **kwargs)

    def __len__(self):
        return len(self.ixes)

    def __str__(self):
        return f"""NuscData: {len(self)} samples. Split: {"train" if self.is_train else "val"}."""

    def __getitem__(self, index):

        rec = self.ixes[index]  # 按索引取出sample
        cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

        imgs, rots, trans, intrins = self.get_image_data(rec, cams)  # 读取图像数据、相机参数和数据增强的像素坐标映射关系

        occ = self.get_occ_data(rec)

        # lidar_R, lidar_T = self.get_lidar_RT(rec)

        return imgs, rots, trans, intrins, occ,  # lidar_R, lidar_T


def compile_data(batch):


    nusc = NuScenes(version='v1.0-trainval', dataroot='/..../nusense', verbose=True)

    traindata = SegmentationData(nusc=nusc, is_train=True)
    valdata = SegmentationData(nusc=nusc, is_train=False)



    trainloader = torch.utils.data.DataLoader(traindata, batch_size = batch, shuffle = True, num_workers = 4, drop_last = False)
    valloader = torch.utils.data.DataLoader(valdata , batch_size = batch, shuffle = True, num_workers = 4, drop_last = False)


    print('len(trainloader) :', len(trainloader))
    print('len(valloader) :', len(valloader))

    return trainloader,valloader





# trainloader,valloader = compile_data(batch=1)
# i = 0
# for batchi, (imgs, rots, trans, intrins, occ, lidar_R, lidar_T) in enumerate(valloader):
#
#
#     # print('imgs.shape:', imgs.shape)
#     # print('rots.shape:', rots[0,:].shape)
#     # print('trans.shape:', trans[0,:].shape)
#     # print('intrins.shape:', intrins.shape)
#     # print('occ.shape:', occ.shape)
#     # print('lidar_R.shape:', lidar_R[0,:].shape)
#     # print('lidar_T.shape:', lidar_T[0,:].shape)
#
#     if i == :

#         np.save("/...../R.npy", rots)
#         np.save("/...../T.npy", trans)
#         np.save("/...../lidar_R.npy", lidar_R)
#         np.save("/...../lidar_T.npy", lidar_T)
#         np.save("/...../I.npy", intrins)
#
#         break
#
# print('结束')
