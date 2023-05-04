import os
import cv2
import torch
import numpy as np
from torchvision.transforms import Resize

def reszie_RGB(imgs):

    torch_resize = Resize([240, 360])
    imgs = torch.tensor(imgs, dtype = torch.int8).permute(2, 0, 1)
    imgs = torch_resize(imgs)

    return imgs

dir = '/..../RGB'
imgs = []

for i in range(1600):
    i = i + 1

    Bleft_name = 'Bleft (' + str(i) + ').png'
    Bfront_name = 'Bfront (' + str(i) + ').png'
    Brear_name = 'Brear (' + str(i) + ').png'
    Bright_name = 'Bright (' + str(i) + ').png'
    Front_name = 'Front (' + str(i) + ').png'
    Rear_name = 'Rear (' + str(i) + ').png'

    Bleft_path = os.path.join(dir, os.path.join('Bleft', Bleft_name))
    Bfront_path = os.path.join(dir,os.path.join('Bfront', Bfront_name))
    Brear_path = os.path.join(dir, os.path.join('Brear', Brear_name))
    Bright_path = os.path.join(dir, os.path.join('Bright', Bright_name))
    Front_path = os.path.join(dir, os.path.join('Front',  Front_name))
    Rear_path = os.path.join(dir, os.path.join('Rear', Rear_name))


    img_Bleft = cv2.imread(Bleft_path)
    img_Bfront = cv2.imread(Bfront_path)
    img_Brear = cv2.imread(Brear_path)
    img_Bright = cv2.imread(Bright_path)
    img_Front = cv2.imread(Front_path)
    img_Rear = cv2.imread(Rear_path)

    img_Bleft = reszie_RGB(img_Bleft)
    img_Bfront = reszie_RGB(img_Bfront)
    img_Brear = reszie_RGB(img_Brear)
    img_Bright = reszie_RGB(img_Bright)
    img_Front = reszie_RGB(img_Front)
    img_Rear = reszie_RGB(img_Rear)

    Sum_cam = torch.stack([img_Bleft, img_Bfront, img_Brear, img_Bright, img_Front, img_Rear],dim=0)
    imgs.append(Sum_cam)



Images = torch.stack(imgs,dim=0)


np.save("/,,,,/Imgsr.npy", Images)



