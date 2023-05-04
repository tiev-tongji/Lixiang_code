import os
import cv2
import torch
import math
import numpy
import numpy as np
from torchvision.transforms import Resize

def reszie_Depth(depths):

    torch_resize = Resize([240, 360])
    depths = torch.tensor(depths, dtype = torch.float16)
    depths = torch_resize(depths)

    return depths


def ger_depth(depth):
    max = 50.0
    min = 0.0
    m = max - min
    emp = np.empty([640, 960], dtype = numpy.int8)
    R = depth[:, :, 0]

    num = R / 255
    d = m * (1 - num)
    for x in range(640):
        for y in range(960):
            dis_pixel_center_to_corner = math.sqrt(
                round((y + 0.5) - 320) * round((y + 0.5) - 320) + round((x + 0.5) - 480) * round((x + 0.5) - 480))
            depth = d[x][y] * math.cos(math.atan(dis_pixel_center_to_corner * 3 / 80.0 / 12.447))
            image_depth = depth

            emp[x][y] = image_depth


    return emp.reshape(1, 640, 960)


dir = '/..../Depth'
depths = []
for i in range(1600):

    i = i + 1
    print('i:',i)
    Bleft_name = 'Depth (' + str(i) + ').png'
    Bfront_name = 'Depth (' + str(i) + ').png'
    Brear_name = 'Depth (' + str(i) + ').png'
    Bright_name = 'Depth (' + str(i) + ').png'
    Front_name = 'Depth (' + str(i) + ').png'
    Rear_name = 'Depth (' + str(i) + ').png'

    Bleft_path = os.path.join(dir, os.path.join('Bleft', Bleft_name))
    Bfront_path = os.path.join(dir,os.path.join('Bfront', Bfront_name))
    Brear_path = os.path.join(dir, os.path.join('Brear', Brear_name))
    Bright_path = os.path.join(dir, os.path.join('Bright', Bright_name))
    Front_path = os.path.join(dir, os.path.join('Front',  Front_name))
    Rear_path = os.path.join(dir, os.path.join('Rear', Rear_name))


    dep_Bleft = ger_depth(cv2.imread(Bleft_path))
    dep_Bfront = ger_depth(cv2.imread(Bfront_path))
    dep_Brear = ger_depth(cv2.imread(Brear_path))
    dep_Bright = ger_depth(cv2.imread(Bright_path))
    dep_Front = ger_depth(cv2.imread(Front_path))
    dep_Rear = ger_depth(cv2.imread(Rear_path))

    dep_Bleft = reszie_Depth(dep_Bleft).squeeze(-3)
    dep_Bfront = reszie_Depth(dep_Bfront).squeeze(-3)
    dep_Brear = reszie_Depth(dep_Brear).squeeze(-3)
    dep_Bright = reszie_Depth(dep_Bright).squeeze(-3)
    dep_Front = reszie_Depth(dep_Front).squeeze(-3)
    dep_Rear = reszie_Depth(dep_Rear).squeeze(-3)

    Sum_cam = np.array([dep_Bleft, dep_Bfront, dep_Brear, dep_Bright, dep_Front, dep_Rear])
    depths.append(Sum_cam)


Depths = torch.tensor(np.array(depths),dtype=torch.float32)
print(Depths.shape)



np.save("/...../Depth.npy", Depths.squeeze(-3))


