import numpy as np

class_frequencies = np.array(
    [
        26045845.0,
        7857862.0,
        448871.0,
        6261496.0,
        51071127.0,

        4501781.0,
        580789.0,
        9361000.0,
        1682942.0,
        7810659.0,

        17623642.0,
        390948608.0,
        12220628.0,
        125580676.0,
        182596987.0,

        438889498.0,
        447609890.0
    ]
)
kitti_class_names = \
    [

    'void',
    'barrier',
    'bicycle',
    'bus',
    'car',

    'construction_vehicle',
    'motorcycle',
    'pedestrian',
    'traffic_cone',
    'trailer',

    'truck',
    'driveable_surface',
    'other_flat',
    'sidewalk',
    'terrain',

    'manmade',
    'vegetation'

     ]