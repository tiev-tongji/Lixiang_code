import torch
import numpy as np
import torch.utils.data

def load_project_mask():

    projected_pix = np.load("/..../projected_pix.npy", allow_pickle=True)
    fov_mask = np.load("/...../fov_mask.npy", allow_pickle=True)


    projected_pix = torch.tensor(projected_pix, dtype=torch.int16)
    fov_mask = torch.tensor(fov_mask, dtype=torch.int16)

    print('加载projected_pix: ', projected_pix.shape)
    print('加载fov_mask: ', fov_mask.shape)


    return projected_pix, fov_mask

