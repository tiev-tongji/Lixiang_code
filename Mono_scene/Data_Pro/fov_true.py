import torch
import numpy as np


def Fov_true(T, fov_mask):

    # true = torch.tensor(true, dtype=torch.int16)
    # fov_mask = torch.tensor(fov_mask, dtype=torch.int16)

    true = T.clone()
    mask = fov_mask.clone()
    true = true.reshape(-1)

    mask_0 = mask[0, :].reshape(-1)
    mask_1 = mask[1, :].reshape(-1)
    mask_2 = mask[2, :].reshape(-1)
    mask_3 = mask[3, :].reshape(-1)
    mask_4 = mask[4, :].reshape(-1)
    mask_5 = mask[5, :].reshape(-1)

    mask_all = torch.logical_or((mask_0 == True), torch.logical_or((mask_1 == True),torch.logical_or((mask_2 == True),
                                torch.logical_or((mask_3 == True),torch.logical_or((mask_4 == True),(mask_5 == True))))))

    true[~mask_all] = 18

    return true.reshape(200,200,16)


