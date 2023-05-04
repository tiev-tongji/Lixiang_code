import torch.nn as nn
import torch
import torch.nn.functional as F
import os



class Sampler(nn.Module):

    def __init__(self, mode="bilinear", padding_mode="zeros"):

        super().__init__()
        self.mode = mode
        self.padding_mode = padding_mode

    def forward(self, input_features, grid):

        input_features = torch.tensor(input_features,dtype=torch.float)
        grid = torch.tensor(grid, dtype=torch.float)
        # Sample from grid
        output = F.grid_sample(input=input_features, grid=grid, mode=self.mode, padding_mode=self.padding_mode)
        return output