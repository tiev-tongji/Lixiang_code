import torch
import torch.nn as nn
import kornia
import math

def project_to_image(project, points):

    # Reshape tensors to expected shape
    points = kornia.geometry.conversions.convert_points_to_homogeneous(points)
    points = points.unsqueeze(dim=-1)
    project = project.unsqueeze(dim=1)

    # Transform points to image and get depths
    points_t = project.to(torch.double) @ points.to(torch.double)
    points_t = points_t.squeeze(dim=-1)
    points_img = kornia.geometry.conversions.convert_points_from_homogeneous(points_t)
    points_depth = points_t[..., -1] - project[..., 2, 3]

    return points_img, points_depth


def normalize_coords(coords, shape):
    """
    Normalize coordinates of a grid between [-1, 1]
    Args:
        coords [torch.Tensor(..., 2)]: Coordinates in grid
        shape [torch.Tensor(2)]: Grid shape [H, W]
    Returns:
        norm_coords [torch.Tensor(.., 2)]: Normalized coordinates in grid
    """
    min_n = -1
    max_n = 1
    shape = torch.flip(shape, dims=[0])  # Reverse ordering of shape #[960 x 640 x 80]

    norm_coords = coords / (shape - 1) * (max_n - min_n) + min_n
    return norm_coords


def bin_depths(depth_map, mode, depth_min, depth_max, num_bins, target = True):

    if mode == "UD":
        bin_size = (depth_max - depth_min) / num_bins
        indices = ((depth_map - depth_min) / bin_size)
    elif mode == "LID":
        bin_size = 2 * (depth_max - depth_min) / (num_bins * (1 + num_bins))
        indices = -0.5 + 0.5 * torch.sqrt(1 + 8 * (depth_map - depth_min) / bin_size)
    elif mode == "SID":
        indices = num_bins * (torch.log(1 + depth_map) - math.log(1 + depth_min)) / \
            (math.log(1 + depth_max) - math.log(1 + depth_min))
    else:
        raise NotImplementedError

    if target:
        # Remove indicies outside of bounds
        mask = (indices < 0) | (indices > num_bins) | (~torch.isfinite(indices))
        indices[mask] = num_bins

        # Convert to integer
        indices = indices.type(torch.int64)
    return indices


class FrustumGridGenerator(nn.Module):

    def __init__(self, grid_size, pc_range, disc_cfg):

        super().__init__()
        self.dtype = torch.float64
        self.grid_size = torch.as_tensor(grid_size)  # Occupancy_grid 的大小[360 x 280 x 100]
        self.pc_range = pc_range
        self.out_of_bounds_val = -2
        self.disc_cfg = disc_cfg

        # 计算出 voxel size
        pc_range = torch.as_tensor(pc_range).reshape(2, 3)
        self.pc_min = pc_range[0]
        self.pc_max = pc_range[1]
        self.voxel_size = (self.pc_max - self.pc_min) / self.grid_size  # 每个voxel 的大小 [0.05 x 0.05 x 0.05]


        # 创建 voxel grid
        self.depth, self.width, self.height = self.grid_size.int()
        self.voxel_grid = kornia.utils.create_meshgrid3d(depth=self.depth,
                                                         height=self.height,
                                                         width=self.width,
                                                         normalized_coordinates=False) #[1 x 360 x 280 x 100 x 3]

        # Add offsets to center of voxel
        self.voxel_grid += 0.5
        self.un_project = self.Unproject(pc_min=self.pc_min, voxel_size=self.voxel_size)



    def Unproject(self, pc_min, voxel_size):


        x_size, y_size, z_size = voxel_size
        x_min, y_min, z_min = pc_min
        un_project = torch.tensor([[x_size, 0, 0, x_min],
                                  [0, y_size, 0, y_min],
                                  [0,  0, z_size, z_min],
                                  [0,  0, 0, 1]],
                                 dtype=self.dtype)  # (4, 4)

        return un_project


    def transform_grid(self, voxel_grid, un_project, cam_to_img, RT):

        device = RT.device
        B = cam_to_img.shape[0]

        # Create transformation matricies
        Tran = (RT.to(device)).to(torch.float) @ (un_project.to(device)).to(torch.float)  #(4, 4) -> （B，4, 4)
        Tran = Tran.reshape(B, 1, 1, 4, 4)
        Tran = torch.as_tensor(Tran, dtype=torch.double)

        # Reshape to match dimensions
        voxel_grid = voxel_grid.repeat_interleave(repeats=B, dim=0)
        voxel_grid = torch.as_tensor(voxel_grid, dtype=torch.double)

        # Transform to camera frame
        camera_grid = kornia.geometry.linalg.transform_points(trans_01=Tran.to(device), points_1=voxel_grid.to(device)) #[B x 360 x 280 x 100 x 3]

        # Project to image
        I_C = cam_to_img.to(device)                     # Camera -> Image (B, 3, 4)
        I_C = I_C.reshape(B, 1, 1, 3, 4)
        image_grid, image_depths = project_to_image(project=I_C, points=camera_grid)

        # Convert depths to depth bins
        image_depths = bin_depths(depth_map=image_depths, mode='LID', depth_min= 2.0, depth_max=46.80, num_bins = 40, target = True ) #[B x 360 x 280 x 100]

        # Stack to form frustum grid
        image_depths = image_depths.unsqueeze(-1)
        frustum_grid = torch.cat((image_grid, image_depths), dim=-1)
        return frustum_grid



    def forward(self, intrins, RT):

        frustum_grid = self.transform_grid(voxel_grid = self.voxel_grid,
                                           un_project = self.un_project,
                                           cam_to_img = intrins,
                                           RT = RT)

        image_shape = [240, 360]
        image_shape = torch.tensor(image_shape)

        intrins = intrins * (3 / 8)
        intrins[2, 2] = 1.0

        image_depth = torch.tensor([40], device=image_shape.device, dtype=image_shape.dtype)
        frustum_shape = torch.cat((image_depth, image_shape))  # [80 x 640 x 960]

        frustum_grid = normalize_coords(coords=frustum_grid, shape=frustum_shape) #[B x 360 x 280 x 100 x 3]

        # Replace any NaNs or infinites with out of bounds
        mask = ~torch.isfinite(frustum_grid)
        frustum_grid[mask] = self.out_of_bounds_val

        return frustum_grid