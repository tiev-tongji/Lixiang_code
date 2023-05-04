import torch
import math
import kornia

def project_to_image(project, points):

    device = points.device
    # Reshape tensors to expected shape
    points = kornia.geometry.conversions.convert_points_to_homogeneous(points)
    points = points.unsqueeze(dim=-1).to(device)
    project = project.unsqueeze(dim=1).to(device)


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

    device = coords.device
    min_n = -1
    max_n = 1
    shape = torch.flip(shape, dims=[0]).to(device)  # Reverse ordering of shape #[960 x 640 x 80]

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


def cumulative_warp_features(x, flow, mode='bilinear'):

    out = []
    flow_ttt = flow[:, 0]
    flow_t = flow[:, -1]

    for t in range(3):
        if t == 1 :
            y = x[:, t]
            out.append(y)
            continue

        if t == 0 :
            y = x[:, t]
            grid = torch.nn.functional.affine_grid(flow_ttt, size = y.shape, align_corners=False)
            warped_x = torch.nn.functional.grid_sample(y, grid.float(), mode = mode, padding_mode='zeros',align_corners=False)
            out.append(warped_x)
            continue

        if t == 2 :
            y = x[:, t]
            grid = torch.nn.functional.affine_grid(flow_t, size = y.shape, align_corners=False)
            warped_x = torch.nn.functional.grid_sample(y, grid.float(), mode=mode, padding_mode='zeros', align_corners=False)
            out.append(warped_x)
            break

    return torch.stack(out, dim = 1)