import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import utils

class Resampler(nn.Module):

    def __init__(self, resolution, extents):
        super().__init__()
        # resolution: 0.5
        # grid_extents: [-25, 39, 25, 50], [-25, 19.5, 25, 39], [-25, 9.5, 25, 19.5], [-25, 4.5, 25, 9.5], [-25, 1, 25, 4.5]

        # Store z positions of the near and far planes
        self.near = extents[1]
        self.far = extents[3]

        # Make a grid in the x-z plane
        self.grid = _make_grid(resolution, extents) # [98][100][2]


    def forward(self, features, calib):
        # features: [batch, channel=64, out_depth, width]
        # calib: [batch_size][3][3]
        # [ [1252.8131021185304,0.0,826.588114781398],  : intrinsics[0]
        #   [0.0,1252.8131021185304,469.9846626224581], : intrinsics[1]
        #   [0.0,0.0,1.0] ]

        # Copy grid to the correct device
        self.grid = self.grid.to(features)
        
        # We ignore the image v-coordinate, and assume the world Y-coordinate
        # is zero, so we only need a 2x2 submatrix of the original 3x3 matrix
        # [[fx, u], [0, 1]]
        # [batch_size][2][2] -> [batch_size][1][1][2][2]
        calib = calib[:, [0, 2]][..., [0, 2]].view(-1, 1, 1, 2, 2)

        # Transform grid center locations into image u-coordinates
        cam_coords = torch.matmul(calib, self.grid.unsqueeze(-1)).squeeze(-1) # u, z

        # Apply perspective projection and normalize so that values range in [-1, 1]
        ucoords = cam_coords[..., 0] / cam_coords[..., 1] # only x direction, all positive (pixel coordinate)
        ucoords = ucoords / features.size(-1) * 2 - 1 # features.size(-1): width

        # Normalize z coordinates so that values range in [-1, 1]
        zcoords = (cam_coords[..., 1]-self.near) / (self.far-self.near) * 2 - 1

        # Resample 3D feature map
        # grid_coords: grid from world-bev(X-Z) to camera-bev(u-z), because bev_feature width is same as image width
        grid_coords = torch.stack([ucoords, zcoords], -1).clamp(-1.1, 1.1) # why -1.1, 1.1?? grid value should be [-1, 1] e.g. (-1, -1): top-left, (1, 1): bottom-right
        return F.grid_sample(features, grid_coords) # why sample in bev feature, not image feature?
        # [batch, channel=64, out_depth=22, width=image_width], [batch, 22(z_grid), 100(x_grid), 2] -> [batch, channel=64, 22(z_grid), 100(x_grid)]


def _make_grid(resolution, extents):
    # Create a grid of cooridinates in the birds-eye-view
    x1, z1, x2, z2 = extents # [-25, zmin, 25, zmax]
    zz, xx = torch.meshgrid(
        torch.arange(z1, z2, resolution), torch.arange(x1, x2, resolution))

    return torch.stack([xx, zz], dim=-1) # stack along last dim