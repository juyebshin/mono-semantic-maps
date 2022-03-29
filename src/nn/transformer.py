import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .resampler import Resampler

class DenseTransformer(nn.Module):

    def __init__(self, in_channels, channels, resolution, grid_extents, 
                 ymin, ymax, focal_length, groups=1):
                 # in_channels=256, channels=64, resolution=0.5, 
                 # ymin=-2, ymin=4, focal_length: [78.75, 39.375, 19.6875, ],
                 # grid_extents: [-25, 39, 25, 50], [-25, 19.5, 25, 39], [-25, 9.5, 25, 19.5], [-25, 4.5, 25, 9.5], [-25, 1, 25, 4.5]
        super().__init__()

        # Initial convolution to reduce feature dimensions
        self.conv = nn.Conv2d(in_channels, channels, 1)
        self.bn = nn.GroupNorm(16, channels)

        # Resampler transforms perspective features to BEV
        self.resampler = Resampler(resolution, grid_extents)

        # Compute input height based on region of image covered by grid
        self.zmin, zmax = grid_extents[1], grid_extents[3]
        self.in_height = math.ceil(focal_length * (ymax - ymin) / self.zmin) # 13 for all feature maps
        self.ymid = (ymin + ymax) / 2

        # Compute number of output cells required
        self.out_depth = math.ceil((zmax - self.zmin) / resolution)

        # Dense layer which maps UV features to UZ
        self.fc = nn.Conv1d(
            channels * self.in_height, channels * self.out_depth, 1, groups=groups
        ) # 
        self.out_channels = channels
    

    def forward(self, features, calib, *args):

        # Crop feature maps to a fixed input height
        features = torch.stack([self._crop_feature_map(fmap, cal) 
                                for fmap, cal in zip(features, calib)])
        
        # Reduce feature dimension to minimize memory usage
        features = F.relu(self.bn(self.conv(features)))

        # Flatten height and channel dimensions
        B, C, _, W = features.shape # [batch, channel=64, in_height, width]
        flat_feats = features.flatten(1, 2) # flaten in channel and height: [batch, channel*in_height, width]
        bev_feats = self.fc(flat_feats).view(B, C, -1, W) # [batch, (channel=64)*out_depth, width] -> [batch, channel=64, out_depth, width]

        # Resample to orthographic grid
        return self.resampler(bev_feats, calib)


    def _crop_feature_map(self, fmap, calib):
        
        # Compute upper and lower bounds of visible region
        focal_length, img_offset = calib[1, 1:]
        vmid = self.ymid * focal_length / self.zmin + img_offset
        vmin = math.floor(vmid - self.in_height / 2)
        vmax = math.floor(vmid + self.in_height / 2)

        # Pad or crop input tensor to match dimensions
        return F.pad(fmap, [0, 0, -vmin, vmax - fmap.shape[-2]])