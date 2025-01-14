import math
import torch
import torch.nn as nn

from .transformer import DenseTransformer

class TransformerPyramid(nn.Module):

    def __init__(self, in_channels, channels, resolution, extents, ymin, ymax, 
                 focal_length): 
                 # in_channels=256, channels=64, resolution=0.5, 
                 # extents=[-25., 1., 25., 50.], ymin=-2, ymin=4, 
                 # focal_length=630.
        
        super().__init__()
        self.transformers = nn.ModuleList()
        for i in range(5): # 0, 1, 2, 3, 4
            
            # Scaled focal length for each transformer
            focal = focal_length / pow(2, i + 3) # 8, 16, 32, 64, 128 : scales in feature pyramid

            # Compute grid bounds for each transformer
            zmax = min(math.floor(focal * 2) * resolution, extents[3]) # math.floor(focal * 2): focal length at i-1 level
            zmin = math.floor(focal) * resolution if i < 4 else extents[1]
            subset_extents = [extents[0], zmin, extents[2], zmax]
            # Build transformers
            tfm = DenseTransformer(in_channels, channels, resolution, 
                                   subset_extents, ymin, ymax, focal)
            self.transformers.append(tfm)
    

    def forward(self, feature_maps, calib):
        
        bev_feats = list()
        for i, fmap in enumerate(feature_maps):
            
            # Scale calibration matrix to account for downsampling
            scale = 8 * 2 ** i
            calib_downsamp = calib.clone()
            calib_downsamp[:, :2] = calib[:, :2] / scale

            # Apply orthographic transformation to each feature map separately
            bev_feats.append(self.transformers[i](fmap, calib_downsamp))
        
        # Combine birds-eye-view feature maps along the depth axis
        return torch.cat(bev_feats[::-1], dim=-2)
