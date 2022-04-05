import os

import torch
import numpy as np
from PIL import Image, ImageFile
from pyquaternion import Quaternion
from nuscenes import NuScenes

from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

import src.data.nuscenes.utils as nusc_utils
from src.data.utils import get_visible_mask, get_occlusion_mask, transform, get_distance_transform

from .rasterize import preprocess_map
from .vector_map import VectorizedLocalMap
from ..nuscenes.utils import CAMERA_NAMES, HDMAPNET_CLASSES, NUSCENES_CLASS_NAMES, get_sensor_transform, iterate_samples, STATIC_CLASSES


class HDMapNetDataset(Dataset):
    def __init__(self, version, dataroot, extents=[-25., 1., 25., 50.], resolution=0.25, 
                 image_size=(800, 450), scene_names=None):
        super(HDMapNetDataset, self).__init__()
        # xbound: longitudinal direction
        # ybound: lateral direction
        self.map_extents = extents
        self.map_resolution = resolution
        x1, z1, x2, z2 = self.map_extents
        patch_h = z2 - z1 # 49.0
        patch_w = x2 - x1 # 50.0
        canvas_h = int(patch_h / self.map_resolution) # 196
        canvas_w = int(patch_w / self.map_resolution) # 200
        self.patch_size = (patch_h, patch_w) # (49.0, 50.0)
        self.canvas_size = (canvas_h, canvas_w) # (196, 200)
        self.image_size = image_size
        self.nusc = NuScenes(version=version, dataroot=dataroot)
        self.vector_map = VectorizedLocalMap(dataroot, patch_size=self.patch_size, canvas_size=self.canvas_size)

        self.get_tokens(scene_names)

        # Allow PIL to load partially corrupted images
        # (otherwise training crashes at the most inconvenient possible times!)
        ImageFile.LOAD_TRUNCATED_IMAGES = True

    def get_tokens(self, scene_names=None):
        
        self.tokens = list()

        # Iterate over scenes
        for scene in self.nusc.scene:
            
            # Ignore scenes which don't belong to the current split
            if scene_names is not None and scene['name'] not in scene_names:
                continue
             
            # Iterate over samples
            for sample in iterate_samples(self.nusc, 
                                          scene['first_sample_token']):
                
                # Iterate over cameras
                for camera in CAMERA_NAMES: 
                    # ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_BACK']
                    self.tokens.append(sample['data'][camera])
        
        return self.tokens
    
    def __len__(self):
        return len(self.tokens)

    # def get_imgs(self, rec):
    #     imgs = []
    #     trans = []
    #     rots = []
    #     intrins = []
    #     for cam in CAMERA_NAMES:
    #         samp = self.nusc.get('sample_data', rec['data'][cam])
    #         imgname = os.path.join(self.nusc.dataroot, samp['filename'])
    #         img = Image.open(imgname)
    #         imgs.append(img)

    #         sens = self.nusc.get('calibrated_sensor', samp['calibrated_sensor_token'])
    #         trans.append(torch.Tensor(sens['translation']))
    #         rots.append(torch.Tensor(Quaternion(sens['rotation']).rotation_matrix))
    #         intrins.append(torch.Tensor(sens['camera_intrinsic']))
    #     return imgs, trans, rots, intrins

    def __getitem__(self, idx):
        token = self.tokens[idx]

        sample_data = self.nusc.get('sample_data', token) # camera token

        image = self.load_image(token)
        calib = self.load_calib(token)
        # map location in ['boston-seaport', 'singapore-onenorth', 
        # 'singapore-queenstown','singapore-hollandvillage']
        location = self.nusc.get( 'log', 
                                  self.nusc.get( 'scene', 
                                                 self.nusc.get( 'sample',
                                                                sample_data['sample_token'] )['scene_token'] )['log_token'] )['location']
        # rec: sample, rec['data']: sample_data
        ego_pose = self.nusc.get('ego_pose', sample_data['ego_pose_token']) # camera pose in map coord
        # Create a transform from birds-eye-view coordinates to map coordinates
        tfm = get_sensor_transform(self.nusc, sample_data)[[0, 1, 3]][:, [0, 2, 3]]
        vectors = self.vector_map.gen_vectorized_samples(tfm, self.map_extents, location, ego_pose['translation'], ego_pose['rotation'])
        semantic_masks, instance_masks, forward_masks, backward_masks = preprocess_map(vectors, self.map_extents, self.patch_size, self.canvas_size, self.max_channel, self.thickness, self.angle_class)
        
        visible_mask = np.expand_dims(np.zeros(semantic_masks.shape[1:3], dtype=np.uint8), axis=0)
        masks = np.concatenate([semantic_masks, visible_mask], axis=0)
        sensor = self.nusc.get('calibrated_sensor', 
                          sample_data['calibrated_sensor_token'])
        intrinsics = np.array(sensor['camera_intrinsic'])

        masks[-1] |= ~get_visible_mask(intrinsics, sample_data['width'],
                                       self.map_extents, self.map_resolution)

        
        sample = self.nusc.get('sample', sample_data['sample_token'])
        lidar_data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        lidar_pcl = nusc_utils.load_point_cloud(self.nusc, lidar_data)
        
        # Transform points into world coordinate system
        lidar_transform = nusc_utils.get_sensor_transform(self.nusc, lidar_data)
        lidar_pcl = transform(lidar_transform, lidar_pcl)
        
        cam_transform = nusc_utils.get_sensor_transform(self.nusc, sample_data)
        cam_points = transform(np.linalg.inv(cam_transform), lidar_pcl)

        masks[-1] |= get_occlusion_mask(cam_points, self.map_extents,
                                    self.map_resolution)

        labels, mask = masks[:-1].astype(np.bool), ~(masks[-1].astype(np.bool))
        
        return image, calib, torch.tensor(labels, dtype=torch.bool), torch.tensor(mask, dtype=torch.bool)

    
    def load_image(self, token):

        # Load image as a PIL image
        image = Image.open(self.nusc.get_sample_data_path(token))

        # Resize to input resolution
        image = image.resize(self.image_size)

        # Convert to a torch tensor
        return to_tensor(image)
    
    def load_calib(self, token):

        # Load camera intrinsics matrix
        sample_data = self.nusc.get('sample_data', token)
        sensor = self.nusc.get(
            'calibrated_sensor', sample_data['calibrated_sensor_token'])
        intrinsics = torch.tensor(sensor['camera_intrinsic'])
        # "camera_intrinsic": 
        # [ [1252.8131021185304,0.0,826.588114781398],  : intrinsics[0]
        #   [0.0,1252.8131021185304,469.9846626224581], : intrinsics[1]
        #   [0.0,0.0,1.0] ]

        # Scale calibration matrix to account for image downsampling
        intrinsics[0] *= self.image_size[0] / sample_data['width']
        intrinsics[1] *= self.image_size[1] / sample_data['height']
        return intrinsics

class HDMapNetSemanticDataset(HDMapNetDataset):
    def __init__(self, version, dataroot, extents=[-25., 1., 25., 50.], resolution=0.25, 
                 image_size=(800, 450), scene_names=None, max_channel=3, thickness=3, angle_class=36):
        super(HDMapNetSemanticDataset, self).__init__(version, dataroot, extents, resolution, image_size, scene_names)
        self.max_channel = max_channel
        self.thickness = thickness
        self.angle_class = angle_class

    def __getitem__(self, idx):
        token = self.tokens[idx]

        sample_data = self.nusc.get('sample_data', token) # camera token

        image = self.load_image(token)
        calib = self.load_calib(token)
        # map location in ['boston-seaport', 'singapore-onenorth', 
        # 'singapore-queenstown','singapore-hollandvillage']
        location = self.nusc.get( 'log', 
                                  self.nusc.get( 'scene', 
                                                 self.nusc.get( 'sample',
                                                                sample_data['sample_token'] )['scene_token'] )['log_token'] )['location']
        # rec: sample, rec['data']: sample_data
        ego_pose = self.nusc.get('ego_pose', sample_data['ego_pose_token']) # camera pose in map coord
        # Create a transform from birds-eye-view coordinates to map coordinates
        tfm = get_sensor_transform(self.nusc, sample_data)[[0, 1, 3]][:, [0, 2, 3]]
        vectors = self.vector_map.gen_vectorized_samples(tfm, self.map_extents, location, ego_pose['translation'], ego_pose['rotation'])
        semantic_masks, instance_masks, forward_masks, backward_masks, distance_masks = preprocess_map(vectors, self.map_extents, self.patch_size, self.canvas_size, self.max_channel, self.thickness, self.angle_class)

        # semantic_masks: (3, 196, 200) np
        # Create distance transform
        distance_masks = get_distance_transform(distance_masks, 10.0)
        
        visible_mask = np.expand_dims(np.zeros(semantic_masks.shape[1:3], dtype=np.uint8), axis=0)
        masks = np.concatenate([semantic_masks, visible_mask], axis=0)
        sensor = self.nusc.get('calibrated_sensor', 
                          sample_data['calibrated_sensor_token'])
        intrinsics = np.array(sensor['camera_intrinsic'])

        masks[-1] |= ~get_visible_mask(intrinsics, sample_data['width'],
                                       self.map_extents, self.map_resolution)

        
        sample = self.nusc.get('sample', sample_data['sample_token'])
        lidar_data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        lidar_pcl = nusc_utils.load_point_cloud(self.nusc, lidar_data)
        
        # Transform points into world coordinate system
        lidar_transform = nusc_utils.get_sensor_transform(self.nusc, lidar_data)
        lidar_pcl = transform(lidar_transform, lidar_pcl)
        
        cam_transform = nusc_utils.get_sensor_transform(self.nusc, sample_data)
        cam_points = transform(np.linalg.inv(cam_transform), lidar_pcl)

        masks[-1] |= get_occlusion_mask(cam_points, self.map_extents,
                                    self.map_resolution)

        labels, mask = masks[:-1].astype(np.bool), ~(masks[-1].astype(np.bool))
        
        return image, calib, torch.tensor(labels, dtype=torch.bool), torch.tensor(mask, dtype=torch.bool), torch.tensor(distance_masks, dtype=torch.float32)

