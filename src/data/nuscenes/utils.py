import os
import numpy as np
from shapely import geometry, affinity
from pyquaternion import Quaternion

from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.eval.detection.constants import DETECTION_NAMES
from nuscenes.utils.data_classes import LidarPointCloud

from ..utils import transform_polygon, render_polygon, transform, render_line, get_ped_crossing_line, render_polygon_to_vectors

CAMERA_NAMES = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 
                'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_BACK']

NUSCENES_CLASS_NAMES = [
    'drivable_area', 'ped_crossing', 'walkway', 
    # 'stop_line', # added
    'carpark_area', 
    # 'road_divider', 'lane_divider', # added
    'car', 'truck', 
    'bus', 'trailer', 'construction_vehicle', 'pedestrian', 'motorcycle', 
    'bicycle', 'traffic_cone', 'barrier'
]

STATIC_CLASSES = [
    'drivable_area', 'ped_crossing', 'walkway', 
    # 'stop_line', # added
    'carpark_area',
    'road_divider', 'lane_divider' # added
    ]

HDMAPNET_CLASSES = ['contour', 'ped_crossing', 'line_classes']

LOCATIONS = ['boston-seaport', 'singapore-onenorth', 'singapore-queenstown',
             'singapore-hollandvillage']


def iterate_samples(nuscenes, start_token):
    sample_token = start_token
    while sample_token != '':
        sample = nuscenes.get('sample', sample_token)
        yield sample
        sample_token = sample['next']
    

def get_map_masks(nuscenes, map_data, sample_data, extents, resolution):
    # sample_data: camera
    # Render each layer sequentially
    # layer: ['road_segment', 'lane', 'ped_crossing', 'road_divider', 'lane_divider']
    layers = [get_layer_mask(nuscenes, layer, polys, sample_data, extents, 
              resolution) for layer, polys in map_data.items()] # polys and lines
    
    # layers = []
    # for layer, polys in map_data.items():
    #     layer_mask = get_layer_mask(nuscenes, layer, polys, sample_data, extents, 
    #           resolution)
    #     layers.append(layer_mask)

    return np.stack(layers, axis=0)


def get_layer_mask(nuscenes, layer, polygons, sample_data, extents, resolution):
    # polygons: shapely Polygon or LineString in STRtree

    # Get the 2D affine transform from bev coords to map coords 4x4
    tfm = get_sensor_transform(nuscenes, sample_data)[[0, 1, 3]][:, [0, 2, 3]] # 3x3
    inv_tfm = np.linalg.inv(tfm)

    # Create a patch representing the birds-eye-view region in map coordinates
    map_patch = geometry.box(*extents)
    map_patch = transform_polygon(map_patch, tfm) # map patch in global coordinate

    # Initialise the map mask
    x1, z1, x2, z2 = extents
    mask = np.zeros((int((z2 - z1) / resolution), int((x2 - x1) / resolution)),
                    dtype=np.uint8)

    # Find all polygons which intersect with the area of interest
    for polygon in polygons.query(map_patch): # SRTree.query

        polygon = polygon.intersection(map_patch)
        
        # Transform into map coordinates, in [-25, 1, 25, 50]
        polygon = transform_polygon(polygon, inv_tfm)

        if layer in HDMAPNET_CLASSES:
            # Sample vector points and render the line to the mask
            render_vectorized_layer(layer, mask, polygon, extents, resolution)
        else:

            # Render the polygon to the mask
            render_shapely_polygon(mask, polygon, extents, resolution)
    
    return mask.astype(np.bool)




def get_object_masks(nuscenes, sample_data, extents, resolution):

    # Initialize object masks
    nclass = len(DETECTION_NAMES) + 1
    grid_width = int((extents[2] - extents[0]) / resolution)
    grid_height = int((extents[3] - extents[1]) / resolution)
    masks = np.zeros((nclass, grid_height, grid_width), dtype=np.uint8)

    # Get the 2D affine transform from bev coords to map coords
    tfm = get_sensor_transform(nuscenes, sample_data)[[0, 1, 3]][:, [0, 2, 3]] # 3x3
    inv_tfm = np.linalg.inv(tfm)

    for box in nuscenes.get_boxes(sample_data['token']):

        # Get the index of the class
        det_name = category_to_detection_name(box.name)
        if det_name not in DETECTION_NAMES:
            class_id = -1
        else:
            class_id = DETECTION_NAMES.index(det_name)
        
        # Get bounding box coordinates in the grid coordinate frame
        bbox = box.bottom_corners()[:2]
        local_bbox = np.dot(inv_tfm[:2, :2], bbox).T + inv_tfm[:2, 2]

        # Render the rotated bounding box to the mask
        render_polygon(masks[class_id], local_bbox, extents, resolution)
    
    return masks.astype(np.bool)


def get_sensor_transform(nuscenes, sample_data):

    # Load sensor transform data
    sensor = nuscenes.get(
        'calibrated_sensor', sample_data['calibrated_sensor_token'])
    sensor_tfm = make_transform_matrix(sensor) # 4x4, P

    # Load ego pose data
    pose = nuscenes.get('ego_pose', sample_data['ego_pose_token'])
    pose_tfm = make_transform_matrix(pose) # 4x4

    return np.dot(pose_tfm, sensor_tfm)


def load_point_cloud(nuscenes, sample_data):

    # Load point cloud
    lidar_path = os.path.join(nuscenes.dataroot, sample_data['filename'])
    pcl = LidarPointCloud.from_file(lidar_path)
    return pcl.points[:3, :].T


def make_transform_matrix(record):
    """
    Create a 4x4 transform matrix from a calibrated_sensor or ego_pose record
    """
    transform = np.eye(4)
    transform[:3, :3] = Quaternion(record['rotation']).rotation_matrix
    transform[:3, 3] = np.array(record['translation'])
    return transform


def render_shapely_polygon(mask, polygon, extents, resolution):

    if polygon.geom_type == 'Polygon':

        # Render exteriors
        render_polygon(mask, polygon.exterior.coords, extents, resolution, 1)

        # Render interiors
        for hole in polygon.interiors:
            render_polygon(mask, hole.coords, extents, resolution, 0)
    
    # Handle LineString object
    elif polygon.geom_type == 'LineString':
        render_line(mask, polygon.coords, extents, resolution, 1)

    # Handle MultiLineString object
    # elif polygon.geom_type == 'MultiLineString':
    #     # raise NotImplementedError
    #     for line in polygon:
    #         render_shapely_polygon(mask, line, extents, resolution)
    # Handle the case of compound shapes (+ MultiLineString)
    else:
        for poly in polygon:
            render_shapely_polygon(mask, poly, extents, resolution)


def render_vectorized_layer(layer, mask, polygon, extents, resolution):
    if polygon.is_empty:
        return

    if layer in ['road_divider', 'lane_divider']:
        render_shapely_polygon(mask, polygon, extents, resolution)
    elif layer in ['ped_crossing']:
        polygon = get_ped_crossing_line(polygon)
        render_shapely_polygon(mask, polygon, extents, resolution)
    elif layer == 'contour':
        # polygon to vectors
        render_polygon_to_vectors(mask, polygon, extents, resolution)
    else:
        raise NotImplementedError



