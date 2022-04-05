import cv2
import numpy as np

from shapely import affinity
from shapely.geometry import LineString, box


def get_patch_coord(patch_box, patch_angle=0.0):
    patch_x, patch_y, patch_h, patch_w = patch_box # (25.5, 0, 50, 49)

    x_min = patch_x - patch_w / 2.0 # 1.0
    y_min = patch_y - patch_h / 2.0 # 25.0
    x_max = patch_x + patch_w / 2.0 # 50.0
    y_max = patch_y + patch_h / 2.0 # 25.0

    patch = box(x_min, y_min, x_max, y_max)
    patch = affinity.rotate(patch, patch_angle, origin=(patch_x, patch_y), use_radians=False)

    return patch


def get_discrete_degree(vec, angle_class=36):
    deg = np.mod(np.degrees(np.arctan2(vec[1], vec[0])), 360)
    deg = (int(deg / (360 / angle_class) + 0.5) % angle_class) + 1
    return deg


def mask_for_lines(lines, mask, thickness, idx, type='index', angle_class=36):
    coords = np.asarray(list(lines.coords), np.int32)
    coords = coords.reshape((-1, 2))
    if len(coords) < 2:
        return mask, idx
    if type == 'backward':
        coords = np.flip(coords, 0)

    if type == 'index':
        cv2.polylines(mask, [coords], False, color=idx, thickness=thickness)
        idx += 1
    else:
        for i in range(len(coords) - 1):
            cv2.polylines(mask, [coords[i:]], False, color=get_discrete_degree(coords[i + 1] - coords[i], angle_class=angle_class), thickness=thickness)
    return mask, idx


def line_geom_to_mask(layer_geom, confidence_levels, extents, canvas_size, thickness, idx, type='index', angle_class=36):
    # patch_x, patch_y, patch_h, patch_w = local_box # (25.5, 0, 50, 49)
    x1, z1, x2, z2 = extents
    patch_h = z2 - z1 # 49.0
    patch_w = x2 - x1 # 50.0

    # patch = get_patch_coord(local_box) # 1.0, 25.0, 50.0, 25.0
    patch = box(*extents)

    canvas_h = canvas_size[0] # 196
    canvas_w = canvas_size[1] # 200
    scale_height = canvas_h / patch_h # 196 / 49.0 = 4
    scale_width = canvas_w / patch_w # 200 / 50.0 = 4

    trans_x = -x1 # 25.0
    trans_z = -z1 # -1.0

    map_mask = np.zeros(canvas_size, np.uint8) # shape (196, 200)

    for line in layer_geom: # layer_geom: LineString or tuple of LineString
        if isinstance(line, tuple):
            line, confidence = line
        else:
            confidence = None
        new_line = line.intersection(patch)
        if not new_line.is_empty:
            # new_line in camera xz coord
            new_line = affinity.affine_transform(new_line, [1.0, 0.0, 0.0, 1.0, trans_x, trans_z]) # # xy to xz and (-25, 1) will be origin (0, 0)
            new_line = affinity.scale(new_line, xfact=scale_width, yfact=scale_height, origin=(0, 0))
            confidence_levels.append(confidence)
            if new_line.geom_type == 'MultiLineString':
                for new_single_line in new_line:
                    map_mask, idx = mask_for_lines(new_single_line, map_mask, thickness, idx, type, angle_class)
            else:
                map_mask, idx = mask_for_lines(new_line, map_mask, thickness, idx, type, angle_class)
    return map_mask, idx


def overlap_filter(mask, filter_mask):
    C, _, _ = mask.shape
    for c in range(C-1, -1, -1):
        filter = np.repeat((filter_mask[c] != 0)[None, :], c, axis=0)
        mask[:c][filter] = 0

    return mask


def preprocess_map(vectors, extents, patch_size, canvas_size, max_channel, thickness, angle_class):
    # vectors: patch center as origin
    # patch_size:  (49.0, 50.0)
    # canvas_size: (196, 200)
    confidence_levels = [-1]
    vector_num_list = {}
    for i in range(max_channel):
        vector_num_list[i] = []

    for vector in vectors:
        if vector['pts_num'] >= 2:
            vector_num_list[vector['type']].append(LineString(vector['pts'][:vector['pts_num']]))
    # vector['type']: 
    # 'road_divider': 0,
    # 'lane_divider': 0, # r
    # 'ped_crossing': 1, # b
    # 'contours': 2, # g
    # 'others': -1,

    x1, z1, x2, z2 = extents
    center_x = (x1 + x2) / 2.0 # 0
    center_z = (z1 + z2) / 2.0 # 25.5

    local_box = (center_z, center_x, patch_size[1], patch_size[0]) # (25.5, 0, 50.0, 49.0)

    idx = 1
    filter_masks = []
    instance_masks = []
    forward_masks = [] # vector direction forward
    backward_masks = [] # vector direction forward
    distance_masks = []
    for i in range(max_channel): # max_channel=3, 0: line_classes, 1: ped_crossing, 2: boundary_classes
        # idx: instance index
        map_mask, idx = line_geom_to_mask(vector_num_list[i], confidence_levels, extents, canvas_size, thickness, idx)
        instance_masks.append(map_mask)
        filter_mask, _ = line_geom_to_mask(vector_num_list[i], confidence_levels, extents, canvas_size, thickness + 4, 1)
        filter_masks.append(filter_mask)
        forward_mask, _ = line_geom_to_mask(vector_num_list[i], confidence_levels, extents, canvas_size, thickness, 1, type='forward', angle_class=angle_class)
        forward_masks.append(forward_mask)
        backward_mask, _ = line_geom_to_mask(vector_num_list[i], confidence_levels, extents, canvas_size, thickness, 1, type='backward', angle_class=angle_class)
        backward_masks.append(backward_mask)
        distance_mask, _ = line_geom_to_mask(vector_num_list[i], confidence_levels, extents, canvas_size, 1, 1)
        distance_masks.append(distance_mask)

    filter_masks = np.stack(filter_masks) # (3, 200, 400)
    instance_masks = np.stack(instance_masks) # (3, 200, 400)
    forward_masks = np.stack(forward_masks)
    backward_masks = np.stack(backward_masks)
    distance_masks = np.stack(distance_masks)

    instance_masks = overlap_filter(instance_masks, filter_masks)
    forward_masks = overlap_filter(forward_masks, filter_masks).sum(0).astype('int32')
    backward_masks = overlap_filter(backward_masks, filter_masks).sum(0).astype('int32')

    semantic_masks = instance_masks != 0
    distance_masks = distance_masks != 0

    return semantic_masks, instance_masks, forward_masks, backward_masks, distance_masks


def rasterize_map(vectors, patch_size, canvas_size, max_channel, thickness):
    confidence_levels = [-1]
    vector_num_list = {}
    for i in range(max_channel + 1):
        vector_num_list[i] = []

    for vector in vectors:
        if vector['pts_num'] >= 2:
            vector_num_list[vector['type']].append((LineString(vector['pts'][:vector['pts_num']]), vector.get('confidence_level', 1)))

    local_box = (0.0, 0.0, patch_size[0], patch_size[1])

    idx = 1
    masks = []
    for i in range(max_channel):
        map_mask, idx = line_geom_to_mask(vector_num_list[i], confidence_levels, local_box, canvas_size, thickness, idx)
        masks.append(map_mask)

    return np.stack(masks), confidence_levels
