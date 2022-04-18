import cv2
import numpy as np
import torch
from shapely import affinity, ops
from shapely.geometry import LineString, MultiPolygon, MultiLineString, box

def decode_binary_labels(labels, nclass):
    bits = torch.pow(2, torch.arange(nclass))
    return (labels & bits.view(-1, 1, 1)) > 0 # bits: (nclass, 1, 1)


def encode_binary_labels(masks): # masks: binary
    bits = np.power(2, np.arange(len(masks), dtype=np.int32)) # len(mask) = 15
    return (masks.astype(np.int32) * bits.reshape(-1, 1, 1)).sum(0) # (15, 196, 200) * (15, 1, 1)


def get_distance_transform(masks, threshold=None):
    # masks: (3, 196, 200) np bool
    labels = (~masks).astype('uint8')
    distances = np.zeros(masks.shape, dtype=np.float32)
    for i, label in enumerate(labels):
        distances[i] = cv2.distanceTransform(label, cv2.DIST_L2, maskSize=5)
        # truncate to [0.0, 10.0] and invert values
        if threshold is not None:
            distances[i] = float(threshold) - distances[i]
            distances[i][distances[i] < 0.0] = 0.0
        cv2.normalize(distances[i], distances[i], 0, 1.0, cv2.NORM_MINMAX)
    return distances

def transform(matrix, vectors):
    vectors = np.dot(matrix[:-1, :-1], vectors.T)
    vectors = vectors.T + matrix[:-1, -1]
    return vectors


def transform_polygon(polygon, affine):
    """
    Transform a 2D polygon
    affine: 3x3 array
    """
    a, b, tx, c, d, ty = affine.flatten()[:6]
    return affinity.affine_transform(polygon, [a, b, c, d, tx, ty])


def render_polygon(mask, polygon, extents, resolution, value=1):
    # polygon: exterior.coords
    if len(polygon) == 0:
        return
    polygon = (polygon - np.array(extents[:2])) / resolution
    polygon = np.ascontiguousarray(polygon).round().astype(np.int32)
    cv2.fillConvexPoly(mask, polygon, value)

def render_line(mask, line, extents, resolution, value=1):
    # mask: (196, 200)
    # line: LineString in camera xz coord
    # extents: [-25.0, 1.0, 25.0, 50.0]
    if len(line) == 0:
        return

    # pts, pts_num = sample_pts_from_line(line)
    # vector_num_list = LineString(pts[:pts_num])
    line = (line - np.array(extents[:2])) / resolution
    line = np.ascontiguousarray(line).round().astype(np.int32)
    if len(line) < 2:
        return
    cv2.polylines(mask, [line], False, value, thickness=3)

# def line_geom_to_mask(mask, line, confidence_levels, local_box, canvas_size, thickness, value):
#     patch_x, patch_y, patch_h, patch_w = local_box # [-25., 1., 25., 50.]
#     patch_h = patch_h - patch_x
#     patch_w = patch_w - patch_x


def sample_pts_from_line(line):
    sample_dist = 1
    distances = np.arange(0, line.length, sample_dist)
    sampled_points = np.array([list(line.interpolate(distance).coords) for distance in distances]).reshape(-1, 2) # Nx2

    num_valid = len(sampled_points)

    return sampled_points, num_valid


def get_ped_crossing_line(polygon):
    def add_line(poly_xy, idx):
        points = [(p0, p1) for p0, p1 in zip(poly_xy[0, idx:idx + 2], poly_xy[1, idx:idx + 2])]
        line = LineString(points)
        # line = line.intersection(patch)
        return line

    poly_xy = np.array(polygon.exterior.xy)
    dist = np.square(poly_xy[:, 1:] - poly_xy[:, :-1]).sum(0)
    x1, x2 = np.argsort(dist)[-2:]

    line1 = add_line(poly_xy, x1)
    line2 = add_line(poly_xy, x2)

    return MultiLineString([line1, line2])

def render_polygon_to_vectors(mask, polygon, extents, resolution, value=1):
    union_segments = ops.unary_union(polygon)
    # roads = polygon[0][1] # road_segment
    # lanes = polygon[1][1] # lane
    # union_roads = ops.unary_union(roads)
    # union_lanes = ops.unary_union(lanes)
    # union_segments = ops.unary_union([union_roads, union_lanes])
    exteriors = []
    interiors = []
    x1, z1, x2, z2 = extents
    local_patch = box(x1 + 0.2, z1 + 0.2, x2 - 0.2, z2 - 0.2)
    if union_segments.geom_type != 'MultiPolygon':
        union_segments = MultiPolygon([union_segments])
    for poly in union_segments:
        exteriors.append(poly.exterior) # polygon
        for inter in poly.interiors:
            interiors.append(inter)

    results = []
    for ext in exteriors:
        if ext.is_ccw:
            ext.coords = list(ext.coords)[::-1]
        lines = ext.intersection(local_patch)
        if isinstance(lines, MultiLineString):
            lines = ops.linemerge(lines)
        results.append(lines)

    for inter in interiors:
        if not inter.is_ccw:
            inter.coords = list(inter.coords)[::-1]
        lines = inter.intersection(local_patch)
        if isinstance(lines, MultiLineString):
            lines = ops.linemerge(lines)
        results.append(lines)
    
    line_vectors = []
    for line in results:
        if not line.is_empty:
            if line.geom_type == 'MultiLineString':
                for l in line:
                    line_vectors.append(sample_pts_from_line(l))
            elif line.geom_type == 'LineString': # road_divider, lane_divider
                line_vectors.append(sample_pts_from_line(line))
            else:
                raise NotImplementedError

    for line in line_vectors:
        pts, pts_num = line
        if pts_num >= 2:
            line_vector = LineString(pts[:pts_num])
            line = np.asarray(list(line_vector.coords))
            line = (line - np.array(extents[:2])) / resolution
            line_vector = np.ascontiguousarray(line).round().astype(np.int32)

            cv2.polylines(mask, [line_vector], False, value, thickness=3)
    


def get_visible_mask(instrinsics, image_width, extents, resolution):

    # Get calibration parameters
    fu, cu = instrinsics[0, 0], instrinsics[0, 2]

    # Construct a grid of image coordinates
    x1, z1, x2, z2 = extents
    x, z = np.arange(x1, x2, resolution), np.arange(z1, z2, resolution)
    ucoords = x / z[:, None] * fu + cu

    # Return all points which lie within the camera bounds
    return (ucoords >= 0) & (ucoords < image_width)


def get_occlusion_mask(points, extents, resolution):

    x1, z1, x2, z2 = extents

    # A 'ray' is defined by the ratio between x and z coordinates
    ray_width = resolution / z2 # 0.25 / 50.0 = 0.005
    ray_offset = x1 / ray_width # -25 / 0.005 = -5000
    max_rays = int((x2 - x1) / ray_width) # 50.0 / 0.005 = 10000

    # Group LiDAR points into bins
    rayid = np.round(points[:, 0] / points[:, 2] / ray_width - ray_offset)
    depth = points[:, 2]

    # Ignore rays which do not correspond to any grid cells in the BEV
    valid = (rayid > 0) & (rayid < max_rays) & (depth > 0)
    rayid = rayid[valid]
    depth = depth[valid]

    # Find the LiDAR point with maximum depth within each bin
    max_depth = np.zeros((max_rays,))
    np.maximum.at(max_depth, rayid.astype(np.int32), depth)

    # For each bev grid point, sample the max depth along the corresponding ray
    x = np.arange(x1, x2, resolution)
    z = np.arange(z1, z2, resolution)[:, None]
    grid_rayid = np.round(x / z / ray_width - ray_offset).astype(np.int32)
    grid_max_depth = max_depth[grid_rayid]

    # A grid position is considered occluded if the there are no LiDAR points
    # passing through it
    occluded = grid_max_depth < z
    return occluded





    




    






