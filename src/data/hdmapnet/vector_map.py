import numpy as np
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from shapely import affinity, ops, geometry
from shapely.geometry import LineString, box, MultiPolygon, MultiLineString

from ..nuscenes.utils import get_sensor_transform, transform_polygon


class VectorizedLocalMap(object):
    def __init__(self,
                 dataroot,
                 patch_size, # (49.0, 50.0)
                 canvas_size, # (196, 200)
                 line_classes=['road_divider', 'lane_divider'],
                 ped_crossing_classes=['ped_crossing'],
                 contour_classes=['road_segment', 'lane'],
                 sample_dist=1,
                 num_samples=250,
                 padding=False,
                 normalize=False,
                 fixed_num=-1,
                 class2label={
                     'road_divider': 0,
                     'lane_divider': 0, # r
                     'ped_crossing': 1, # b
                     'contours': 2, # g
                     'others': -1,
                 }):
        '''
        Args:
            fixed_num = -1 : no fixed num
        '''
        super().__init__()
        self.data_root = dataroot
        self.MAPS = ['boston-seaport', 'singapore-hollandvillage',
                     'singapore-onenorth', 'singapore-queenstown']
        self.line_classes = line_classes
        self.ped_crossing_classes = ped_crossing_classes
        self.polygon_classes = contour_classes
        self.class2label = class2label
        self.nusc_maps = {}
        self.map_explorer = {}
        for loc in self.MAPS: # ['boston-seaport', 'singapore-hollandvillage', 'singapore-onenorth', 'singapore-queenstown']
            self.nusc_maps[loc] = NuScenesMap(dataroot=self.data_root, map_name=loc)
            self.map_explorer[loc] = NuScenesMapExplorer(self.nusc_maps[loc])

        self.patch_size = patch_size
        self.canvas_size = canvas_size
        self.sample_dist = sample_dist
        self.num_samples = num_samples
        self.padding = padding
        self.normalize = normalize
        self.fixed_num = fixed_num

    def gen_vectorized_samples(self, tfm, extents, location, ego2global_translation, ego2global_rotation):
        # location: "singapore-onenorth", ...
        x1, z1, x2, z2 = extents
        center_x = (x1 + x2) / 2.0 # 0
        center_z = (z1 + z2) / 2.0 # 25.5
        map_pose = ego2global_translation[:2] # x y
        rotation = Quaternion(ego2global_rotation)

        # Get the 2D affine transform from bev coords to map coords 4x4
        inv_tfm = np.linalg.inv(tfm)

        # Create a patch representing the birds-eye-view region in map coordinates
        map_patch = geometry.box(*extents)
        map_patch = transform_polygon(map_patch, tfm) # map patch in global coordinate
        # rotation = ego2global_rotation

        # Original
        # patch_box = [x_center, y_center, height, width] -> ego_x + 1.0 + 24.5, ego_y, 49.0, 50.0
        # patch_box = (map_pose[0] + self.patch_size[0]/2., map_pose[1], self.patch_size[0], self.patch_size[1])

        # for PON
        # patch_box = [x_center, y_center, height, width] -> ego_x + 25.5, ego_y + 0, 50.0, 50.0*2
        #  + center_z,  + center_x
        patch_box = (map_pose[0], map_pose[1], self.patch_size[1], z2*2)
        # yaw
        patch_angle = quaternion_yaw(rotation) / np.pi * 180

        # LineString
        line_geom = self.get_map_geom(map_patch, inv_tfm, self.line_classes, location)
        # line_geom = list of [layer_name, Shapely geom]
        line_vector_dict = self.line_geoms_to_vectors(line_geom)
        # line_vector_dict = { layer_name: list of [array of sampled points, num of points] }

        ped_geom = self.get_map_geom(map_patch, inv_tfm, self.ped_crossing_classes, location)
        # ped_vector_list = self.ped_geoms_to_vectors(ped_geom)
        ped_vector_list = self.line_geoms_to_vectors(ped_geom)['ped_crossing']

        polygon_geom = self.get_map_geom(map_patch, inv_tfm, self.polygon_classes, location)
        # polygon_geom = list of [layer_name, Shapely geom]
        poly_bound_list = self.poly_geoms_to_vectors(polygon_geom, extents)
        # poly_bound_list = list of [sampled_points, num_valid]

        vectors = [] # pts, pts_num, type
        for line_type, vects in line_vector_dict.items():
            for line, length in vects:
                vectors.append((line.astype(float), length, self.class2label.get(line_type, -1)))

        for ped_line, length in ped_vector_list:
            vectors.append((ped_line.astype(float), length, self.class2label.get('ped_crossing', -1)))

        for contour, length in poly_bound_list:
            vectors.append((contour.astype(float), length, self.class2label.get('contours', -1))) # default -1

        # filter out -1
        filtered_vectors = []
        for pts, pts_num, type in vectors:
            if type != -1:
                filtered_vectors.append({
                    'pts': pts,
                    'pts_num': pts_num,
                    'type': type
                })

        return filtered_vectors

    ## original HDMapNet
    # def get_map_geom(self, patch_box, patch_angle, layer_names, location):
    #     # patch_box: ego_x, ego_y, 49.0, 50.0
    #     map_geom = []
    #     for layer_name in layer_names:
    #         if layer_name in self.line_classes: # ['road_divider', 'lane_divider']
    #             geoms = self.map_explorer[location]._get_layer_line(patch_box, patch_angle, layer_name)
    #             map_geom.append((layer_name, geoms))
    #         elif layer_name in self.polygon_classes: # ['road_segment', 'lane']
    #             geoms = self.map_explorer[location]._get_layer_polygon(patch_box, patch_angle, layer_name)
    #             map_geom.append((layer_name, geoms))
    #         elif layer_name in self.ped_crossing_classes: # ['ped_crossing']
    #             geoms = self.get_ped_crossing_line(patch_box, patch_angle, location)
    #             # geoms = self.map_explorer[location]._get_layer_polygon(patch_box, patch_angle, layer_name)
    #             map_geom.append((layer_name, geoms))
    #     return map_geom
    
    ## for PON
    def get_map_geom(self, map_patch, inv_tfm, layer_names, location):
        # patch_box: ego_x, ego_y, 49.0, 50.0
        map_geom = []
        for layer_name in layer_names:
            if layer_name in self.line_classes: # ['road_divider', 'lane_divider']
                # geoms = self.map_explorer[location]._get_layer_line(patch_box, patch_angle, layer_name)
                geoms = []
                records = getattr(self.nusc_maps[location], layer_name)
                for record in records:
                    line = self.nusc_maps[location].extract_line(record['line_token'])
                    if line.is_empty:  # Skip lines without nodes.
                        continue

                    new_line = line.intersection(map_patch)
                    if not new_line.is_empty:
                        # new_line = affinity.rotate(new_line, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
                        # new_line = affinity.affine_transform(new_line,
                        #                                     [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                        new_line = transform_polygon(new_line, inv_tfm)
                        geoms.append(new_line)
                map_geom.append((layer_name, geoms))
            elif layer_name in self.polygon_classes: # ['road_segment', 'lane']
                # geoms = self.map_explorer[location]._get_layer_polygon(patch_box, patch_angle, layer_name)
                geoms = []
                records = getattr(self.nusc_maps[location], layer_name)
                if layer_name == 'drivable_area':
                    for record in records:
                        polygons = [self.nusc_maps[location].extract_polygon(polygon_token) for polygon_token in record['polygon_tokens']]

                        for polygon in polygons:
                            new_polygon = polygon.intersection(map_patch)
                            if not new_polygon.is_empty:
                                # new_polygon = affinity.rotate(new_polygon, -patch_angle, # origin is (patch_x, patch_y)
                                #                             origin=(patch_x, patch_y), use_radians=False)
                                # new_polygon = affinity.affine_transform(new_polygon,
                                #                                         [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                                if new_polygon.geom_type is 'Polygon':
                                    new_polygon = MultiPolygon([new_polygon])
                                new_polygon = transform_polygon(new_polygon, inv_tfm)
                                geoms.append(new_polygon)

                else:
                    for record in records:
                        polygon = self.nusc_maps[location].extract_polygon(record['polygon_token'])

                        if polygon.is_valid:
                            new_polygon = polygon.intersection(map_patch)
                            if not new_polygon.is_empty:
                                # new_polygon = affinity.rotate(new_polygon, -patch_angle,
                                #                             origin=(patch_x, patch_y), use_radians=False)
                                # new_polygon = affinity.affine_transform(new_polygon,
                                #                                         [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                                if new_polygon.geom_type is 'Polygon':
                                    new_polygon = MultiPolygon([new_polygon])
                                new_polygon = transform_polygon(new_polygon, inv_tfm)
                                geoms.append(new_polygon)
                map_geom.append((layer_name, geoms))
            elif layer_name in self.ped_crossing_classes: # ['ped_crossing']
                # geoms = self.get_ped_crossing_line(patch_box, patch_angle, location)
                geoms = self.get_ped_crossing_line(map_patch, inv_tfm, location)
                # geoms = self.map_explorer[location]._get_layer_polygon(patch_box, patch_angle, layer_name)
                map_geom.append((layer_name, geoms))
        return map_geom

    def _one_type_line_geom_to_vectors(self, line_geom):
        # line_geom
        line_vectors = []
        for line in line_geom:
            if not line.is_empty:
                if line.geom_type == 'MultiLineString':
                    for l in line:
                        line_vectors.append(self.sample_pts_from_line(l))
                elif line.geom_type == 'LineString': # road_divider, lane_divider
                    line_vectors.append(self.sample_pts_from_line(line))
                else:
                    raise NotImplementedError
        return line_vectors # list of [array of sampled points, num of points]

    def poly_geoms_to_vectors(self, polygon_geom, extents):
        roads = polygon_geom[0][1] # road_segment
        lanes = polygon_geom[1][1] # lane
        union_roads = ops.unary_union(roads)
        union_lanes = ops.unary_union(lanes)
        union_segments = ops.unary_union([union_roads, union_lanes])
        max_x = self.patch_size[1] / 2 # 30 50.0
        max_y = self.patch_size[0] / 2 # 15 49.0
        x1, z1, x2, z2 = extents # [-25.0, 1.0, 25.0, 50.0]
        local_patch = box(x1 + 0.2, z1 + 0.2, x2 - 0.2, z2 - 0.2)
        exteriors = []
        interiors = []
        if union_segments.geom_type != 'MultiPolygon':
            union_segments = MultiPolygon([union_segments])
        for poly in union_segments:
            exteriors.append(poly.exterior)
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

        return self._one_type_line_geom_to_vectors(results)

    def line_geoms_to_vectors(self, line_geom):
        line_vectors_dict = dict()
        for line_type, a_type_of_lines in line_geom:
            # line_type: str type (e.g. 'lane_divider'), a_type_of_lines: shapely LineString
            one_type_vectors = self._one_type_line_geom_to_vectors(a_type_of_lines)
            # one_type_vectors: list of [array of sampled points, num of points]
            line_vectors_dict[line_type] = one_type_vectors

        return line_vectors_dict

    def ped_geoms_to_vectors(self, ped_geom):
        ped_geom = ped_geom[0][1]
        union_ped = ops.unary_union(ped_geom)
        if union_ped.geom_type != 'MultiPolygon':
            union_ped = MultiPolygon([union_ped])

        max_x = self.patch_size[1] / 2 # 30
        max_y = self.patch_size[0] / 2 # 15
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        results = []
        for ped_poly in union_ped:
            # rect = ped_poly.minimum_rotated_rectangle
            ext = ped_poly.exterior
            if not ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            results.append(lines)

        return self._one_type_line_geom_to_vectors(results)

    # def get_ped_crossing_line(self, patch_box, patch_angle, location):
    #     def add_line(poly_xy, idx, patch, patch_angle, patch_x, patch_y, line_list):
    #         points = [(p0, p1) for p0, p1 in zip(poly_xy[0, idx:idx + 2], poly_xy[1, idx:idx + 2])]
    #         line = LineString(points)
    #         line = line.intersection(patch)
    #         if not line.is_empty:
    #             line = affinity.rotate(line, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
    #             # move lines so that ego pose is set as origin
    #             line = affinity.affine_transform(line, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
    #             line_list.append(line)

    #     patch_x = patch_box[0]
    #     patch_y = patch_box[1]

    #     patch = NuScenesMapExplorer.get_patch_coord(patch_box, patch_angle) # patch in global coord
    #     # ego_x + 25.5, ego_y + 0, 49.0, 50.0 -> 
    #     line_list = []
    #     records = getattr(self.nusc_maps[location], 'ped_crossing')
    #     for record in records:
    #         polygon = self.map_explorer[location].extract_polygon(record['polygon_token'])
    #         poly_xy = np.array(polygon.exterior.xy)
    #         dist = np.square(poly_xy[:, 1:] - poly_xy[:, :-1]).sum(0)
    #         x1, x2 = np.argsort(dist)[-2:]

    #         add_line(poly_xy, x1, patch, patch_angle, patch_x, patch_y, line_list)
    #         add_line(poly_xy, x2, patch, patch_angle, patch_x, patch_y, line_list)

    #     return line_list

    def get_ped_crossing_line(self, map_patch, inv_tfm, location):
        def add_line(poly_xy, idx, patch, inv_tfm, line_list):
            points = [(p0, p1) for p0, p1 in zip(poly_xy[0, idx:idx + 2], poly_xy[1, idx:idx + 2])]
            line = LineString(points)
            line = line.intersection(patch)
            if not line.is_empty:
                # line = affinity.rotate(line, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
                # # move lines so that ego pose is set as origin
                # line = affinity.affine_transform(line, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                line = transform_polygon(line, inv_tfm)
                line_list.append(line)

        # patch_x = patch_box[0]
        # patch_y = patch_box[1]

        # patch = NuScenesMapExplorer.get_patch_coord(patch_box, patch_angle) # patch in global coord
        # ego_x + 25.5, ego_y + 0, 49.0, 50.0 -> 
        line_list = []
        records = getattr(self.nusc_maps[location], 'ped_crossing')
        for record in records:
            polygon = self.map_explorer[location].extract_polygon(record['polygon_token'])
            poly_xy = np.array(polygon.exterior.xy)
            dist = np.square(poly_xy[:, 1:] - poly_xy[:, :-1]).sum(0)
            x1, x2 = np.argsort(dist)[-2:]

            add_line(poly_xy, x1, map_patch, inv_tfm, line_list)
            add_line(poly_xy, x2, map_patch, inv_tfm, line_list)

        return line_list

    def sample_pts_from_line(self, line):
        if self.fixed_num < 0: # default: -1
            distances = np.arange(0, line.length, self.sample_dist) # sample_dist: 1
            sampled_points = np.array([list(line.interpolate(distance).coords) for distance in distances]).reshape(-1, 2) # Nx2
        else:
            # fixed number of points, so distance is line.length / self.fixed_num
            distances = np.linspace(0, line.length, self.fixed_num)
            sampled_points = np.array([list(line.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)

        if self.normalize: # normalize by (x=60.0, y=30.0)
            sampled_points = sampled_points / np.array([self.patch_size[1], self.patch_size[0]])

        num_valid = len(sampled_points)

        if not self.padding or self.fixed_num > 0:
            # fixed num sample can return now!
            return sampled_points, num_valid

        # fixed distance sampling need padding!
        num_valid = len(sampled_points)

        if self.fixed_num < 0:
            if num_valid < self.num_samples: # num_samples=250
                padding = np.zeros((self.num_samples - len(sampled_points), 2))
                sampled_points = np.concatenate([sampled_points, padding], axis=0)
            else:
                sampled_points = sampled_points[:self.num_samples, :]
                num_valid = self.num_samples

            if self.normalize:
                sampled_points = sampled_points / np.array([self.patch_size[1], self.patch_size[0]])
                num_valid = len(sampled_points)

        return sampled_points, num_valid
