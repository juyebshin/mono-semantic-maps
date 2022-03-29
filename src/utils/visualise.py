from matplotlib.cm import get_cmap
import numpy as np
from PIL import Image

def colorise(tensor, cmap, vmin=None, vmax=None):

    if isinstance(cmap, str):
        cmap = get_cmap(cmap)
    
    tensor = tensor.detach().cpu().float()

    vmin = float(tensor.min()) if vmin is None else vmin
    vmax = float(tensor.max()) if vmax is None else vmax

    tensor = (tensor - vmin) / (vmax - vmin)
    return cmap(tensor.numpy())[..., :3] # :3 -> 0, 1, 2

def get_color_pallete(npimg):
    out_img = Image.fromarray(npimg.astype('uint8'))
    out_img.putpalette(palette)

palette = []

color_map = {
    'drivable_area': (0, 0, 128), # Navy 0
    'ped_crossing': (128, 0, 0), # Maroon 1
    'walkway': (0, 128, 128), # Teal 2
    'carpark_area': (128, 128, 0), # Olive 3
    'car': (255, 0, 0), # Red 4
    'truck': (0, 255, 0), # Lime 5
    'bus': (0, 0, 255), # Blue 6
    'trailer': (0, 255, 255), # Cyan 7
    'construction_vehicle': (255, 0, 255), # Magenta 8
    'pedestrian': (255, 255, 0), # Yellow 9
    'motorcycle': (0, 128, 0), # Green 10
    'bicycle': (128, 0, 128), # Purple 11
    'traffic_cone': (255, 140, 0), # Dark orange 12
    'barrier': (128, 128, 128) # Gray 13
}

map_layer_color = {
    'drivable_area': (0, 0, 128), # Navy 0
    'ped_crossing': (128, 0, 0), # Maroon 1
    'walkway': (0, 128, 128), # Teal 2
    'carpark_area': (128, 128, 0), # Olive 4
    'road_divider': (255, 255, 0), # Yellow 5
    'lane_divider': (0, 128, 0), # Green 6
}

hdmapnet_color = {
    'line_classes': (0, 128, 0), # Red 4
    'ped_crossing': (255, 255, 0), # Yellow 5
    'contour': (255, 0, 0), # Green 3
}