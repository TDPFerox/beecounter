import numpy as np
from scipy.ndimage import gaussian_filter

H, W = 288, 512
density = np.zeros((H, W), dtype=np.float32)


def generate_density_map(points, height, width, sigma=6):
    density = np.zeros((height, width), dtype=np.float32)

    for x, y in points:
        x = int(round(x))
        y = int(round(y))

        if 0 <= x < width and 0 <= y < height:
            density[y, x] += 1.0

    density = gaussian_filter(density, sigma=sigma)

    return density
