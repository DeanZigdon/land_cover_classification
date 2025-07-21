# data_utils
import rasterio
import numpy as np
import logging
from config import DEFAULT_PATCH_SIZE, EPSG_CODE


def load_geotiff(path):
    """
    Load a GeoTIFF image using rasterio.

    Returns:
        image: HxWxC array
        transform: Affine transform
        crs: Coordinate reference system
    """
    with rasterio.open(path) as src:
        image = src.read().transpose(1, 2, 0)  # CHW to HWC
        transform = src.transform
        crs = src.crs

    return image, transform, crs


def generate_adaptive_patches(image, patch_size=DEFAULT_PATCH_SIZE, stride=None):
    """
    Split image into overlapping patches (adaptive stride optional).

    Returns:
        List of (x, y, patch) tuples
    """
    if stride is None:
        stride = patch_size // 2  # 50% overlap by default

    patches = []
    h, w, _ = image.shape

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image[y:y+patch_size, x:x+patch_size]
            patches.append((x, y, patch))

    return patches


def reconstruct_from_predictions(pred_map, patch_size, image_shape):
    """
    (Optional) Reconstruct prediction raster from patch-wise predictions.
    """
    raise NotImplementedError("Reconstruction logic to be added later.")
