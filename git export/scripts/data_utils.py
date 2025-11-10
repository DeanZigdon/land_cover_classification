# data_utils.py

"""
this script loads GeoTIFF images, masks white pixels, generates image patches and extracts labels from ground truth masks
it also functions for downsampling and managing patch sizes
"""
import rasterio
import numpy as np
import logging
import os
from config import DEFAULT_PATCH_SIZE, EPSG_CODE
from scipy.stats import mode

def load_geotiff(path,bands_to_read=None):
    """
    load GeoTIFF image using rasterio with optional band selection,returns metadata of available bands

    parameters:
        path (str): Path of GeoTIFF file
        bands_to_read (list, optional): list of 1 based band indices to read
                                        if 'None' all bands are read

    return:
        tuple: (image_array, transform, crs, band_info)
                image_array: HxWxC array for each selected band
                transform: affine transform
                crs (isr projects- 6991)
                band_info: dict with 'count' (total bands) and 'names' (list of band descriptions/numbers)
    """
    with rasterio.open(path) as src:
        total_bands = src.count
        transform = src.transform
        crs = src.crs
        band_names = [src.descriptions[i-1] if src.descriptions and src.descriptions[i-1] else f"Band {i}" for i in range(1, total_bands + 1)]
        
        band_info = {
            'count': total_bands,
            'names': band_names}

        if bands_to_read:
            # validate requested bands
            valid_bands_to_read = [b for b in bands_to_read if 1 <= b <= total_bands]
            
            if not valid_bands_to_read:
                logging.error(f"no valid bands selected from {bands_to_read}. Reading all available bands as fallback.")
                image = src.read().transpose(1, 2, 0)
            else:
                image = src.read(valid_bands_to_read).transpose(1, 2, 0)
        else:
            image = src.read().transpose(1, 2, 0)  # read all bands if none specified

    logging.info(f"loaded image from {os.path.basename(path)} with shape {image.shape}. Original bands: {band_info['count']} ({', '.join(band_info['names'])})")
    
    return image, transform, crs, band_info

def mask_white_pixels(image, white_threshold=250, replace_value=0):
    """
    filter function
    masks out "pure white" or very bright pixels of the image

    parameters:
        image (np.ndarray): HxWxC RGB image array (uint8)
        white_threshold (int): pixels where all bands are >= this value will be masked
                               default 250 to allow for small variations from 255
        replace_value (int/float): value to replace masked pixels with
                                   0 (black) = common\np.nan if feature extractor
                                   can handle NaN (ignore them\interpolate).
                                   for RandomForest 0 is better over NaN.

    returns:
        np.ndarray: image with white pixels masked out
    """
    if image.ndim != 3 or image.shape[2] < 3:
        logging.warning("image is not a multi-channel (likely RGB) image. skipping white pixel masking.")
        return image

    # create boolean mask where all channels are above the threshold
    # using 'all' ensures its only truly white/overexposed regions
    temp_image = image.astype(np.float32)
    mask = np.all(temp_image >= white_threshold, axis=-1)

    # apply the mask, ensure replace_value is compatible with image dtype
    masked_image = image.copy()
    masked_image[mask] = replace_value

    logging.info(f"masked {np.sum(mask)} pure white/overexposed pixels.")
    return masked_image

def generate_adaptive_patches(image, patch_size=DEFAULT_PATCH_SIZE, stride=None, max_patches=None):
    """
    split image into overlapping patches (adaptive stride optional).
    returns:
        List of (x, y, patch) tuples
    """
    if stride is None:
        stride = patch_size // 2  # editable: default 50% overlap

    patches = []
    h, w, _ = image.shape
    patch_count = 0

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            if max_patches is not None and patch_count >= max_patches:
                return patches
            patch = image[y:y+patch_size, x:x+patch_size]
            patches.append((x, y, patch))
            patch_count += 1

    print(f"[generate_adaptive_patches] generated {len(patches)} patches")

    return patches


def get_labels_at_patch_centroids(patches, ground_truth_mask):
    """
    determines the ground truth label for each patch by sampling the pixel
    at the centroid of the patch from a ground truth mask

    parameters:
        patches (list): list of patches (x, y, patch_data) from raster image
        ground_truth_mask (np.ndarray): rasterized ground truth mask image
        
    returns:
        list: list of intger labels, one for each patch
    """
    labels = []
    
    # ensure the ground truth mask is 2D (HxW) for simple indexing
    if ground_truth_mask.ndim > 2:
        logging.warning("ground truth mask has multiple bands. Using the first band for evaluation.")
        gt_mask = ground_truth_mask[:, :, 0]
    else:
        gt_mask = ground_truth_mask

    for x, y, _ in patches:
        # calculate the centroid pixel coordinates of the patch
        centroid_x = x + (patches[0][2].shape[1] // 2)
        centroid_y = y + (patches[0][2].shape[0] // 2)

        # get the label from the ground truth mask at the centroid
        if 0 <= centroid_y < gt_mask.shape[0] and 0 <= centroid_x < gt_mask.shape[1]:
            label = gt_mask[centroid_y, centroid_x]
            labels.append(int(label))
        else:
            # fallback for patches that could be outside mask bounds
            labels.append(0) 

    return labels

def reconstruct_from_predictions(pred_map, patch_size, image_shape):
    """
    incomplete function, for further improvements
    optional improvement: reconstruct prediction raster from patch wise predictions
    """
    raise NotImplementedError("reconstruction logic to be added later.")


def downsample_raster(array, factor=2):
    """
    for evaluation mode debugging
    downsample the raster by X integer factor, keeping adjacency
    input:
        array (np.ndarray): input raster as 2D np array
        factor (int): downsampling factor: 2 reduces size by 4.
    returns:
        np.ndarray: downsampled raster
    """
    if factor <= 1:
        return array
    h, w = array.shape
    new_h, new_w = h // factor, w // factor
    return array[:new_h*factor, :new_w*factor].reshape(new_h, factor, new_w, factor).mean(axis=(1,3))

def compute_downsample_factor(n_patches, max_patches):
    """
    compute the smallest integer downsample factor so:
    n_patches / factor^2 <= max_patches
    args:
        n_patches (int): current number of patches
        max_patches (int): max allowed patches
    returns:
        int: downsampling factor (>=1)
    """
    if max_patches is None or n_patches <= max_patches:
        return 1
    factor = 1
    while (n_patches // (factor**2)) > max_patches:
        factor += 1
    return factor
