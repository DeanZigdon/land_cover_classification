# feature_extractor
"""
this file extracts features based on INCLUDE_RGB, INCLUDE_HARALICK, INCLUDE_ENTROPY or any other vectors presetted for the process
"""


import numpy as np
import cv2
try:
    from skimage.feature import graycomatrix, graycoprops
except ImportError:
    from skimage.feature.texture import graycomatrix, graycoprops
    graycomatrix, graycoprops = graycomatrix, graycoprops

from skimage.filters.rank import entropy
from skimage.morphology import disk
import logging
from config import INCLUDE_RGB, INCLUDE_HARALICK, INCLUDE_ENTROPY, \
                   GLCM_DISTANCES, GLCM_ANGLES, GLCM_LEVELS, FEATURE_LIST


def extract_features(patch):
    """
    extracts selected features from a given rgb patch
    includes rgb values, haralick textures, and entropy

    parameters:
        patch (np.ndarray): HxWxC rgb image patch (loaded as hwc)
    returns:
        feature_vector (np.ndarray): 1d array of concatenated features
    """
    FEATURE_NAMES = list(FEATURE_LIST)  # exact order used
    features = []

    # guard: empty patch -> zeros of correct length
    if patch.size == 0:
        logging.warning("empty patch provided to extract_features; filling zeros for all requested features.")
        logging.info(f"Feature names: {FEATURE_NAMES}")
        return np.zeros(len(FEATURE_LIST), dtype=np.float32)

    # takes first 3 channels for RGB means to avoid RGBA alpha 'leakage'
    if patch.ndim == 3 and patch.shape[2] >= 3:
        rgb3 = patch[:, :, :3]
    elif patch.ndim == 3 and patch.shape[2] == 1:
        rgb3 = np.repeat(patch, 3, axis=2)
    else:
        rgb3 = np.stack([patch]*3, axis=2) if patch.ndim == 2 else patch

    rgb_means = rgb3.mean(axis=(0, 1)).tolist()  # mean RGB

    need_gray = any(n in FEATURE_LIST for n in ("haralick_contrast","haralick_homogeneity","entropy"))
    gray_patch, contrast, homogeneity = None, None, None
    if need_gray:
        if rgb3.shape[2] >= 3:
            gray_patch = cv2.cvtColor(rgb3, cv2.COLOR_RGB2GRAY)
        else:
            gray_patch = rgb3[:, :, 0]

        if gray_patch.dtype != np.uint8: # glcm requires integer type, 8-bit (0-255)
            m = float(gray_patch.max())
            gray_patch = (np.zeros_like(gray_patch, dtype=np.uint8)
                          if m == 0 else ((gray_patch / m) * 255).astype(np.uint8))  

        if any(n in FEATURE_LIST for n in ("haralick_contrast","haralick_homogeneity")):
            try:
                levels = max(1, GLCM_LEVELS)
                gray_q = gray_patch // max(1, (256 // levels))
                if (gray_q.shape[0] >= max(GLCM_DISTANCES)+1) and (gray_q.shape[1] >= max(GLCM_DISTANCES)+1):
                    glcm = graycomatrix(gray_q, GLCM_DISTANCES, GLCM_ANGLES, levels, symmetric=True, normed=True)
                    contrast = graycoprops(glcm, 'contrast')[0, 0]   # INPORTANT note: if edits are made, keep "graycoprops" and not "greycoprops" to avoid errors
                    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
                else:
                    logging.warning(f"patch too small for GLCM distances {GLCM_DISTANCES}. Haralick -> zeros.")
                    contrast, homogeneity = 0.0, 0.0
            except Exception as e:
                logging.warning(f"GLCM/HARALICK failed: {e}. Using zeros.")
                contrast, homogeneity = 0.0, 0.0
    #  editable: must be perciesly simillar to the feature_list in config.py for correct vectors calculations 
    for name in FEATURE_LIST:
        if name == "mean_R":
            features.append(float(rgb_means[0]))
        elif name == "mean_G":
            features.append(float(rgb_means[1]))
        elif name == "mean_B":
            features.append(float(rgb_means[2]))
        elif name == "haralick_contrast":
            features.append(0.0 if contrast is None else float(contrast))
        elif name == "haralick_homogeneity":
            features.append(0.0 if homogeneity is None else float(homogeneity))
        elif name == "entropy":
            try:
                ent = entropy(gray_patch, disk(3)).mean() if gray_patch is not None else 0.0
            except Exception as e: # troubleshooting entropy
                logging.warning(f"entropy failed: {e}. Using 0.")
                ent = 0.0
            features.append(float(ent))
        else:
            logging.error(f"Unknown feature '{name}' in FEATURE_LIST. Appending 0.") 
            features.append(0.0)

    logging.info(f"Extracted {len(features)} features: {features}")
    logging.info(f"Feature names: {FEATURE_NAMES}")
    return np.array(features, dtype=np.float32)