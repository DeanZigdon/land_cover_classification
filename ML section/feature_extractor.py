# feature extractor
import numpy as np
import cv2
try:
    from skimage.feature import graycomatrix, graycoprops
except ImportError:
    from skimage.feature.texture import greycomatrix, greycoprops
    graycomatrix, graycoprops = greycomatrix, greycoprops

from skimage.filters.rank import entropy
from skimage.morphology import disk
import logging

from config import INCLUDE_RGB, INCLUDE_HARALICK, INCLUDE_ENTROPY


def extract_features(patch):
    """
    Extracts selected features from a given RGB patch.
    Includes RGB values, Haralick textures, and entropy.

    Parameters:
        patch (np.ndarray): HxWx3 RGB image patch

    Returns:
        feature_vector (np.ndarray): 1D array of concatenated features
    """
    features = []

    if INCLUDE_RGB:
        rgb_mean = patch.mean(axis=(0, 1))  # mean R, G, B
        features.extend(rgb_mean.tolist())

    if INCLUDE_HARALICK:
        # Convert to grayscale for GLCM
        gray_patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        glcm = greycomatrix(gray_patch, [1], [0], 256, symmetric=True, normed=True)
        contrast = greycoprops(glcm, 'contrast')[0, 0]
        homogeneity = greycoprops(glcm, 'homogeneity')[0, 0]
        features.extend([contrast, homogeneity])

    if INCLUDE_ENTROPY:
        gray_patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        ent = entropy(gray_patch, disk(3)).mean()
        features.append(ent)

    return np.array(features, dtype=np.float32)
