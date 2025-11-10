# config.py
"""configuration file for the project, most of the editbale features are located here"""
import os
import numpy as np

# surface classes (with numbering)
SURFACE_CLASSES = [
    "vegetation", # 0
    "bare_ground", # 1
    "water", # 2
    "buildings", # 3
    "roads", # 4
    "railway", # 5
    "sidewalk",# 6
    "private_areas", # 7
    "others"]# 8

NUM_CLASSES = len(SURFACE_CLASSES)
#  editable: surface class accessibility = accessible\inaccessible
ACCESSIBILITY_MAP = {
    "vegetation": "accessible",
    "bare_ground": "accessible",
    "water": "inaccessible",
    "buildings": "inaccessible",
    "roads": "accessible",
    "railway": "inaccessible",
    "sidewalk": "accessible",
    "private_areas": "inaccessible",
    "others": "inaccessible"}

EPSG_CODE = 6991 # editable: set for ISR projects
DEFAULT_PATCH_SIZE = 32  # editable: can be overridden per image
AVAILABLE_MODELS = ["RandomForest", "SVM", "XGBoost"] # editable: model choices, can be switched from GUI interface aswell
TRAIN_TEST_SPLIT_RATIO = 0.8  # editable: train ratio value (0.8 train 0.2 test, 0.7-0.3 etc)

# file paths (adjusted at runtime via GUI)
MODEL_DIR = "model_weights/"
OUTPUT_DIR = "ML section/tests/" # change to match local path
TRAIN_DATA_DIR = "train_data/" # this is where "raw" training images/masks will be stored
LOG_DIR = "logs/"  # txt files output, ovverides per run (aprx 4.5gb)

# feature extraction parameters
INCLUDE_RGB = True # editable
INCLUDE_HARALICK = True # editable
INCLUDE_ENTROPY = True # editable
INCLUDE_NDVI = False  # true: if NIR is available


AVAILABLE_FEATURES = ["mean_R", "mean_G", "mean_B",
    "haralick_contrast", "haralick_homogeneity","entropy",]

# grouping (used to build active list from flags)
RGB_FEATURES = ["mean_R", "mean_G", "mean_B"]
HARALICK_FEATURES = ["haralick_contrast", "haralick_homogeneity"]
ENTROPY_FEATURES = ["entropy"]

FEATURE_LIST = [f for f in AVAILABLE_FEATURES
    if ((INCLUDE_RGB and f in RGB_FEATURES)
        or (INCLUDE_HARALICK and f in HARALICK_FEATURES)
        or (INCLUDE_ENTROPY and f in ENTROPY_FEATURES))]

# haralick glcm parameters (used by feature_extractor.py)
GLCM_DISTANCES = [1]
GLCM_ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4] # 0, 45, 90, 135 degrees
GLCM_LEVELS = 256 #  skimage default

# fallback settings
RANDOM_SEED = 42
VERBOSE = True

# GUI settings
ENABLE_GUI = True  # editable
GUI_DEFAULT_WIDTH = 800
GUI_DEFAULT_HEIGHT = 600

# default settings used in GUI\ CLI
DEFAULT_PATCH_SIZE = 64  # editable
DEFAULT_MODEL_NAME = "randomforest_6_features"  # editable: choose model

# patch limits for all modes. None = all , 30,000 is a good sample run for now
PATCH_LIMITS = {
    "prediction": {
        "max_patches": None,  # None = all.  Max patches for fast prediction mode
        "description": "Limit patches during quick prediction run. Set None for all pixels."},
    "evaluation": {
        "max_patches": None,    # None = all patches to match ground truth
        "description": "Limit patches during evaluation. Set None to use the full ground truth raster."},
    "ground_truth": {
        "max_patches": None,    # None = all ppatches. can be used to limit the ground truth raster if needed
        "description": "Limit patches read from ground truth raster for evaluation. Keep None for full raster."}}

# max patches for prediction mode (None = all pixels)  (all next 3 variables are needed)
PREDICT_MAX_PATCHES = PATCH_LIMITS["prediction"]["max_patches"] # None= all input pixels, any other number: sampling size.
# max patches for evaluation mode (not affecting prediction mode)
EVAL_MAX_PATCHES = PATCH_LIMITS["evaluation"]["max_patches"] # None: all input pixels, any other number: evaluation pixels limit. keep in mind that it should be equal to 'GROUND_TRUTH_RASTER' ground truth raster
# max patches for ground truth raster (change only for evaluation mode)
GROUND_TRUTH_RASTER = PATCH_LIMITS["ground_truth"]["max_patches"] # None: all input pixels, any other number: X pixels limit. keep in mind that it should be equal to 'EVAL_MAX_PATCHES' (max patches for evaluation mode)

# mapping for classes and indices
CLASS_TO_INDEX = {cls: i for i, cls in enumerate(SURFACE_CLASSES)} # class_to_index map
INDEX_TO_CLASS = {i: cls for i, cls in enumerate(SURFACE_CLASSES)} # index_to_class map

# full path to the default model file
MODEL_PATH = os.path.join(MODEL_DIR, DEFAULT_MODEL_NAME + ".joblib")

# paths for orthophoto labeling
# these are specific inputs for the 'train' command when using manually labeled data
MANUAL_LABEL_ORTHOPHOTO_PATH = os.path.join("orthophoto", "result.tif") # editable if other relevant tif files are availble

#LABELLED_SHP_PATH is for original shapefile, not the rasterized mask
LABELLED_SHP_PATH = os.path.join("orthophoto", "training_labels_terminus_30.7.shp")
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"

# rasterization paths
# verify that these paths align with saved shapefile and the raster mask location
RASTERIZE_ORTHOPHOTO_PATH = MANUAL_LABEL_ORTHOPHOTO_PATH 
RASTERIZE_SHAPEFILE_PATH = LABELLED_SHP_PATH # labeled shapefile
RASTERIZE_OUTPUT_MASK_FILENAME = "training_labels_rasterized.tif" # editable: name for output mask
RASTERIZE_OUTPUT_MASK_PATH = os.path.join(os.path.dirname(RASTERIZE_ORTHOPHOTO_PATH), RASTERIZE_OUTPUT_MASK_FILENAME)
MANUAL_LABEL_RASTERIZED_MASK_PATH = RASTERIZE_OUTPUT_MASK_PATH


# the column in the shapefile that holds the class names
SHAPEFILE_CLASS_COLUMN = 'class_name' # editable: change only when the feature in the shp file is changed

# band selection for feature extraction
# example: [1, 2, 3] for RGB, [1, 2, 3, 4] RGB + NIR (# order is constant)
# the number of selected bands MUST match the expected features of the  model
# another example: for orthophoto with RGB only + model trained by RGB + Haralick + Entropy (3 + 2 + 1 = 6 features)
# then [1, 2, 3] = RGB and feature_extractor.py will add the others

# additional band feature counting explanation:
# if INCLUDE_RGB is True, it adds 3 features (R, G, B)
# if INCLUDE_HARALICK is True, it adds 2 features (contrast, homogeneity)
# if INCLUDE_ENTROPY is True, it adds 1 feature (entropy)
# total features = 3 (RGB) + 2 (Haralick) + 1 (Entropy) = 6 features tot

SELECTED_BANDS_FOR_MODEL = [1, 2, 3] # standard RGB bands