# comfig
# ml_landcover_classifier/
# Initial scaffold and configuration

# config.py
# Contains global settings and label mappings

# surface classes (flat label list)
SURFACE_CLASSES = [
    "vegetation",
    "bare_ground",
    "water",
    "buildings",
    "roads",
    "railway",
    "sidewalk",
    "private_areas",
    "others"
]

# whether the surface is accessible or not
ACCESSIBILITY_MAP = {
    "vegetation": "accessible",
    "bare_ground": "accessible",
    "water": "inaccessible",
    "buildings": "inaccessible",
    "roads": "accessible",
    "railway": "inaccessible",
    "sidewalk": "accessible",
    "private_areas": "inaccessible",
    "others": "inaccessible"
}

EPSG_CODE = 6991 # editable, set for Israeli projects
DEFAULT_PATCH_SIZE = 32  # editable, can be overridden per image
AVAILABLE_MODELS = ["RandomForest", "SVM", "XGBoost"] # editable: model choices
TRAIN_TEST_SPLIT_RATIO = 0.8  # editable: train ratio value (0.8 train -> 0.2 test, 0.7-0.3 etc)

# file paths (adjusted at runtime via GUI)
MODEL_DIR = "model_weights/"
OUTPUT_DIR = "outputs/"
TRAIN_DATA_DIR = "train_data/"
LOG_DIR = "logs/"

# feature extraction parameters
INCLUDE_RGB = True # editable
INCLUDE_HARALICK = True # editable
INCLUDE_ENTROPY = True # editable
INCLUDE_NDVI = False  # true: if NIR is available

# fallback settings
RANDOM_SEED = 42
VERBOSE = True

# GUI settings
ENABLE_GUI = True  # editable, keep on
GUI_DEFAULT_WIDTH = 800
GUI_DEFAULT_HEIGHT = 600

# Default settings used in GUI\ CLI
DEFAULT_PATCH_SIZE = 64  # editable
DEFAULT_MODEL_NAME = "randomforest"  # editable: choose model