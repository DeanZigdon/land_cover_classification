#appendix_mask_file_config
"""
this script reconfigures the preset labels for landcovers into the wanted labels (classes) desired
"""

import rasterio
import numpy as np
from config import SURFACE_CLASSES, CLASS_TO_INDEX, INDEX_TO_CLASS

# path to the train data mask file
mask_file_path = "train_data/masks/M-33-7-A-d-2-3.tif" # editable, "train_data/1_1_mask.tif"

# LandCover.ai known original labels 
LANDCOVER_AI_ORIG_LABELS = {
    0: "Building",
    1: "Woodland",
    2: "Water",
    3: "Road"}
print(f"--- Validating Mask: {mask_file_path} ---")

try:
    with rasterio.open(mask_file_path) as src:
        mask_data = src.read(1) # read first band of the mask

    unique_values = np.unique(mask_data)
    print(f"unique pixel values found in mask: {unique_values}")
    print("\n--- Mapping LandCover.ai labels to your SURFACE_CLASSES ---")
    for original_val in unique_values:
        if original_val in LANDCOVER_AI_ORIG_LABELS:
            original_label_name = LANDCOVER_AI_ORIG_LABELS[original_val]
            print(f"  LandCover.ai Value {original_val} corresponds to: '{original_label_name}'")

            # re labeling
            your_class_name = None
            if original_label_name == "Building":
                your_class_name = "buildings"
            elif original_label_name == "Woodland":
                your_class_name = "vegetation"
            elif original_label_name == "Water":
                your_class_name = "water"
            elif original_label_name == "Road":
                your_class_name = "roads"
            # possible addition of specific mappings if LandCover.ai has other values for refinement

            if your_class_name:
                if your_class_name in CLASS_TO_INDEX:
                    your_index = CLASS_TO_INDEX[your_class_name]
                    print(f"    -> Your Project's Class: '{your_class_name}' (Index: {your_index})")
                    if original_val == your_index:
                        print(f"       ✅ direct match found! (mask value {original_val} is also your index for '{your_class_name}')")
                    else:
                        print(f"       ⚠️ mismatch! mask value {original_val} needs to be remapped to index {your_index} for '{your_class_name}'.")
                else:
                    print(f"    -> Your Project's Class: '{your_class_name}' (NOT IN YOUR SURFACE_CLASSES!)")
            else:
                print(f"    -> No direct mapping rule defined for '{original_label_name}' in your project.")
        else:
            print(f"  Unknown value {original_val} found in mask. Not in LandCover.ai's known labels.")

except FileNotFoundError:
    print(f"Error: Mask file not found at {mask_file_path}. Please check the path and filename.")
except Exception as e:
    print(f"An error occurred: {e}")

print("\n--- Your Project's Configured Classes ---")
for idx, cls_name in INDEX_TO_CLASS.items():
    print(f"  Index {idx}: '{cls_name}'")