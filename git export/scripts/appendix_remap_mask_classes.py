# appendix_remap_mask_classes
"""
this file should be executed after "appendix_mask_file_config", it guides the user into reclassifying the classes in the mask file to fit the projects classes (vital preprocess before training the model)
"""
import rasterio
import numpy as np
import os

# input directory of *original* LandCover.ai mask files are located
input_masks_dir = "train_data/unedited/masks"  # editable: change as folder needed
# output directory to save the *remapped* mask files
output_masks_dir = "train_data/edited/masks" # editable: change as folder needed

# define the remapping dictionary: {original_value: new_value}
# based on config.py:
# LandCover.ai preset: 0=Building, 1=Woodland, 2=Water, 3=Road
# this project: 0=vegetation, 1=bare_ground, 2=water, 3=buildings, 4=roads,5= railway, 6=sidewalk, 7=private_areas, 8=others

REMAP_MAP = {
    0: 3,  # original Building (0) -> buildings (3)
    1: 0,  # original Woodland (1) -> vegetation (0)
    2: 2,  # original Water (2) ->  water (2) - match, verification needed anyway
    4: 8   # original Unknown Value (4) -> Others (8)
}

# creates output directory if it doesnt exist
os.makedirs(output_masks_dir, exist_ok=True)

# process all mask files
print(f"--- starting mask remapping from {input_masks_dir} to {output_masks_dir} ---")
processed_count = 0

for filename in os.listdir(input_masks_dir):
    if filename.endswith(".tif"): # mask files must be TIFF 
        input_filepath = os.path.join(input_masks_dir, filename)
        output_filepath = os.path.join(output_masks_dir, f"remapped_{filename}") #adds prefix

        try:
            with rasterio.open(input_filepath) as src:
                mask_data = src.read(1) # read the first band
                # creates editable copy
                remapped_mask_data = mask_data.copy()
                # apply the remapping
                for original_val, new_val in REMAP_MAP.items():
                    remapped_mask_data[mask_data == original_val] = new_val

                # for any values not explicitly in REMAP_MAP (values that were not 0,1,2,3,4),
                # it verifies all pixels have valid class index from the schema
                # could require future adjustment if adding other unmapped values.
                # in this project, I expect only the values defined in REMAP_MAP
                # but if there were others, they would remain unchanged when not mapped
                # and cause errors.

                # gets metadata for the new TIFF
                profile = src.profile
                profile.update(
                    dtype=rasterio.uint8, # verify output is byte array (0-255)
                    count=1 )# verify output is single band

                with rasterio.open(output_filepath, 'w', **profile) as dst:
                    dst.write(remapped_mask_data.astype(rasterio.uint8), 1)

            print(f"Successfully remapped: {filename} -> {os.path.basename(output_filepath)}")
            processed_count += 1

        except FileNotFoundError:
            print(f"Error: Mask file not found at {input_filepath}. Skipping.")
        except Exception as e:
            print(f"An error occurred while processing {filename}: {e}")

print(f"\n--- Remapping complete! Processed {processed_count} mask files. ---")
print(f"Remapped masks saved to: {output_masks_dir}")