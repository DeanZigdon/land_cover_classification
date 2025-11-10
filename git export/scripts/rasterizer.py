# rasterizer
"""rasterization of original orthophoto files"""

import rasterio
from rasterio.features import rasterize
import geopandas as gpd
import numpy as np
import logging
import os
from config import CLASS_TO_INDEX, EPSG_CODE

def rasterize_labels(orthophoto_path, labeled_shp_path, output_mask_path, class_column='Class_Name'):
    """
    rasterizes a labeled shapefile to create a pixel mask aligned with an orthophoto

    parameters:
        orthophoto_path (str): path to the input orthophoto GeoTIFF
        labeled_shp_path (str): path to the labeled shapefile (shp)
        output_mask_path (str): path to the rasterized mask GeoTIFF output
        class_column (str): column name  in the shpfile attribute table
                            that contains the class names ("vegetation","buildings"...)

    returns:
        str: path to the generated rasterized mask or None if failed
    """
    logging.info(f"starting rasterization process for {labeled_shp_path} to {output_mask_path}")

    if not os.path.exists(orthophoto_path):
        logging.error(f"orthophoto not found at {orthophoto_path}.")
        return None
    if not os.path.exists(labeled_shp_path):
        logging.error(f"labeled shapefile not found at {labeled_shp_path}.")
        return None

    try:
        # reads orthophoto for aligment metadata
        with rasterio.open(orthophoto_path) as src:
            transform = src.transform
            crs = src.crs
            width = src.width
            height = src.height
            profile = src.profile # get profile to copy for output mask

        logging.info(f"orthophoto metadata: width={width}, height={height}, crs={crs}")

        # read labeled shapefile
        gdf = gpd.read_file(labeled_shp_path)
        
        # verify that shapefile CRS = orthophoto CRS (isr= 6991)
        if gdf.crs != crs:
            logging.warning(f"shapefile CRS ({gdf.crs}) does not match orthophoto CRS ({crs}). Attempting re-projection.")
            gdf = gdf.to_crs(crs)
            logging.info("shapefile re-projected to orthophoto CRS.")

        # prepare geometries,values for rasterization
        # makes (geometry, value) pairs. value = class index
        shapes_with_values = []
        # filter out invalid geometries or have missing class names
        valid_gdf = gdf.dropna(subset=[class_column])
        if valid_gdf.empty:
            logging.error(f"no valid geometries found in shapefile with column '{class_column}'. Check shapefile attributes.")
            return None

        for _, row in valid_gdf.iterrows():
            class_name = str(row[class_column]).strip().lower() # case insensitivity and no leading/trailing spaces
            if class_name in CLASS_TO_INDEX:
                class_id = CLASS_TO_INDEX[class_name]
                shapes_with_values.append((row.geometry, class_id))
            else:
                logging.warning(f"class '{class_name}' from shapefile not found in config.CLASS_TO_INDEX. Skipping geometry.")

        if not shapes_with_values:
            logging.error("no valid geometries with recognized class names found for rasterization.")
            return None

        # rasterize
        # the 'all_touched=True' argument can be used if pixels partially covered are needed
        # by a polygon to also be included. 'fill=0' = background pixels will be 0.
        # Class_to_index should be 0\'no data'.
        out_arr = rasterize(
            shapes=shapes_with_values,
            out_shape=(height, width),
            transform=transform,
            all_touched=True,
            fill=CLASS_TO_INDEX.get("others", 0),
            dtype=np.uint8)

        # saves rasterized mask, update profile for output mask
        profile.update(
            dtype=rasterio.uint8, # use uint8 for mask values 0-255
            count=1,             # single band output
            compress='lzw')       # LZW compression

        output_dir = os.path.dirname(output_mask_path)
        os.makedirs(output_dir, exist_ok=True)

        with rasterio.open(output_mask_path, 'w', **profile) as dst:
            dst.write(out_arr, 1)

        logging.info(f"✅ rasterized mask saved to {output_mask_path}")
        return output_mask_path

    except Exception as e:
        logging.error(f"error during rasterization: {e}", exc_info=True)
        return None