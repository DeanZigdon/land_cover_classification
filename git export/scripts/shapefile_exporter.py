# shp exporter
"""
this script exports the predicted patches of the project as shp file (polygon)
"""
import shapefile
from shapely.geometry import box
import os
from config import ACCESSIBILITY_MAP
import logging

def export_predictions_to_shapefile(predictions, transform, crs, patch_size, base_output_dir, output_name="landcover"): 
    """
    exports predicted patches as polygon shapefile

    predictions: list of (x, y, label, uncertainty)
    transform: rasterio affine transform
    crs: rasterio CRS object
    patch_size: size of patches in pixels
    base_output_dir: base saving directory for the output (timestamped)
    output_name: basename for output shapefile ("landcover")
    """
    # makedir is done in 'main.py' due to script issues
    shp_path = os.path.join(base_output_dir, f"{output_name}.shp") # use base_output_dir

    with shapefile.Writer(shp_path) as shp:
        shp.field("label", "C")
        shp.field("uncertainty", "N", decimal=3)
        shp.field("accessibility", "C")
        for x, y, label, uncertainty in predictions:
            x = int(x)
            y = int(y)
            logging.debug(f"transforming coordinates: x={x}, y={y}, type={type(x)}, {type(y)}")

            x_geo, y_geo = transform * (x, y)
            x2_geo, y2_geo = transform * (x + patch_size, y + patch_size)

            poly = box(x_geo, y2_geo, x2_geo, y_geo)
            shp.poly([list(poly.exterior.coords)])
            shp.record(
                label,
                float(uncertainty) if uncertainty is not None else -1.0,
                ACCESSIBILITY_MAP.get(label, "inaccessible"))

    # save projection
    with open(shp_path.replace(".shp", ".prj"), "w") as prj:
        prj.write(crs.to_wkt())

    print(f"shapefile saved to: {shp_path}")