# shp exporter
import shapefile
from shapely.geometry import box
import os
from config import OUTPUT_DIR, ACCESSIBILITY_MAP


def export_predictions_to_shapefile(predictions, transform, crs, patch_size, output_name="landcover"): 
    """
    Exports predicted patches as a polygon shapefile.

    predictions: list of (x, y, label, uncertainty)
    transform: rasterio Affine transform
    crs: rasterio CRS object
    patch_size: size of patches in pixels
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    shp_path = os.path.join(OUTPUT_DIR, f"{output_name}.shp")

    with shapefile.Writer(shp_path) as shp:
        shp.field("label", "C")
        shp.field("uncertainty", "N", decimal=3)
        shp.field("accessibility", "C")

        for x, y, label, uncertainty in predictions:
            x_geo, y_geo = transform * (x, y)
            x2_geo, y2_geo = transform * (x + patch_size, y + patch_size)

            poly = box(x_geo, y2_geo, x2_geo, y_geo)  # shapely: (minx, miny, maxx, maxy)
            shp.poly([list(poly.exterior.coords)])
            shp.record(label, float(uncertainty) if uncertainty is not None else -1.0,
                       ACCESSIBILITY_MAP.get(label, "inaccessible"))

    # Save projection
    with open(shp_path.replace(".shp", ".prj"), "w") as prj:
        prj.write(crs.to_wkt())

    print(f"Shapefile saved to: {shp_path}")
