# main.py
import argparse
import sys
import logging
import os
from config import LOG_DIR, DEFAULT_PATCH_SIZE
from predict_landcover import predict_image
from shapefile_exporter import export_predictions_to_shapefile

def setup_logging():
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, "run.log")
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info("Logging initialized.")

def run_cli():
    parser = argparse.ArgumentParser(description="Landcover Classifier CLI")
    parser.add_argument("--model", type=str, required=True, help="Name of the model to load (without .joblib)")
    parser.add_argument("--image", type=str, required=True, help="Path to input GeoTIFF image")
    parser.add_argument("--patch_size", type=int, default=DEFAULT_PATCH_SIZE, help="Patch size in pixels")
    parser.add_argument("--output", type=str, default="landcover", help="Basename for output shapefile")
    args = parser.parse_args()

    setup_logging()
    logging.info("Starting prediction pipeline...")

    try:
        predictions, transform, crs = predict_image(args.model, args.image, args.patch_size)
        logging.info(f"Predictions generated for {len(predictions)} patches.")
        export_predictions_to_shapefile(predictions, transform, crs, args.patch_size, output_name=args.output)
        logging.info("Shapefile export complete.")
    except Exception as e:
        logging.error(f"Error during classification: {e}", exc_info=True)

def main():
    if len(sys.argv) == 1:
        from GUI import launch_gui
        launch_gui()
    else:
        run_cli()

if __name__ == "__main__":
    main()
