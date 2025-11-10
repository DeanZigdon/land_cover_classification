# ML section/main.py
"""
main project file, initiates the landcover classification process via GUI for both evaluation mode and prediction mode
some files have local running options for testing
"""

import argparse
import sys
import logging
import os
import datetime
from config import LOG_DIR, DEFAULT_PATCH_SIZE, TIMESTAMP_FORMAT, OUTPUT_DIR, \
                     DEFAULT_MODEL_NAME, MODEL_PATH, \
                     MANUAL_LABEL_ORTHOPHOTO_PATH, MANUAL_LABEL_RASTERIZED_MASK_PATH, \
                     RASTERIZE_ORTHOPHOTO_PATH, RASTERIZE_SHAPEFILE_PATH, RASTERIZE_OUTPUT_MASK_PATH, \
                     SHAPEFILE_CLASS_COLUMN, SELECTED_BANDS_FOR_MODEL,PATCH_LIMITS
from predict_landcover import predict_image
from evaluation_metrics import calculate_evaluation_metrics, save_evaluation_report_to_csv,plot_and_save_confusion_matrix
from shapefile_exporter import export_predictions_to_shapefile
from model_trainer import load_and_prepare_training_data, train_model_from_patches
from rasterizer import rasterize_labels
from GUI import launch_gui
from data_utils import load_geotiff, generate_adaptive_patches, get_labels_at_patch_centroids
from feature_extractor import extract_features

def setup_logging():
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, "run.log")
    logging.basicConfig(
        filename=log_path,
        level=logging.DEBUG, # editbale: INFO\debug
        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info("logging initialized.")


def run_cli():
    parser = argparse.ArgumentParser(description="landcover classifier cli")
    #  'mode' argument for switching between 'predict', 'train', and 'evaluate'
    parser.add_argument("--mode", type=str, default="predict", choices=["predict", "train", "rasterize", "evaluate"],
                        help="operation mode: 'predict' for classification, 'train' for model training, 'rasterize' for converting shapefile to mask, 'evaluate' for model evaluation.")
    # arguments for prediction mode
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME,
                        help="name of the model to load (without .joblib)")
    parser.add_argument("--image", type=str, 
                        help="path to input geotiff image for prediction.")
    # arguments for training mode (can also be used for specific prediction images)
    parser.add_argument("--train_image", type=str, default=MANUAL_LABEL_ORTHOPHOTO_PATH,
                        help="path to input orthophoto for training (for 'train' mode).")
    parser.add_argument("--train_mask", type=str, default=MANUAL_LABEL_RASTERIZED_MASK_PATH,
                        help="path to input rasterized mask for training (for 'train' mode).")
    parser.add_argument("--patch_size", type=int, default=DEFAULT_PATCH_SIZE, help="patch size in pixels")
    parser.add_argument("--output", type=str, default="landcover", help="basename for output shapefile (for 'predict' mode)")
    # arguments for rasterize mode
    parser.add_argument("--raster_ortho", type=str, default=RASTERIZE_ORTHOPHOTO_PATH,
                        help="path to orthophoto image for rasterization")
    parser.add_argument("--raster_shp", type=str, default=RASTERIZE_SHAPEFILE_PATH,
                        help="path to labeled shapefile for rasterization")
    parser.add_argument("--raster_output", type=str, default=RASTERIZE_OUTPUT_MASK_PATH,
                        help="path where the output mask will be saved")
    parser.add_argument("--class_col", type=str, default=SHAPEFILE_CLASS_COLUMN,
                        help="column name in shapefile that contains class labels")
    # arguments for evaluation mode
    parser.add_argument("--gt_mask", type=str, help="path to ground truth rasterized mask for evaluation mode.")
    args = parser.parse_args()

    setup_logging()
    logging.info(f"starting pipeline in {args.mode} mode...")

    if args.mode == "predict":
        if not args.image:
            logging.error("input image path is required for 'predict' mode. use --image.")
            sys.exit(1)
        try:
            timestamp = datetime.datetime.now().strftime(TIMESTAMP_FORMAT)
            run_output_dir = os.path.join(OUTPUT_DIR, f"run_{timestamp}")
            os.makedirs(run_output_dir, exist_ok=True)
            logging.info(f"prediction output will be saved to: {run_output_dir}")

            predictions, transform, crs, csv_path = predict_image(args.model, args.image,args.patch_size, output_dir=run_output_dir)
            logging.info(f"predictions generated for {len(predictions)} patches.")
            export_predictions_to_shapefile(predictions, transform, crs, args.patch_size,base_output_dir=run_output_dir, output_name=args.output) 
            logging.info("shapefile export complete.")
            if csv_path:
                logging.info(f"predictions CSV also saved to: {csv_path}")
        except Exception as e:
            logging.error(f"error during classification: {e}", exc_info=True)
    
    elif args.mode == "train":
        if not args.train_image or not args.train_mask:
            logging.error("both --train_image and --train_mask paths are required for 'train' mode.")
            sys.exit(1)
        image_mask_pairs_for_training = [(args.train_image, args.train_mask)]

        try:
            logging.info("starting training pipeline...")
            X_train_data, y_train_data = load_and_prepare_training_data(image_mask_pairs=image_mask_pairs_for_training,patch_size=args.patch_size)
            trained_classifier = train_model_from_patches(X_train_data, y_train_data)
            logging.info("model training process completed.")
        except Exception as e:
            logging.error(f"error during training: {e}", exc_info=True)
            
    elif args.mode == "rasterize":
            logging.info("starting rasterization pipeline...")
            rasterized_path = rasterize_labels(
                orthophoto_path=args.raster_ortho,
                labeled_shp_path=args.raster_shp,
                output_mask_path=args.raster_output,
                class_column=args.class_col)
            if rasterized_path:
                logging.info(f"rasterization successful. Mask saved to: {rasterized_path}")
            else:
                logging.error("rasterization failed. Check logs for details.")
                sys.exit(1)

    elif args.mode == "evaluate":
        if not args.image or not args.gt_mask or not args.model:
            logging.error("for 'evaluate' mode, --image, --model, and --gt_mask are required.")
            sys.exit(1)
        try:
            logging.info("starting evaluation pipeline...")
            
            # run prediction to get predicted labels and patches
            # patches list to directed to the centroid of each prediction
            image_for_prediction, _, _, _ = load_geotiff(args.image)
            prediction_patches = generate_adaptive_patches(image_for_prediction, args.patch_size)
            # predict on the following patches
            predictions_with_feats, _, _, _ = predict_image(args.model, args.image, args.patch_size)
            y_pred = [p['predicted_class'] for p in predictions_with_feats]
            
            # loads ground truth mask (gt)
            logging.info(f"loading ground truth mask {args.gt_mask}...")
            gt_image, _, _, _ = load_geotiff(args.gt_mask, bands_to_read=[1])
            
            # gets ground truth labels from the mask at the centroid of each patch
            y_true = get_labels_at_patch_centroids(prediction_patches, gt_image)
            
            if len(y_true) != len(y_pred):
                logging.error(f"mismatch in number of samples. Ground truth has {len(y_true)} patches, but predictions have {len(y_pred)}. Cannot evaluate.")
                sys.exit(1)
                
            timestamp = datetime.datetime.now().strftime(TIMESTAMP_FORMAT)
            run_output_dir = os.path.join(OUTPUT_DIR, f"evaluation_report_{timestamp}") # diff output directory over mixing issues
            os.makedirs(run_output_dir, exist_ok=True)
            metrics = calculate_evaluation_metrics(y_true, y_pred, args.model)
            csv_path = save_evaluation_report_to_csv(metrics, run_output_dir, args.model)
            if csv_path:
                logging.info(f"evaluation report saved to: {csv_path}")
                print(f"✅ Evaluation report saved to: {csv_path}")
            else:
                logging.warning("⚠️ Evaluation report could not be saved.")
        except Exception as e:
            logging.error(f"error during evaluation: {e}", exc_info=True)
            sys.exit(1)

def run_gui():
    setup_logging()
    logging.info("launching gui...")
    gui_inputs = launch_gui()
    if not gui_inputs:
        logging.warning("no input received from gui. exiting.")
        sys.exit(0)

    mode = gui_inputs.get("mode")
    logging.info(f"inputs received from gui. starting pipeline in {mode} mode...")

    try:
        timestamp = datetime.datetime.now().strftime(TIMESTAMP_FORMAT)
        base_output_root = gui_inputs.get("output_dir", OUTPUT_DIR)
        
        # both modes
        if mode == "predict":
            run_output_dir = os.path.join(base_output_root, f"run_{timestamp}")
            os.makedirs(run_output_dir, exist_ok=True)
            logging.info(f"output will be saved to: {run_output_dir}")

            predictions, transform, crs, csv_path = predict_image(
                gui_inputs["model"],
                gui_inputs["image"],
                gui_inputs["patch_size"],
                bands_to_use=SELECTED_BANDS_FOR_MODEL,
                output_dir=run_output_dir
            )
            logging.info(f"predictions generated for {len(predictions)} patches.")

            tuple_predictions = [(int(p["x"]), int(p["y"]), int(p["predicted_class"]), float(p.get("confidence", -1.0)))for p in predictions]
            export_predictions_to_shapefile(
                tuple_predictions,
                transform,
                crs,
                gui_inputs["patch_size"],
                base_output_dir=run_output_dir,
                output_name=gui_inputs["output"])
            logging.info(f"shapefile exported to {os.path.join(run_output_dir, gui_inputs['output'] + '.shp')}.")
            if csv_path:
                logging.info(f"predictions CSV also saved to: {csv_path}")

        elif mode == "evaluate":
            run_output_dir = os.path.join(base_output_root, f"evaluation_report_{timestamp}")
            os.makedirs(run_output_dir, exist_ok=True)
            logging.info("starting evaluation pipeline...")
            image_for_prediction, _, _, _ = load_geotiff(gui_inputs["image"])
            prediction_patches = generate_adaptive_patches(image_for_prediction, gui_inputs["patch_size"])
            predictions_with_feats, _, _, _ = predict_image(gui_inputs["model"], gui_inputs["image"], gui_inputs["patch_size"])
            y_pred = [p['predicted_class'] for p in predictions_with_feats]
            
            logging.info(f"loading ground truth mask {gui_inputs['gt_mask']}...")
            gt_image, _, _, _ = load_geotiff(gui_inputs['gt_mask'], bands_to_read=[1])
            y_true = get_labels_at_patch_centroids(prediction_patches, gt_image)
            
            if len(y_true) != len(y_pred):
                logging.error(f"mismatch in number of samples. Ground truth has {len(y_true)} patches, but predictions have {len(y_pred)}. Cannot evaluate.")
                sys.exit(1)
            
            metrics = calculate_evaluation_metrics(y_true, y_pred, gui_inputs["model"])
            csv_path = save_evaluation_report_to_csv(metrics, run_output_dir, gui_inputs["model"])
            if csv_path:
                logging.info(f"evaluation report saved to: {csv_path}")
                print(f"✅ Evaluation report saved to: {csv_path}")
            else:
                logging.warning("⚠️ Evaluation report could not be saved.")
            
            class_labels = sorted(list(metrics['per_class_accuracy'].keys()))
            plot_and_save_confusion_matrix(metrics['confusion_matrix'], run_output_dir, gui_inputs['model'], class_labels)

    except Exception as e:
        logging.error(f"error during pipeline execution: {e}", exc_info=True)
        
def main():
    if len(sys.argv) == 1:
        if config.ENABLE_GUI:
            run_gui()
        else:
            logging.error("GUI is disabled in config and no CLI arguments provided. Exiting.")
            sys.exit(1)
    else:
        run_cli()

if __name__ == "__main__":
    import config
    main()
