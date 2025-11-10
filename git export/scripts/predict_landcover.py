# predict_landcover
"""
this script runs the land cover prediction process
it loads trained classification model and predicts the land cover class for input GeoTIFF image based on patches.
output: predictions.csv,summarry analysis.csv with extracted top patch features
"""
import numpy as np
import pandas as pd
import joblib
import os
from feature_extractor import extract_features
from data_utils import load_geotiff, mask_white_pixels
from config import MODEL_DIR, DEFAULT_MODEL_NAME, DEFAULT_PATCH_SIZE, SURFACE_CLASSES,EVAL_MAX_PATCHES,PREDICT_MAX_PATCHES,PATCH_LIMITS
import logging
from tqdm import tqdm # for some reason only works in that way
import tifffile as tiff
from data_utils import generate_adaptive_patches # for some reason it only works when seperated from the rest of data_utils imports
import csv
import datetime
import rasterio
from config import INDEX_TO_CLASS, CLASS_TO_INDEX, SELECTED_BANDS_FOR_MODEL
from collections import Counter
from evaluation_metrics import calculate_evaluation_metrics, save_evaluation_report_to_csv
from sklearn.metrics import precision_score, recall_score, f1_score


def predict_image(model_name, image_path, patch_size, bands_to_use=None, output_dir=None,  mode="predict"): #editable: mode predict\evaluate
    """
    loads a trained model and predicts land cover for an input image.

    parameters:
        model_name (str): model file name (without .joblib extension)
        image_path (str): input GeoTIFF image path
        patch_size (int): patch sizes in prediction pixels
        bands_to_use (list, optional): list of 1 based band indices to use from the image
                                        if None, uses all bands.
        output_dir (str, optional): output directory
        mode: predict\evaluation 
    returns:
        tuple: (predictions, transform, crs, csv_path)
                predictions: list of dictionaries, each with 'x', 'y', 'predicted_class', 'confidence'
                transform: affine transform of input image
                crs: coordinate system of input image
                csv_path: output CSV dir
    """
    model_path = os.path.join(MODEL_DIR, model_name + ".joblib")
    if not os.path.exists(model_path):
        logging.error(f"model not found at {model_path}. please train a model first or check path.")
        return [], None, None, None

    logging.info(f"loading model from {model_path}...")
    model = joblib.load(model_path)
    logging.info("model loaded.")

    logging.info(f"loading image {image_path} for prediction...")
    # pass bands_to_use to load_geotiff
    image, transform, crs, band_info = load_geotiff(image_path, bands_to_read=bands_to_use)
    
    if image is None:
        logging.error(f"Failed to load image {image_path}. Exiting prediction.")
        return [], None, None, None

    # verify image is always 3D (H, W, bands)
    if image.ndim == 2:  # single band
        image = image[:, :, np.newaxis]
    elif image.shape[2] != len(bands_to_use) if bands_to_use else image.shape[2]:
        logging.warning(f"Number of bands loaded ({image.shape[2]}) does not match bands_to_use ({len(bands_to_use) if bands_to_use else 'all'}).")

    # apply white pixel masking
    image = mask_white_pixels(image)
    logging.info(f"applied white pixel masking to image for prediction.")

    # get image dimensions safely
    img_height, img_width, num_bands_loaded = image.shape
    logging.info(f"image dimensions after band selection and masking: {img_height}x{img_width} with {num_bands_loaded} bands.")

    
    # decide ammount of patches to generate based on mode
    if mode == "predict":
            max_patches_to_use = PATCH_LIMITS["prediction"]["max_patches"]
    elif mode == "eval":
            max_patches_to_use = PATCH_LIMITS["evaluation"]["max_patches"]
    else:
            logging.warning(f"Unknown mode {mode}, defaulting to prediction max patches")
            max_patches_to_use = PATCH_LIMITS["prediction"]["max_patches"]

    logging.info(f"max_patches_to_use set to {max_patches_to_use} for mode '{mode}'")
    logging.info(f"image dimensions after band selection and masking: {img_height}x{img_width} with {num_bands_loaded} bands.")
    
    
    # log the number of features the model expects (from training)
    # assuming the model has  n_features_in_ attribute after fitting
    expected_features = getattr(model, 'n_features_in_', None)
    if expected_features is not None:
        logging.info(f"loaded model expects {expected_features} features per sample.")
    else:
        logging.warning("could not determine expected features from model. Ensure model was trained correctly.")

    image = mask_white_pixels(image)
    logging.info(f"applied white pixel masking to image for prediction.")

    # get image dimensions after potential band selection
    img_height, img_width, num_bands_loaded = image.shape
    # create mapping from numeric class label to string name
    class_map = {i: class_name for i, class_name in enumerate(SURFACE_CLASSES)}
    predictions = []
    features_list = []
    all_feats = []
    patch_coords = []
    
    total_patches = (img_height // patch_size) * (img_width // patch_size) # approximate count, works well
    logging.info(f"generating patches and predicting (approx. {total_patches} patches)...")
        
    patch_id_counter = 0 # patch ID counter (for csv + logging)
    confidence_map = np.zeros((img_height, img_width), dtype=np.float32) # empty array for confidence map

    # generate_adaptive_patches returns (x_start, y_start, img_patch)
    for x_start, y_start, img_patch in tqdm(generate_adaptive_patches(image, patch_size, max_patches=max_patches_to_use), desc="Extracting features"): 
        feats = extract_features(img_patch) # extract features from the patch
        # logging.debug(f"🧩 Patch at ({x_start},{y_start}) extracted {len(feats)} features: {feats}") # debugging feature numbers
        # CRITICAL CHECK: Ensure the number of features extracted matches the model expectation
        if expected_features is not None and len(feats) != expected_features:
            logging.error(f"feature mismatch for patch at ({x_start},{y_start}). Extracted {len(feats)} features, but model expects {expected_features}. Skipping patch.")
            continue
        all_feats.append(feats)
        patch_coords.append({'patch_id': patch_id_counter, 'x': x_start, 'y': y_start})
        patch_id_counter += 1

    logging.info("feature extraction complete. starting vectorized prediction...")
    if all_feats:
        all_feats_arr = np.array(all_feats)
        # perform prediction on all patches at once
        labels = model.predict(all_feats_arr)
        probas = model.predict_proba(all_feats_arr)
        confidence_map = np.zeros((img_height, img_width), dtype=np.float32) # empty array for confidence map

        # extract coordinates and other info for vectorized processing
        patch_x_coords = np.array([p['x'] for p in patch_coords])
        patch_y_coords = np.array([p['y'] for p in patch_coords])
        
        # vectorized confidence calculation:
        # get the confidence for each predicted class from the probas array
        if hasattr(model, 'classes_'):
            model_classes_array = np.array(model.classes_)
            class_indices = np.searchsorted(model_classes_array, labels) # find the indices of the predicted labels within the model classes
            confidence_values = probas[np.arange(len(probas)), class_indices] # uses the predicted labels indices to select the correct probabilities
        else:
            logging.warning("model does not have classes_ attribute, confidence cannot be calculated. assigning confidence=0.")
            confidence_values = np.zeros(len(labels))

        # fills the confidence map
        for i in tqdm(range(len(labels)), desc="Filling confidence map"):
            x_start = patch_x_coords[i]
            y_start = patch_y_coords[i]
            confidence_map[y_start:y_start+patch_size, x_start:x_start+patch_size] = confidence_values[i]

        # builds final predictions list iteratively
        predictions = []
        features_list = []
        for i in tqdm(range(len(labels)), desc="Processing predictions"):
            label = labels[i]
            proba = probas[i]
            label_name = class_map.get(label, "unknown")
            confidence = confidence_values[i]
            
            # editable: gets top 3 classes and their probabilities
            sorted_class_indices = np.argsort(proba)[::-1]
            top_classes = [
                {'class_name': class_map.get(model.classes_[idx], "unknown"), 'prob': proba[idx]}
                for idx in sorted_class_indices[:min(3, len(sorted_class_indices))]]

            # add unique patch ID and the top classes to the prediction dictionary
            pred_dict = {
                'patch_id': patch_coords[i]['patch_id'],
                'x': patch_coords[i]['x'],
                'y': patch_coords[i]['y'],
                'predicted_class': label,
                'predicted_class_name': label_name,
                'confidence': confidence,
                'top_classes': top_classes}
            predictions.append(pred_dict)
            features_list.append(all_feats[i])
    logging.info("prediction complete.")
    
    # save confidence map to geotiff file
    if output_dir:
        base_filename = os.path.basename(image_path).rsplit('.', 1)[0]
        confidence_path = os.path.join(output_dir, f"confidence_map_{base_filename}.tif")
        logging.info(f"saving confidence map to {confidence_path}...")
        with rasterio.open(
            confidence_path,
            'w',
            driver='GTiff',
            height=img_height,
            width=img_width,
            count=1,
            dtype=confidence_map.dtype,
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(confidence_map, 1)
        logging.info(f"✅ confidence map saved.")

        if 'EVAL_MODE' in os.environ and 'GROUND_TRUTH_RASTER' in os.environ:
            ground_truth_raster = os.environ['GROUND_TRUTH_RASTER']
            logging.info(f"entering evaluation mode. loading ground truth mask {ground_truth_raster}...")
            # using rasterio to load ground truth file and read only the first EVAL_MAX_PATCHES
            try:
                with rasterio.open(ground_truth_raster) as gt_src:
                    gt_height, gt_width = gt_src.shape
                    ground_truth_labels = []
                    # verify coordinates match predicted patches
                    # uses the same list(x,y) coords from 'predictions' list (ensures same spatially ordered patches)
                    logging.info(f"reading ground truth labels for the first {EVAL_MAX_PATCHES} patches...")
                    # empty list to store the true classes
                    true_labels_for_eval = []
                    # loop through all x, y coordinates from predictions to aquire corresponding ground truth label
                    # matching predictions for evaluationwith ground truth patch count
                    predictions_for_eval = predictions
                    for pred in predictions_for_eval:
                        # convert pixel coordinates to row, col
                        row, col = pred['y'], pred['x']
                        # read the value of each corresponding location
                        try:
                            # read single pixel at (row, col)
                            gt_value = gt_src.read(1)[row, col]
                            true_labels_for_eval.append(gt_value)
                        except IndexError:
                            logging.warning(f"coordinate ({row}, {col}) is out of bounds for ground truth raster. Skipping.")
                            continue
                    
                    # mapping numeric ground truth labels to class names for comparison and adds them to predictions_for_eval dictionary
                    for i, label in enumerate(true_labels_for_eval):
                        class_name = INDEX_TO_CLASS.get(label, "unknown")
                        predictions_for_eval[i]['true_class'] = label
                        predictions_for_eval[i]['true_class_name'] = class_name
                        
                    logging.info("starting evaluation...")
                    calculate_evaluation_metrics(predictions_for_eval, output_dir, model_name) # process evaluation metrics
                    logging.info("evaluation complete.")

            except Exception as e:
                logging.error(f"an error occurred during evaluation mode: {e}")

    # 2nd csv 
    csv_path = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        # create timestamped CSV
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"{model_name}_predictions_{timestamp}.csv"
        csv_path = os.path.join(output_dir, csv_filename)
        save_predictions_to_csv(predictions, features_list, transform, patch_size, model_name, csv_path)
        # create summary csv with aggregated class counts & percentages
        summary_csv_filename = f"{model_name}_summary_{timestamp}.csv"
        summary_csv_path = os.path.join(output_dir, summary_csv_filename)

        # aggregate predictions by class
        class_counts = {}
        total_predictions = len(predictions)
        for pred in predictions:
            cname = pred["predicted_class_name"]
            class_counts[cname] = class_counts.get(cname, 0) + 1

        # save summary to CSV
        with open(summary_csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["class_name", "count", "percentage"])
            for cname, count in sorted(class_counts.items(), key=lambda x: -x[1]):
                percentage = (count / total_predictions) * 100 if total_predictions > 0 else 0
                writer.writerow([cname, count, f"{percentage:.2f}"])

        #combined summary + evaluation CSV 
        combined_csv_filename = f"{model_name}_analysis_{timestamp}.csv"
        combined_csv_path = os.path.join(output_dir, combined_csv_filename)

        from collections import Counter
        counts = Counter([p['predicted_class_name'] for p in predictions])
        total_patches = len(predictions)

        # class distribution
        class_distribution = [
            (cls, count, count / total_patches * 100)
            for cls, count in counts.items()]

        # accuracy, kappa, F1: only if ground truth available (evaluation mode)
        accuracy = "N/A"
        kappa = "N/A"
        f1_macro = "N/A"
        if all("true_class_name" in p for p in predictions):
            from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score
            y_true = [p["true_class_name"] for p in predictions]
            y_pred = [p["predicted_class_name"] for p in predictions]
            accuracy = accuracy_score(y_true, y_pred)
            kappa = cohen_kappa_score(y_true, y_pred)
            f1_macro = f1_score(y_true, y_pred, average="macro")

        # mean confidence
        avg_confidence = (
            np.mean([p['confidence'] for p in predictions]) if predictions else 0.0)

        # saves combined CSV
        with open(combined_csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Class", "Count", "Percentage"])
            for cls, count, perc in class_distribution:
                writer.writerow([cls, count, f"{perc:.2f}%"])
            writer.writerow([])  # empty row
            writer.writerow(["Metric", "Value"])
            writer.writerow(["total_patches", total_patches])
            writer.writerow(["overall_accuracy", accuracy if accuracy != "N/A" else "N/A"])
            writer.writerow(["kappa_coefficient", kappa if kappa != "N/A" else "N/A"])
            writer.writerow(["f1_macro", f1_macro if f1_macro != "N/A" else "N/A"])
            writer.writerow(["average_confidence", f"{avg_confidence:.4f}"])
        logging.info(f"✅ combined analysis CSV saved to {combined_csv_path}")
    return predictions, transform, crs, csv_path


def save_predictions_to_csv(predictions, features_list, transform, patch_size, model_name, csv_path):
    """
    save predictions with extracted features to CSV
    predictions: list of dicts with 'x', 'y', 'predicted_class', 'confidence', 'top_classes'
    features_list: list of feature lists per patch, same order as predictions
    transform: affine transform (for geo coords)
    patch_size: patch size in pixels
    model_name: string
    csv_path: output path
    """
    pixel_width = transform.a  # pixel width in meters
    pixel_height = -transform.e  # pixel height (negative for north-up images)
    patch_area = (pixel_width * patch_size) * (pixel_height * patch_size)

    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['patch_id', 'x_pixel', 'y_pixel', 'x_geo', 'y_geo', 'patch_size_pixels', 'patch_area_m2', 
                      'predicted_class', 'confidence', 
                      'class_2', 'prob_2', 
                      'class_3', 'prob_3',
                      'model_name',
                      'haralick', 'homogeneity', 'entropy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for pred, feats in zip(predictions, features_list):
            x, y = pred['x'], pred['y']
            x_geo, y_geo = transform * (x, y)
            # assuming feature indices are: feats[0]=haralick, feats[1]=homogeneity, feats[2]=entropy as before
            haralick = feats[0] if len(feats) > 0 else None
            homogeneity = feats[1] if len(feats) > 1 else None
            entropy = feats[2] if len(feats) > 2 else None
            # extract top X (3) classes and their probabilities
            top_classes = pred.get('top_classes', [])
            # first class name and confidence are already in pred dict
            # editable: get 2nd & 3rd classes ( can add more)
            class_2 = top_classes[1]['class_name'] if len(top_classes) > 1 else None
            prob_2 = top_classes[1]['prob'] if len(top_classes) > 1 else None
            class_3 = top_classes[2]['class_name'] if len(top_classes) > 2 else None
            prob_3 = top_classes[2]['prob'] if len(top_classes) > 2 else None

            writer.writerow({
                'patch_id': pred['patch_id'],
                'x_pixel': x,
                'y_pixel': y,
                'x_geo': x_geo,
                'y_geo': y_geo,
                'patch_size_pixels': patch_size,
                'patch_area_m2': patch_area,
                'predicted_class': pred['predicted_class_name'],
                'confidence': pred['confidence'],
                'class_2': class_2,
                'prob_2': prob_2,
                'class_3': class_3,
                'prob_3': prob_3,
                'model_name': model_name,
                'haralick': haralick,
                'homogeneity': homogeneity,
                'entropy': entropy})
    print(f"✅ predictions CSV saved to {csv_path}")


if __name__ == "__main__":
    # for testing
    preds, transform, crs, csv_path = predict_image("randomforest", "example.tif", patch_size=32)
    # patches = list(generate_adaptive_patches(image, patch_size))
    # print(f"generated {len(patches)} patches for prediction")
    for pred in preds[:10]:
        print(f"Patch at ({pred['x']},{pred['y']}) -> {pred['predicted_class_name']} (confidence={pred['confidence']:.2f})")