# ML section/model_trainer.py
"""
this script manages the entire training pipeline for the land cover classifier 
it process orthophoto with ground truth masks to labeled feature,preparing training data by extracting spectral and textural
features from orthophoto patchesand normalizing them with standaradscaler preprocess,
it trains random forest on the features set.
it evaluates the model performances in a form of a classification report and confusion matrix.
a seperate function exists (predict_image) to load trained model and scaler and apply classification to a new geoTIFF image.

output: serialized model(.joblib),normalization scaler(.pkl),confusion matrix and model classification evaluation reports(csv), 
classified raster and confidence map
"""

import os
import numpy as np
import pandas as pd
import joblib
import rasterio
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report # for evaluation
from sklearn.preprocessing import StandardScaler
from scipy.stats import mode 
import logging
from feature_extractor import extract_features
from data_utils import load_geotiff, generate_adaptive_patches, mask_white_pixels
from config import (SURFACE_CLASSES, CLASS_TO_INDEX, INDEX_TO_CLASS,TRAIN_TEST_SPLIT_RATIO, DEFAULT_PATCH_SIZE, MODEL_PATH,
                    LABELLED_SHP_PATH, TIMESTAMP_FORMAT, OUTPUT_DIR, SELECTED_BANDS_FOR_MODEL,INCLUDE_RGB,INCLUDE_ENTROPY,
                    INCLUDE_NDVI,INCLUDE_HARALICK,FEATURE_LIST,MODEL_DIR,GROUND_TRUTH_RASTER)
import config 
from tqdm import tqdm
import datetime
import csv


def train_model_from_patches(X, y, model_save_path=None):
    """
    trains a RandomForestClassifier model using provided feature (X) and label (y) data
    splits data into training\testing sets, trains the model and saves it

    parameters:
        X (np.ndarray): feature matrix (samples,features)
        y (np.ndarray): label vector (samples)
        model_save_path (str, optional): full path to save the trained model
        defaults to config.MODEL_PATH if None

    returns:
        sklearn.ensemble.RandomForestClassifier: the trained classifier model
    """
    logging.info("starting model training...")

    # verify model_save_path is set
    if model_save_path is None:
        model_save_path = MODEL_PATH

    # debugging for evaluation mode, change raster sample size in config.py
    if GROUND_TRUTH_RASTER is not None and len(X) > GROUND_TRUTH_RASTER:
        logging.warning(f"Sampling {GROUND_TRUTH_RASTER} records from {len(X)} total for training/testing.")
        sample_indices = np.random.choice(len(X), GROUND_TRUTH_RASTER, replace=False)
        X = X[sample_indices]
        y = y[sample_indices]
        # split data into training\testing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=TRAIN_TEST_SPLIT_RATIO, random_state=42, stratify=y)

    # normalization by standardscaler
    logging.info("starting data normalization with StandardScaler...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    logging.info("normalization complete.")

    # save the normalization scaler
    scaler_save_path = os.path.join(MODEL_DIR, "scaler.pkl")
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(scaler, scaler_save_path)
    logging.info(f"✅ scaler saved to {scaler_save_path}")

    # feature counter for debugging
    print("🛠 Training data shape:", X_train.shape)
    print("🛠 Number of features per sample during training:", X_train.shape[1])
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1) # added n_jobs=-1 for parallelism
    logging.info(f"training random forest classifier with {X_train.shape[0]} samples and {X_train.shape[1]} features...")
    clf.fit(X_train, y_train)
    logging.info("training complete.")

    # saves trained model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(clf, model_save_path)
    logging.info(f"✅ model saved to {model_save_path}")

    # evaluate the test set and print reports
    evaluate_model(clf, X_test, y_test, "test")

    return clf


def evaluate_model(model, X_data, y_true, dataset_name="evaluation"):
    """
    evaluates the trained model on given dataset, prints performance metrics, export detailed report 
    and confusion matrix to CSV files

    parameters:
        model (sklearn.ensemble.RandomForestClassifier): the trained classifier
        X_data (np.ndarray): feature matrix for evaluation
        y_true (np.ndarray): true labels for evaluation
        dataset_name (str): name of evaluated dataset("validation", "test")
    """
    if X_data.size == 0 or y_true.size == 0:
        logging.warning(f"no data for {dataset_name} set. skipping evaluation.")
        return

    logging.info(f"\n--- evaluating model on {dataset_name} set ({X_data.shape[0]} pixels) ---")
    y_pred = model.predict(X_data)

    accuracy = accuracy_score(y_true, y_pred)
    logging.info(f"{dataset_name} accuracy: {accuracy:.4f}")

    # generates classification report as dict for simpler processing
    unique_labels = sorted(np.unique(np.concatenate((y_true, y_pred))))
    target_names = [INDEX_TO_CLASS.get(i, f'unknown_{i}') for i in unique_labels]

    report_dict = classification_report(y_true, y_pred, labels=unique_labels, target_names=target_names, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose()
    logging.info(f"{dataset_name} classification report:\n{classification_report(y_true, y_pred, labels=unique_labels, target_names=target_names, zero_division=0)}")
    
    # calculate per class pixel counts
    true_counts = pd.Series(y_true).value_counts().sort_index()
    pred_counts = pd.Series(y_pred).value_counts().sort_index()

    # map indices to class names for the new df columns
    true_counts.index = [INDEX_TO_CLASS.get(i, f'unknown_{i}') for i in true_counts.index]
    pred_counts.index = [INDEX_TO_CLASS.get(i, f'unknown_{i}') for i in pred_counts.index]

    # add pixel counts to the report df, verifying alignment by index (class name)
    report_df['true_pixels'] = true_counts
    report_df['predicted_pixels'] = pred_counts

    # add overall metrics as a new row
    report_df.loc['overall'] = {
        'precision': report_df.loc['weighted avg', 'precision'],
        'recall': report_df.loc['weighted avg', 'recall'],
        'f1-score': report_df.loc['weighted avg', 'f1-score'],
        'support': report_df.loc['weighted avg', 'support'],
        'true_pixels': y_true.size,
        'predicted_pixels': y_pred.size,}
    report_df.loc['overall', 'accuracy'] = accuracy
    
    # clean up the output by dropping redundant 'accuracy' and 'macro avg' rows
    report_df = report_df.drop(['accuracy', 'macro avg'])

    # CSV Export 
    csv_output_dir = os.path.join(OUTPUT_DIR, "evaluation")
    os.makedirs(csv_output_dir, exist_ok=True)
    
    timestamp = pd.Timestamp.now().strftime(TIMESTAMP_FORMAT)
    
    # export comprehensive report to CSV
    report_csv_path = os.path.join(csv_output_dir, f"evaluation_report_{dataset_name}_{timestamp}.csv")
    report_df.to_csv(report_csv_path)
    logging.info(f"📊 Detailed evaluation report saved to {report_csv_path}")

    # export confusion matrix to separate CSV
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    cm_csv_path = os.path.join(csv_output_dir, f"confusion_matrix_{dataset_name}_{timestamp}.csv")
    cm_df.to_csv(cm_csv_path)
    logging.info(f"📊 Confusion matrix saved to {cm_csv_path}")
    logging.info(f"\n--- illustrating uncertainty scores for {dataset_name} set (first 100 predictions) ---")
    y_proba = model.predict_proba(X_data)
    
    # create df for structured output
    metrics_df = pd.DataFrame(columns=[
        'pixel_idx', 'true_label', 'predicted_label_index', 'predicted_class_name', 'confidence', 'uncertainty'])
    # adds columns for each class probability based on existing classes in the model output
    for i in range(y_proba.shape[1]):
        class_name = INDEX_TO_CLASS.get(i, f'class_{i}')
        metrics_df[f'proba_{class_name}'] = 0.0

    for i in range(min(100, X_data.shape[0])): # editable: show first 'X'(100)
        true_label_idx = y_true[i]
        predicted_label_idx = y_pred[i]
        
        true_class_name = INDEX_TO_CLASS.get(true_label_idx, 'unknown')
        predicted_class_name = INDEX_TO_CLASS.get(predicted_label_idx, 'unknown')
        # confidence = probability of predicted class
        # verify predicted_label_idx is inside the bounds of y_proba columns
        confidence = y_proba[i, predicted_label_idx] if predicted_label_idx < y_proba.shape[1] else 0.0
        
        uncertainty = 1 - confidence  # opposite of confidence (%)

        row_data = {
            'pixel_idx': i,
            'true_label': true_class_name,
            'predicted_label_index': predicted_label_idx,
            'predicted_class_name': predicted_class_name,
            'confidence': confidence,
            'uncertainty': uncertainty}
        
        # populate class probabilities
        for class_idx in range(y_proba.shape[1]):
            class_name = INDEX_TO_CLASS.get(class_idx, f'class_{class_idx}')
            row_data[f'proba_{class_name}'] = y_proba[i, class_idx]

        metrics_df.loc[i] = row_data
    logging.info(metrics_df.to_string())

def predict_image(image_path, model_path=MODEL_PATH, bands_to_use=SELECTED_BANDS_FOR_MODEL, patch_size=DEFAULT_PATCH_SIZE):
    """
    loads GeoTIFF image, applies trained model to predict land cover classes
    and returns classified raster image

    input parameters:
        image_path (str): input GeoTIFF image
        model_path (str): input trained RandomForest model
        bands_to_use (list): list of 1 based band indices to extract from the geoTIFF
        patch_size (int): size of patches to process

    returns:
        tuple: (predictions, transform, crs)
               predictions (np.ndarray): 2D np array of classified pixel labels
               transform (affine.Affine): affine transform of output raster
               crs (rasterio.crs.CRS): raster CRS
    """
    logging.info(f"loading model from {model_path}...")
    try:
        model = joblib.load(model_path)
        logging.info("model loaded.")

        # scaler loading
        scaler = None
        scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
        if os.path.exists(scaler_path):
            try:
                scaler = joblib.load(scaler_path)
                logging.info("loaded scaler.pkl and will apply it to features before prediction.")
            except Exception as e:
                logging.warning(f"could not load scaler at {scaler_path}: {e}. proceeding without scaling.")

        # log the number of features the model expects
        if hasattr(model, 'n_features_in_'):
            logging.info(f"loaded model expects {model.n_features_in_} features per sample.")
        else:
            logging.warning("could not determine the number of features the loaded model expects.")

    except FileNotFoundError:
        logging.error(f"model file not found at {model_path}. please train the model first.")
        raise
    except Exception as e:
        logging.error(f"error loading model from {model_path}: {e}")
        raise

    logging.info(f"loading image {image_path} for prediction...")
    try:
        # load the image with selected bands and metadata
        image, transform, crs, band_info = load_geotiff(image_path, bands_to_read=bands_to_use)
    except Exception as e:
        logging.error(f"error loading image {image_path}: {e}")
        raise

    #  1st feature-count section: canonical FEATURE_LIST check (global)
    expected_features_from_extractor = len(FEATURE_LIST)
    logging.info(f"canonical FEATURE_LIST -> {FEATURE_LIST} (len={expected_features_from_extractor})")
    if config.INCLUDE_RGB: expected_features_from_extractor += 3
    if config.INCLUDE_HARALICK: expected_features_from_extractor += 2
    if config.INCLUDE_ENTROPY: expected_features_from_extractor += 1
    logging.info(f"feature_extractor is configured to produce {expected_features_from_extractor} features per patch.")
    if hasattr(model, 'n_features_in_') and model.n_features_in_ != expected_features_from_extractor:
        raise ValueError(f"Loaded model expects {model.n_features_in_} features but FEATURE_LIST length is {expected_features_from_extractor}. "
                         f"Refuse to run to prevent feature drift. Update FEATURE_LIST or retrain the model.")
    image = mask_white_pixels(image)
    logging.info(f"applied white pixel masking to image for prediction.")
    img_height, img_width, num_bands = image.shape
    logging.info(f"image dimensions after band selection and masking: {img_height}x{img_width} with {num_bands} bands.")

    # empty array for predictions and confidence map
    predictions = np.zeros((img_height, img_width), dtype=np.uint8)
    confidence_map = np.zeros((img_height, img_width), dtype=np.float32)
    # aquires a list of patch infos
    patches_info = list(generate_adaptive_patches(image, patch_size))
    logging.info(f"generating patches and predicting (approx. {len(patches_info)} patches)...")

    for x_start, y_start, patch in tqdm(patches_info, desc="predicting patches", total=len(patches_info)):
        # extract features for the patch
        features = extract_features(patch)

        # 2nd feature count section: per patch check + reshape + optional scaling 
        features_reshaped = features.reshape(1, -1)
        if features_reshaped.shape[1] != expected_features_from_extractor:
            logging.error(f"feature vector size mismatch at patch ({x_start},{y_start}). "
                          f"Extracted {features_reshaped.shape[1]} but expected {expected_features_from_extractor}. Skipping patch.")
            continue
        if scaler is not None:
            try:
                features_reshaped = scaler.transform(features_reshaped)
            except Exception as e:
                logging.warning(f"scaler.transform failed at patch ({x_start},{y_start}): {e}. using unscaled features.")

        # predict class and probability for the patch
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features_reshaped)[0]
            predicted_class = np.argmax(proba)
            confidence = proba[predicted_class]
        else: # troubleshooting
            predicted_class = model.predict(features_reshaped)[0]
            confidence = -1  # no confidence score available

        # fills the corresponding block in the predictions and confidence arrays
        predictions[y_start:y_start+patch_size, x_start:x_start+patch_size] = predicted_class
        confidence_map[y_start:y_start+patch_size, x_start:x_start+patch_size] = confidence

    #  saves confidence map to geoTIFF file 
    output_dir = os.path.dirname(image_path)
    confidence_path = os.path.join(output_dir, f"confidence_map_{os.path.basename(image_path)}")
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
    return predictions, transform, crs


# load and prepare training data from raster masks (modded for multiple inputs)
def load_and_prepare_training_data(image_mask_pairs, patch_size=DEFAULT_PATCH_SIZE, bands_to_use=None):
    """
    loads orthophotos and corresponding label masks, extracts features per patch,
    prepares X and Y for model training, masks purely white pixels from tiff file

    parameters:
        image_mask_pairs (list of tuples): list, each tuple is (image_path, mask_path)
        patch_size (int): patch pixel size
        bands_to_use (list, optional): list of 1 based band indices to use from the image
        if None = uses all bands

    returns:
        tuple: (X_features, y_labels) as np arrays
    """
    logging.info("loading and preparing training data...")
    
    all_X_features = []
    all_y_labels = []
    # calculates expected features from feature_extractor.py based on selected features (from config.py)
    expected_features_from_extractor = len(FEATURE_LIST) 
    logging.info(f"canonical FEATURE_LIST -> {FEATURE_LIST} (len={expected_features_from_extractor})")
    if config.INCLUDE_RGB: expected_features_from_extractor += 3
    if config.INCLUDE_HARALICK: expected_features_from_extractor += 2
    if config.INCLUDE_ENTROPY: expected_features_from_extractor += 1
    logging.info(f"feature_extractor is configured to produce {expected_features_from_extractor} features per patch.")

    for image_path, mask_path in tqdm(image_mask_pairs, desc="Processing image/mask pairs"):
        logging.info(f"processing image: {image_path} and mask: {mask_path}")

        if not os.path.exists(image_path):
            logging.error(f"image not found at {image_path}. skipping this pair.")
            continue
        if not os.path.exists(mask_path):
            logging.error(f"mask not found at {mask_path}. skipping this pair.")
            continue

        try:
            image, _, _, _ = load_geotiff(image_path, bands_to_read=bands_to_use)
            mask, _, _, _ = load_geotiff(mask_path) # mask will be 1 band, typically (H, W) or (H, W, 1)
        except Exception as e:
            logging.error(f"error loading geotiff for {image_path} or {mask_path}: {e}. skipping this pair.")
            continue
        image = mask_white_pixels(image) # white pixels masking 
        logging.info(f"applied white pixel masking to image: {image_path}")
        
        # verify mask is 2D (HxW) if it loaded as HxWx1
        if mask.ndim == 3 and mask.shape[2] == 1:
            mask = mask.squeeze()
        elif mask.ndim != 2:
            logging.error(f"mask image {mask_path} is not a single-band (H, W) or (H, W, 1) image. cannot process. skipping.")
            continue
        
        image_patches_info = generate_adaptive_patches(image, patch_size) # generates patches for image and mask
        current_image_X = []
        current_image_y = []
        
        # get tot number of patches for the tqdm progress bar
        img_height, img_width, _ = image.shape
        approx_total_patches = (img_height // patch_size) * (img_width // patch_size)

################
        # CRITICAL CHECK: Log the expected number of features from feature_extractor
        # logging the models training data
        expected_features_from_extractor = 0
        if config.INCLUDE_RGB: expected_features_from_extractor += 3 # rgb
        if config.INCLUDE_HARALICK: expected_features_from_extractor += 2 # contrast, homogeneity
        if config.INCLUDE_ENTROPY: expected_features_from_extractor += 1 # entropy
        
        if all_X_features and len(all_X_features[0]) != expected_features_from_extractor:
            logging.warning(f"initial feature vector size ({len(all_X_features[0])}) does not match expected from config ({expected_features_from_extractor}). Check feature_extractor.py and config flags.")
################

        for x_start, y_start, img_patch in tqdm(
            generate_adaptive_patches(image, patch_size),
            desc=f"Extracting patches from {os.path.basename(image_path)}",
            total=approx_total_patches,
            leave=False): # debugging, keeps the inner progress bar from cluttering the screen
        
            mask_patch = mask[y_start : y_start + patch_size, x_start : x_start + patch_size] # getss corresponding mask patch
            if mask_patch.size == 0:# verify mask patch is not empty (can happen in edges)
                continue
            
            # determine the most frequent class in the mask patch as its label
            try:
                # filter out ignore_index values if present (example: 255 for NoData)
                valid_mask_pixels = mask_patch[np.isin(mask_patch, list(INDEX_TO_CLASS.keys()))]
                if valid_mask_pixels.size == 0:
                    logging.debug(f"no valid pixels in mask patch at ({x_start},{y_start}). skipping.")
                    continue

                dominant_label, _ = mode(valid_mask_pixels.ravel(), keepdims=False)
                dominant_label = int(dominant_label) # convert to int
            except Exception as e:
                logging.warning(f"could not determine dominant label for patch at ({x_start},{y_start}): {e}. skipping patch.")
                continue # skip patches where mode cannot be determined

            # filter out "others" or unlabelled pixels if they are not part of training
            if dominant_label not in INDEX_TO_CLASS:
                logging.warning(f"dominant label {dominant_label} for patch at ({x_start},{y_start}) not in defined classes. skipping.")
                continue

            # extract features from image patch using the centralized function
            features = extract_features(img_patch)
            
            # basic check for feature vector size consistency (important for RandomForest model )
            # debugging: this catches issues if feature_extractor.py doesnt return consistent length
            if len(features) != expected_features_from_extractor:
                logging.error(f"feature vector size mismatch at patch ({x_start},{y_start}). Extracted {len(features)} features, but config expects {expected_features_from_extractor}. Skipping patch.")
                continue
            current_image_X.append(features)
            current_image_y.append(dominant_label)
        
        if current_image_X:
            all_X_features.extend(current_image_X)
            all_y_labels.extend(current_image_y)
        else:
            logging.warning(f"no valid patches extracted from {image_path}. check image content and mask labels.")

    final_X = np.array(all_X_features)
    final_y = np.array(all_y_labels)
    
    if final_X.size == 0 or final_y.size == 0:
        logging.error("no valid training samples extracted from any input pair. check input paths, images, masks, and patch size.")
        raise ValueError("no valid training samples extracted.")

    logging.info(f"extracted {len(final_X)} total samples for training.")
    return final_X, final_y


#  DRY RUN
# The dry_run function (test outside of main.py)is kept here for future tests

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("starting model_trainer script in standalone mode.")

    # in dry run mode, use the example paths from config.py for labeled orthophoto
    image_to_train_path = config.MANUAL_LABEL_ORTHOPHOTO_PATH
    mask_to_train_path = config.MANUAL_LABEL_RASTERIZED_MASK_PATH

    # Check if the required files exist for training with labeled data
    if not os.path.exists(image_to_train_path) or not os.path.exists(mask_to_train_path):
        logging.error(f"prerequisite: labeled orthophoto or rasterized mask not found for standalone training.")
        logging.error(f"expected: '{image_to_train_path}' and '{mask_to_train_path}'")
        logging.info("please ensure your orthophoto is at 'ML section/orthophoto/result.tif' and")
        logging.info("you have rasterized your labeled shapefile to 'ML section/orthophoto/training_labels_rasterized.tif'")

    else:
        try:
            # pass bands_to_use to the function call
            X_train_data, y_train_data = load_and_prepare_training_data(
                image_mask_pairs=[(image_to_train_path, mask_to_train_path)],
                patch_size=DEFAULT_PATCH_SIZE,
                bands_to_use=SELECTED_BANDS_FOR_MODEL)

            # train the model, model_save_path will default to config.MODEL_PATH
            trained_classifier = train_model_from_patches(X_train_data, y_train_data)
            logging.info("model training and saving complete with actual data.")

        except Exception as e:
            logging.error(f"an error occurred during training with manual data: {e}", exc_info=True)
            logging.info("training failed. please check logs for details.")
