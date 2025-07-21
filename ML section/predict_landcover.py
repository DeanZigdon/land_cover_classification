# predict land cover
import numpy as np
import joblib
import os
from feature_extractor import extract_features
from data_utils import generate_adaptive_patches, load_geotiff
from config import MODEL_DIR


def predict_image(model_name, image_path, patch_size):
    model_path = os.path.join(MODEL_DIR, f"{model_name}.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    model = joblib.load(model_path)
    image, transform, crs = load_geotiff(image_path)
    patches = generate_adaptive_patches(image, patch_size)

    predictions = []
    coords = []

    for x, y, patch in patches:
        feats = extract_features(patch)
        label = model.predict([feats])[0]
        probas = model.predict_proba([feats])[0] if hasattr(model, "predict_proba") else None
        uncertainty = 1.0 - max(probas) if probas is not None else None
        predictions.append((x, y, label, uncertainty))
        coords.append((x, y))

    return predictions, transform, crs


if __name__ == "__main__":
    # For testing
    preds, transform, crs = predict_image("randomforest", "example.tif", patch_size=32)
    for x, y, label, u in preds[:10]:
        print(f"Patch at ({x},{y}) → {label} (uncertainty={u:.2f})")
