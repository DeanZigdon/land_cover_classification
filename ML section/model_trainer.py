import os
import numpy as np
import matplotlib.pyplot as plt
import joblib
import tifffile as tiff
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from skimage.feature.texture import graycomatrix, graycoprops 
from skimage.color import rgb2gray
from skimage.util import view_as_blocks
from config import (SURFACE_CLASSES,CLASS_TO_INDEX,TRAIN_TEST_SPLIT_RATIO,DEFAULT_PATCH_SIZE,MODEL_PATH)

# ---------- Feature Extraction ----------
def extract_features(patch):
    # Simple RGB means
    means = patch.reshape(-1, 3).mean(axis=0)

    # Haralick texture on grayscale
    gray = rgb2gray(patch)
    gray = (gray * 255).astype(np.uint8)
    glcm = graycomatrix(gray, [1], [0], 256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    entropy = -np.sum(glcm * np.log2(glcm + 1e-10))

    return np.concatenate([means, [contrast, entropy]])

# ---------- Real Model Trainer ----------
def train_model_from_patches(X, y, model_name="randomforest.joblib"):
    clf = RandomForestClassifier(n_estimators=100, random_state=42) #  RETURN WHEN DRY RUN IS FINISHED, after "42" :stratify=y     
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=TRAIN_TEST_SPLIT_RATIO, random_state=42
    )

    clf.fit(X_train, y_train)
    os.makedirs("model_weights", exist_ok=True)
    model_path = os.path.join("model_weights", model_name)
    joblib.dump(clf, model_path)
    print(f"✅ Model saved to {model_path}")
    return clf

# ---------- DRY RUN ----------
def dry_run():
    print("\n=== DRY RUN: Simulated Orthophoto Classification ===")

    # Create dummy RGB image
    img_size = 128
    patch_size = DEFAULT_PATCH_SIZE
    image = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    labels = []

    for i in range(0, img_size, patch_size):
        for j in range(0, img_size, patch_size):
            class_idx = np.random.randint(len(SURFACE_CLASSES))
            color = np.random.randint(50, 200, size=3)
            image[i:i+patch_size, j:j+patch_size] = color
            labels.append(SURFACE_CLASSES[class_idx])

    tiff.imwrite("dry_run_image.tif", image)
    print("📷 Generated dry_run_image.tif")

    # Extract patches and features
    patches = view_as_blocks(image, block_shape=(patch_size, patch_size, 3))
    nrows, ncols = patches.shape[:2]
    X, y, coords = [], [], []

    idx = 0
    for r in range(nrows):
        for c in range(ncols):
            patch = patches[r, c, 0]
            X.append(extract_features(patch))
            y.append(CLASS_TO_INDEX[labels[idx]])
            center_x = c * patch_size + patch_size // 2
            center_y = r * patch_size + patch_size // 2
            coords.append((center_x, center_y))
            idx += 1

    X = np.array(X)
    y = np.array(y)

    # Train and save model
    clf = train_model_from_patches(X, y, model_name="randomforest_dryrun.joblib")

    # Predict and export CSV
    probs = clf.predict_proba(X)
    preds = clf.predict(X)
    uncertainty = 1 - np.max(probs, axis=1)

    csv_rows = []
    for i in range(len(X)):
        row = {
            "x_center": coords[i][0],
            "y_center": coords[i][1],
            "predicted": SURFACE_CLASSES[preds[i]],
            "true_label": SURFACE_CLASSES[y[i]],
            "uncertainty": round(uncertainty[i], 4),
        }
        csv_rows.append(row)

    df = pd.DataFrame(csv_rows)
    df.to_csv("dryrun_patch_predictions.csv", index=False)
    print("📄 Exported dryrun_patch_predictions.csv")

    # Preview image
    preview = image.copy()
    for i, (x, y) in enumerate(coords):
        color = plt.cm.get_cmap("tab10")(preds[i] % 10)
        rgb = tuple(int(255 * c) for c in color[:3])
        cv = slice(y - patch_size//2, y + patch_size//2)
        ch = slice(x - patch_size//2, x + patch_size//2)
        preview[cv, ch] = rgb

    plt.imsave("dryrun_preview.png", preview)
    print("🖼️ Exported dryrun_preview.png")

    print("✅ DRY RUN COMPLETE\n")

# ---------- Script Entry ----------
if __name__ == "__main__":
    dry_run()
