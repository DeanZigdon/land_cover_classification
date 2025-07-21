# model_trainer
import numpy as np
import joblib
import os
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from config import MODEL_DIR, RANDOM_SEED, AVAILABLE_MODELS,TRAIN_TEST_SPLIT_RATIO

def train_model(X, y, model_type="RandomForest"):
    """
    Train and return a classifier.

    Parameters:
        X (np.ndarray): Features (N x D)
        y (np.ndarray): Labels (N,)
        model_type (str): One of AVAILABLE_MODELS

    Returns:
        Trained model
    """
    if model_type == "RandomForest":
        model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
    elif model_type == "SVM":
        model = SVC(probability=True, random_state=RANDOM_SEED)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.fit(X, y)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("\nModel Evaluation Report:")
    print(classification_report(y_test, y_pred))


def save_model(model, name):
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, f"{name}.joblib")
    joblib.dump(model, path)
    logging.info(f"Model saved to {path}")


def load_model(name):
    path = os.path.join(MODEL_DIR, f"{name}.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)
