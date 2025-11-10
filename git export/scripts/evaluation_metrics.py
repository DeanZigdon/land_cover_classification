# evaluation_metrics.py
"""
this script focuses on evaluation mode of the process
it calcualtes the evaluation metrics and the normalized metrics of the process
output: confusion matrix as heatmap image,evaluation metrics as csv
"""
import numpy as np
import csv
import os
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import logging
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report,confusion_matrix, cohen_kappa_score
from config import SURFACE_CLASSES, TIMESTAMP_FORMAT

def plot_and_save_confusion_matrix(cm, output_dir, model_name, class_labels):
    """
    plots the confusion matrix as a heatmap and saves it as a PNG file.
    """
    logging.info("creating and saving confusion matrix heatmap...")
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f"Confusion Matrix for {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    
    timestamp = datetime.datetime.now().strftime(TIMESTAMP_FORMAT)
    heatmap_path = os.path.join(output_dir, f"{model_name}_confusion_matrix_heatmap_{timestamp}.png")
    
    try:
        plt.savefig(heatmap_path, bbox_inches='tight')
        logging.info(f"✅ Confusion matrix heatmap saved to {heatmap_path}")
        return heatmap_path
    except Exception as e:
        logging.error(f"Error saving confusion matrix heatmap: {e}", exc_info=True)
        return None

def plot_and_save_normalized_confusion_matrix(cm, output_dir, model_name, class_labels):
    """
    plots the row normalized confusion matrix (in %) and saves it as a PNG file
    """
    logging.info("creating and saving normalized confusion matrix heatmap...")
    with np.errstate(invalid='ignore', divide='ignore'):
        cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_normalized = np.nan_to_num(cm_normalized)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f"Normalized Confusion Matrix for {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    timestamp = datetime.datetime.now().strftime(TIMESTAMP_FORMAT)
    heatmap_path = os.path.join(output_dir, f"{model_name}_confusion_matrix_heatmap_normalized_{timestamp}.png")

    try:
        plt.savefig(heatmap_path, bbox_inches='tight')
        logging.info(f"✅ Normalized confusion matrix heatmap saved to {heatmap_path}")
        return heatmap_path
    except Exception as e:
        logging.error(f"Error saving normalized confusion matrix heatmap: {e}", exc_info=True)
        return None


def calculate_evaluation_metrics(y_true, y_pred, model_name):
    """
    calculate key evaluation metrics for a landcover classification

    parameters:
        y_true (list or np.array): true class labels (numeric)
        y_pred (list or np.array): predicted class labels (numeric)
        model_name (str): name of evaluated model
    returns:
        dict: dictionary containing calculated metrics
    """
    logging.info("calculating evaluation metrics...")
    
    # get a list of all unique class labels to verify all classes are present
    # combining true and predicted labels provides a complete set
    labels = sorted(list(set(y_true) | set(y_pred)))
    class_names = [
        SURFACE_CLASSES[label] if label < len(SURFACE_CLASSES) else f"Class_{label}"
        for label in labels]  # class names aligned to `labels` order
    # verify ndarray views (non destructive) for counting
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    # calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # IoU per class
    intersection = np.diag(cm)
    ground_truth_set = cm.sum(axis=1)
    predicted_set = cm.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    iou_per_class = intersection / union
    overall_accuracy = np.trace(cm) / np.sum(cm) # calculate overall accuracy
    per_class_accuracy = {} # calculate per class accuracy
    for i, label in enumerate(labels):
        true_positives = cm[i, i]
        total_true = np.sum(cm[i, :])
        accuracy = true_positives / total_true if total_true > 0 else 0
        class_name = class_names[i]
        per_class_accuracy[class_name] = accuracy  
    # per class precision, recall, F1 (aligned to `labels`)
    precisions = precision_score(y_true, y_pred, average=None, labels=labels, zero_division=0) # how many instances of predicted positive instances were true positive
    recalls    = recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0) # recall = how many of the actual positive cases were correctly identified
    f1s        = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0) # f1 = combines percision and recall
    
     # macro & weighted averages
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro    = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro        = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # weighted averages
    precision_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall_weighted    = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_weighted        = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    per_class_precision = dict(zip(class_names, precisions))
    per_class_recall    = dict(zip(class_names, recalls))
    per_class_f1        = dict(zip(class_names, f1s))
    
    # class frequencies (ground truth) aligned to `labels`
    class_frequencies = dict(zip(class_names, [int(np.sum(y_true_arr == lbl)) for lbl in labels]))
    
    # normalized confusion matrix (row wise)
    with np.errstate(invalid='ignore', divide='ignore'):
        cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_normalized = np.nan_to_num(cm_normalized)
    
    # calculate Kappa coefficient
    kappa_coefficient = cohen_kappa_score(y_true, y_pred)
    
    # per class IoU
    iou_per_class = intersection / union
    per_class_iou = dict(zip(class_names, iou_per_class))

    logging.info("metrics calculation complete.")

    # report buildup
    metrics = {
        'model_name': model_name,
        'total_samples': len(y_true),
        'labels': labels,                       # numeric labels (ordered)
        'class_labels': class_names,            # readable class names (same order)
        'confusion_matrix': cm,                 # absolute
        'confusion_matrix_normalized': cm_normalized, # normalized rows

        'overall_accuracy': overall_accuracy,
        'kappa_coefficient': kappa_coefficient,
        
        # macro/weighted scores
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,

        'per_class_accuracy': per_class_accuracy,
        'per_class_precision': per_class_precision,
        'per_class_recall': per_class_recall,
        'per_class_f1': per_class_f1,
        'per_class_iou': per_class_iou, 
        'class_frequencies': class_frequencies}
    return metrics

def save_evaluation_report_to_csv(metrics, output_dir, model_name):
    """
    saves evaluation metrics to timestamped CSV file

    parameters:
        metrics (dict): the dictionary containing the calculated metrics
        output_dir (str): the directory to save the output file
        model_name (str): the name of the model for the output filename
        
    returns:
        str: the path to the saved CSV file or None if error occurred
    """
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime(TIMESTAMP_FORMAT)
        csv_path = os.path.join(output_dir, f"{model_name}_evaluation_report_{timestamp}.csv")

        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # overall metrics
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Model Name', metrics['model_name']])
            writer.writerow(['Total Samples', metrics['total_samples']])
            writer.writerow(['Overall Accuracy', f"{metrics['overall_accuracy']:.4f}"])
            writer.writerow(['Kappa Coefficient', f"{metrics['kappa_coefficient']:.4f}"])
            writer.writerow(["precision_macro", f"{metrics['precision_macro']:.4f}"])
            writer.writerow(["recall_macro", f"{metrics['recall_macro']:.4f}"])
            writer.writerow(["f1_macro", f"{metrics['f1_macro']:.4f}"])
            writer.writerow(["precision_weighted", f"{metrics['precision_weighted']:.4f}"])
            writer.writerow(["recall_weighted", f"{metrics['recall_weighted']:.4f}"])
            writer.writerow(["f1_weighted", f"{metrics['f1_weighted']:.4f}"])
            writer.writerow([]) # Blank row for separation 

            # per class metrics
            writer.writerow(['Per-Class Metrics'])
            writer.writerow(['Class', 'Precision', 'Recall', 'F1', 'Accuracy', 'IoU', 'Frequency'])

            for cname in metrics['class_labels']:
                writer.writerow([
                    cname,
                    f"{metrics['per_class_precision'][cname]:.4f}",
                    f"{metrics['per_class_recall'][cname]:.4f}",
                    f"{metrics['per_class_f1'][cname]:.4f}",
                    f"{metrics['per_class_accuracy'][cname]:.4f}",
                    f"{metrics['per_class_iou'][cname]:.4f}", # access IoU from the dictionary
                    metrics['class_frequencies'][cname]])
            writer.writerow([]) # blank row for separation
            
            # confusion matrix (absolute counts)
            writer.writerow(['Confusion Matrix (rows=True Label, columns=Predicted Label)'])
            writer.writerow([''] + metrics['class_labels'])
            cm = metrics['confusion_matrix']
            for i, true_class in enumerate(metrics['class_labels']):
                row = [true_class] + [int(x) for x in cm[i, :]]
                writer.writerow(row)
            writer.writerow([])  # blank row for separation

            # normalized confusion matrix (row wise percentages)
            writer.writerow(['Normalized Confusion Matrix (row-wise)'])
            writer.writerow([''] + metrics['class_labels'])
            cmn = metrics['confusion_matrix_normalized']
            for i, true_class in enumerate(metrics['class_labels']):
                row = [true_class] + [f"{x:.4f}" for x in cmn[i, :]]
                writer.writerow(row)
        
        logging.info(f"✅ Evaluation report saved to {csv_path}")
        return csv_path

    except Exception as e:
        logging.error(f"Error saving evaluation report: {e}", exc_info=True)
        return None
