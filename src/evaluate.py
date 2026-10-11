import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import os
import json

def generate_evaluation_artifacts(y_true_cls, y_pred_cls, labels_map, output_dir="models"):
    """
    Computes confusion matrix and classification report metrics, returning structured dicts.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    unique_labels = sorted(list(labels_map.keys()))
    target_names = [labels_map[k] for k in unique_labels]

    cm = confusion_matrix(y_true_cls, y_pred_cls, labels=unique_labels)
    report = classification_report(y_true_cls, y_pred_cls, target_names=target_names, output_dict=True)

    cm_data = {
        'matrix': cm.tolist(),
        'labels': target_names
    }

    with open(os.path.join(output_dir, 'confusion_matrix.json'), 'w') as f:
        json.dump(cm_data, f, indent=2)

    with open(os.path.join(output_dir, 'classification_report.json'), 'w') as f:
        json.dump(report, f, indent=2)

    return cm_data, report
