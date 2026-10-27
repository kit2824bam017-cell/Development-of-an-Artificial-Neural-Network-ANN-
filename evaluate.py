"""
evaluate.py - Evaluation metrics and diagnostics for binary classification.
Computes Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix.
"""

from typing import Dict, Any, Union
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)


def evaluate_model_performance(
    model: nn.Module,
    X_test: Union[np.ndarray, torch.Tensor],
    y_test: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Run comprehensive statistical evaluation on test data.

    Args:
        model (nn.Module): Trained neural network model.
        X_test: Test features (NumPy array or PyTorch Tensor).
        y_test: True binary ground-truth labels.
        threshold (float): Decision threshold for classifying as class 1 (default 0.5).
        device (str): Compute device.

    Returns:
        Dict[str, Any]: Dictionary containing all quantitative metrics and ROC points.
    """
    dev = torch.device(device)
    model.eval()
    model.to(dev)

    if isinstance(X_test, np.ndarray):
        x_tensor = torch.tensor(X_test, dtype=torch.float32).to(dev)
        y_true = y_test.copy()
    else:
        x_tensor = X_test.float().to(dev)
        y_true = y_test.cpu().numpy().flatten()

    with torch.no_grad():
        logits = model(x_tensor)
        probabilities = torch.sigmoid(logits).cpu().numpy().flatten()

    predictions = (probabilities >= threshold).astype(int)

    acc = accuracy_score(y_true, predictions)
    prec = precision_score(y_true, predictions, zero_division=0)
    rec = recall_score(y_true, predictions, zero_division=0)
    f1 = f1_score(y_true, predictions, zero_division=0)
    auc = roc_auc_score(y_true, probabilities)
    cm = confusion_matrix(y_true, predictions)
    report = classification_report(y_true, predictions, target_names=["Moon 0", "Moon 1"])
    fpr, tpr, roc_thresholds = roc_curve(y_true, probabilities)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": auc,
        "confusion_matrix": cm,
        "classification_report": report,
        "y_true": y_true,
        "probabilities": probabilities,
        "predictions": predictions,
        "roc_curve": {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": roc_thresholds
        }
    }


def print_evaluation_summary(model_name: str, metrics: Dict[str, Any]):
    """Print a clean, structured console summary of the model metrics."""
    print("=" * 60)
    print(f"       PERFORMANCE REPORT: {model_name}")
    print("=" * 60)
    print(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print("-" * 60)
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])
    print("-" * 60)
    print("Classification Report:")
    print(metrics["classification_report"])
    print("=" * 60)
