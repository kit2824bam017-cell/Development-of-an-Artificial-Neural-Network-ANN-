"""
visualize.py - Publication-grade plotting utilities for the make_moons binary
classification artificial neural network.
"""

from typing import Dict, Any, Optional
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn


# Set a clean, modern style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def plot_raw_dataset(
    X: np.ndarray,
    y: np.ndarray,
    title: str = "Scikit-Learn make_moons Synthetic Dataset",
    save_path: Optional[str] = None
):
    """Plot the 2D crescent moons dataset with class labels."""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # Class 0: Moon 0
    ax.scatter(
        X[y == 0, 0], X[y == 0, 1],
        c="#3b82f6", label="Class 0 (Upper Moon)",
        alpha=0.85, edgecolors="#1e3a8a", s=50, linewidths=0.6
    )
    # Class 1: Moon 1
    ax.scatter(
        X[y == 1, 0], X[y == 1, 1],
        c="#f43f5e", label="Class 1 (Lower Moon)",
        alpha=0.85, edgecolors="#881337", s=50, linewidths=0.6
    )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Feature $X_1$", fontsize=12)
    ax.set_ylabel("Feature $X_2$", fontsize=12)
    ax.legend(frameon=True, facecolor="white", edgecolor="#e2e8f0", loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[+] Saved dataset plot to: {save_path}")
    plt.close()


def plot_learning_curves(
    history: Dict[str, Any],
    title: str = "ANN Training and Validation Diagnostics",
    save_path: Optional[str] = None
):
    """Plot training and validation Loss and Accuracy across epochs."""
    epochs = range(1, len(history["train_loss"]) + 1)
    best_epoch = history.get("best_epoch", 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # 1. Loss Curve
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#2563eb", linewidth=2.0)
    ax1.plot(epochs, history["val_loss"], label="Val Loss", color="#ea580c", linewidth=2.0, linestyle="--")
    ax1.axvline(best_epoch, color="#10b981", linestyle=":", label=f"Best Epoch ({best_epoch})", linewidth=1.5)
    ax1.set_title("Binary Cross-Entropy Loss vs. Epoch", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss (BCE)", fontsize=11)
    ax1.legend(frameon=True, facecolor="white")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # 2. Accuracy Curve
    ax2.plot(epochs, history["train_acc"], label="Train Accuracy", color="#2563eb", linewidth=2.0)
    ax2.plot(epochs, history["val_acc"], label="Val Accuracy", color="#ea580c", linewidth=2.0, linestyle="--")
    ax2.axvline(best_epoch, color="#10b981", linestyle=":", label=f"Best Epoch ({best_epoch})", linewidth=1.5)
    ax2.set_title("Classification Accuracy vs. Epoch", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy", fontsize=11)
    ax2.set_ylim([0.45, 1.02])
    ax2.legend(frameon=True, facecolor="white")
    ax2.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[+] Saved learning curves to: {save_path}")
    plt.close()


def plot_comparative_decision_boundaries(
    linear_model: nn.Module,
    ann_model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    linear_acc: float,
    ann_acc: float,
    save_path: Optional[str] = None
):
    """
    Generate side-by-side decision boundary plots comparing the Linear Perceptron
    vs. the Deep Non-Linear Artificial Neural Network on test data.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    # Generate dense grid across normalized feature space
    x_min, x_max = X_test[:, 0].min() - 0.6, X_test[:, 0].max() + 0.6
    y_min, y_max = X_test[:, 1].min() - 0.6, X_test[:, 1].max() + 0.6
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.tensor(grid_points, dtype=torch.float32)

    models_info = [
        (linear_model, axes[0], f"Linear Classifier (Logistic)\nTest Accuracy: {linear_acc * 100:.2f}%"),
        (ann_model, axes[1], f"Deep Neural Network (3 Hidden Layers)\nTest Accuracy: {ann_acc * 100:.2f}%")
    ]

    for model, ax, title in models_info:
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(grid_tensor)).numpy().reshape(xx.shape)

        # Plot continuous probability background
        contour = ax.contourf(xx, yy, probs, levels=25, cmap="RdBu_r", alpha=0.65)
        # Decision boundary line at p = 0.5
        ax.contour(xx, yy, probs, levels=[0.5], colors="#0f172a", linewidths=2.2, linestyles="-")

        # Overlay actual test data points
        ax.scatter(
            X_test[y_test == 0, 0], X_test[y_test == 0, 1],
            c="#1d4ed8", edgecolors="white", s=45, linewidth=0.8, label="Class 0 (Test)"
        )
        ax.scatter(
            X_test[y_test == 1, 0], X_test[y_test == 1, 1],
            c="#be123c", edgecolors="white", s=45, linewidth=0.8, label="Class 1 (Test)"
        )

        ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Normalized Feature $X_1$", fontsize=11)
        ax.set_ylabel("Normalized Feature $X_2$", fontsize=11)
        ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)
        ax.grid(True, linestyle=":", alpha=0.4)

    # Add shared colorbar
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(contour, cax=cbar_ax)
    cbar.set_label("Predicted Probability $P(Y=1|X)$", fontsize=11)

    fig.suptitle("Comparison: Linear Boundary vs. Deep ANN Non-Linear Decision Boundary",
                 fontsize=14, fontweight="bold", y=0.98)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[+] Saved comparative decision boundary plot to: {save_path}")
    plt.close()


def plot_diagnostics(
    ann_metrics: Dict[str, Any],
    save_path: Optional[str] = None
):
    """Plot Confusion Matrix and ROC Curve for the Deep ANN."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # 1. Confusion Matrix Heatmap
    cm = ann_metrics["confusion_matrix"]
    cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

    labels = np.array([[f"{cm[0, 0]}\n({cm_percent[0, 0]:.1f}%)", f"{cm[0, 1]}\n({cm_percent[0, 1]:.1f}%)"],
                       [f"{cm[1, 0]}\n({cm_percent[1, 0]:.1f}%)", f"{cm[1, 1]}\n({cm_percent[1, 1]:.1f}%)"]])

    sns.heatmap(
        cm, annot=labels, fmt="", cmap="Blues", cbar=True, ax=ax1,
        xticklabels=["Pred Moon 0", "Pred Moon 1"],
        yticklabels=["True Moon 0", "True Moon 1"],
        linewidths=1.0, linecolor="white", annot_kws={"fontsize": 12, "fontweight": "bold"}
    )
    ax1.set_title("Confusion Matrix (Test Set)", fontsize=12, fontweight="bold", pad=12)
    ax1.set_ylabel("True Class", fontsize=11)
    ax1.set_xlabel("Predicted Class", fontsize=11)

    # 2. ROC Curve
    fpr = ann_metrics["roc_curve"]["fpr"]
    tpr = ann_metrics["roc_curve"]["tpr"]
    auc = ann_metrics["roc_auc"]

    ax2.plot(fpr, tpr, color="#2563eb", linewidth=2.5, label=f"Deep ANN (AUC = {auc:.4f})")
    ax2.plot([0, 1], [0, 1], color="#94a3b8", linestyle="--", linewidth=1.5, label="Random Guess (AUC = 0.5000)")
    ax2.fill_between(fpr, tpr, alpha=0.15, color="#2563eb")
    ax2.set_xlim([-0.02, 1.02])
    ax2.set_ylim([-0.02, 1.05])
    ax2.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=12, fontweight="bold", pad=12)
    ax2.set_xlabel("False Positive Rate (FPR)", fontsize=11)
    ax2.set_ylabel("True Positive Rate (TPR)", fontsize=11)
    ax2.legend(loc="lower right", frameon=True, facecolor="white")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[+] Saved diagnostics plot to: {save_path}")
    plt.close()
