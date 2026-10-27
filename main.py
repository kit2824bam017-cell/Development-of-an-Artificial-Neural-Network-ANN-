"""
main.py - End-to-end pipeline for developing, training, evaluating, and visualizing
an Artificial Neural Network (ANN) for binary classification on the make_moons dataset.
"""

import os
import json
import numpy as np
import torch

from dataset import generate_moons_data, prepare_data_splits, create_dataloaders
from model import LinearClassifier, DeepMoonANN
from train import train_model
from evaluate import evaluate_model_performance, print_evaluation_summary
from visualize import (
    plot_raw_dataset,
    plot_learning_curves,
    plot_comparative_decision_boundaries,
    plot_diagnostics
)


def run_pipeline():
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print(" ARTIFICIAL NEURAL NETWORK (ANN) BINARY CLASSIFICATION PIPELINE")
    print("                 Dataset: Scikit-learn make_moons               ")
    print("=" * 70)

    # 1. Dataset Generation
    n_samples = 1500
    noise = 0.20
    random_seed = 42

    print(f"\n[1/6] Generating synthetic make_moons data...")
    print(f"      - Total samples: {n_samples}")
    print(f"      - Noise level: {noise}")
    print(f"      - Random seed: {random_seed}")
    X_raw, y_raw = generate_moons_data(n_samples=n_samples, noise=noise, random_state=random_seed)

    # Save raw data visualization
    raw_plot_path = os.path.join(output_dir, "01_dataset_distribution.png")
    plot_raw_dataset(X_raw, y_raw, save_path=raw_plot_path)

    # 2. Split and Scale Data
    print(f"\n[2/6] Partitioning data (70% Train, 15% Validation, 15% Test) and scaling...")
    splits = prepare_data_splits(X_raw, y_raw, test_size=0.15, val_size=0.15, random_state=random_seed)
    train_loader, val_loader, test_loader = create_dataloaders(splits, batch_size=32)

    print(f"      - Training set:   {len(splits['X_train'])} samples")
    print(f"      - Validation set: {len(splits['X_val'])} samples")
    print(f"      - Testing set:    {len(splits['X_test'])} samples")

    # 3. Train Baseline Linear Model (Single Perceptron / Logistic)
    print(f"\n[3/6] Training Baseline Linear Model (Single Perceptron)...")
    linear_model = LinearClassifier(in_features=2)
    trained_linear, linear_hist = train_model(
        linear_model,
        train_loader,
        val_loader,
        epochs=80,
        lr=0.02,
        patience=15,
        device="cpu"
    )
    linear_metrics = evaluate_model_performance(trained_linear, splits["X_test"], splits["y_test"])
    print_evaluation_summary("Baseline Linear Model (Perceptron)", linear_metrics)

    # 4. Build and Train Deep Artificial Neural Network (ANN)
    print(f"\n[4/6] Initializing and Training Deep Artificial Neural Network (ANN)...")
    ann_model = DeepMoonANN(
        in_features=2,
        hidden_dims=[32, 16, 8],
        dropout_rate=0.1,
        activation="leaky_relu"
    )
    print(f"      - Architecture: Input(2) -> FC(32)+BN+LeakyReLU+Dropout -> FC(16)+BN+LeakyReLU+Dropout -> FC(8)+LeakyReLU -> FC(1)")
    print(f"      - Total trainable parameters: {ann_model.count_parameters()}")

    trained_ann, ann_hist = train_model(
        ann_model,
        train_loader,
        val_loader,
        epochs=150,
        lr=0.01,
        weight_decay=1e-4,
        patience=20,
        device="cpu"
    )

    # 5. Evaluate Deep ANN
    print(f"\n[5/6] Evaluating Deep ANN on independent Test Set...")
    ann_metrics = evaluate_model_performance(trained_ann, splits["X_test"], splits["y_test"])
    print_evaluation_summary("Deep Moon ANN (3 Hidden Layers)", ann_metrics)

    # 6. Generate Diagnostics & Visualizations
    print(f"\n[6/6] Generating analytical plots and saving artifacts...")

    # Learning Curves
    learning_curves_path = os.path.join(output_dir, "02_learning_curves.png")
    plot_learning_curves(ann_hist, save_path=learning_curves_path)

    # Comparative Decision Boundaries
    boundary_path = os.path.join(output_dir, "03_linear_vs_ann_boundary.png")
    plot_comparative_decision_boundaries(
        trained_linear,
        trained_ann,
        splits["X_test"],
        splits["y_test"],
        linear_metrics["accuracy"],
        ann_metrics["accuracy"],
        save_path=boundary_path
    )

    # Confusion Matrix & ROC Curve
    diagnostics_path = os.path.join(output_dir, "04_diagnostics_roc_cm.png")
    plot_diagnostics(ann_metrics, save_path=diagnostics_path)

    # Save model weights
    ann_weights_path = os.path.join(output_dir, "deep_ann_model.pth")
    torch.save(trained_ann.state_dict(), ann_weights_path)
    print(f"[+] Saved trained ANN model weights to: {ann_weights_path}")

    # Save summary metrics to JSON
    summary_data = {
        "dataset": {
            "n_samples": n_samples,
            "noise": noise,
            "train_samples": len(splits["X_train"]),
            "val_samples": len(splits["X_val"]),
            "test_samples": len(splits["X_test"])
        },
        "baseline_linear_model": {
            "accuracy": float(linear_metrics["accuracy"]),
            "precision": float(linear_metrics["precision"]),
            "recall": float(linear_metrics["recall"]),
            "f1_score": float(linear_metrics["f1_score"]),
            "roc_auc": float(linear_metrics["roc_auc"])
        },
        "deep_ann_model": {
            "parameters": ann_model.count_parameters(),
            "best_epoch": ann_hist.get("best_epoch"),
            "best_val_loss": float(ann_hist.get("best_val_loss", 0.0)),
            "accuracy": float(ann_metrics["accuracy"]),
            "precision": float(ann_metrics["precision"]),
            "recall": float(ann_metrics["recall"]),
            "f1_score": float(ann_metrics["f1_score"]),
            "roc_auc": float(ann_metrics["roc_auc"]),
            "confusion_matrix": ann_metrics["confusion_matrix"].tolist()
        }
    }

    json_path = os.path.join(output_dir, "results_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary_data, f, indent=4)
    print(f"[+] Saved execution summary JSON to: {json_path}")

    print("\n" + "=" * 70)
    print(f" PIPELINE COMPLETED SUCCESSFULLY!")
    print(f" Baseline Linear Accuracy: {linear_metrics['accuracy'] * 100:.2f}%")
    print(f" Deep ANN Model Accuracy:  {ann_metrics['accuracy'] * 100:.2f}%")
    print(f" Deep ANN Model ROC-AUC:   {ann_metrics['roc_auc']:.4f}")
    print(f" Artifacts saved in: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()
