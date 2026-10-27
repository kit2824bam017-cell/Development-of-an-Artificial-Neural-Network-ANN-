# Development of an Artificial Neural Network (ANN) for Binary Classification Using the make_moons Dataset

An end-to-end deep learning project implementing a multi-layer **Artificial Neural Network (ANN)** in **PyTorch** for non-linear binary classification on the benchmark **`make_moons`** dataset from **Scikit-Learn**. 

Includes automated model training, early stopping, publication-grade visualizations, quantitative metrics comparison against a linear baseline, a step-by-step Jupyter Notebook, and an interactive web studio.

---

## 🚀 Key Highlights & Benchmark Results

The `make_moons` dataset represents two interleaving, crescent-shaped semicircles in $\mathbb{R}^2$. Because the classes are **non-linearly separable**, single-layer linear models (like Logistic Regression) suffer from an underfitting bottleneck, whereas our Deep ANN achieves near-perfect classification.

| Metric | Baseline Linear Perceptron | Deep ANN (3 Hidden Layers) | Improvement |
| :--- | :---: | :---: | :---: |
| **Test Accuracy** | **88.44%** | **95.56%** | **+7.12%** |
| **ROC-AUC** | 0.9569 | **0.9907** | **+0.0338** |
| **Precision** | 0.8981 | **0.9636** | **+0.0655** |
| **Recall** | 0.8661 | **0.9464** | **+0.0803** |
| **F1-Score** | 0.8818 | **0.9550** | **+0.0732** |
| **Trainable Parameters** | 3 | **881** | — |

---

## 🧠 Neural Network Architecture

The network is designed with batch normalization and leaky non-linearities to avoid gradient vanishing and dying neurons:

```
Input Features [X1, X2] (Dim: 2)
       │
       ▼
Dense Layer 1 (2 ➔ 32)
  ├── Batch Normalization (BatchNorm1d)
  ├── LeakyReLU Activation (α = 0.1)
  └── Dropout (p = 0.1)
       │
       ▼
Dense Layer 2 (32 ➔ 16)
  ├── Batch Normalization (BatchNorm1d)
  ├── LeakyReLU Activation (α = 0.1)
  └── Dropout (p = 0.1)
       │
       ▼
Dense Layer 3 (16 ➔ 8)
  └── LeakyReLU Activation (α = 0.1)
       │
       ▼
Output Projection (8 ➔ 1 Logit)
       │
       ▼
Binary Cross-Entropy with Logits (BCEWithLogitsLoss) ➔ Sigmoid σ(z)
```

### Mathematical Foundations
1. **Leaky ReLU Activation**:
   $$\text{LeakyReLU}(z) = \begin{cases} z & \text{if } z > 0 \\ \alpha z & \text{if } z \le 0 \end{cases} \quad (\alpha = 0.1)$$
   Provides a non-zero gradient $\alpha$ for negative pre-activations, eliminating dead neurons.

2. **Numerically Stable Binary Cross-Entropy**:
   $$\mathcal{L}(z, y) = \max(z, 0) - z \cdot y + \log(1 + e^{-|z|})$$
   Evaluates logits directly via the log-sum-exp trick to prevent numerical overflow in floating-point calculations.

3. **Weight Initialization**:
   Weights are initialized using **Kaiming / He Normal** initialization tailored for LeakyReLU non-linearities:
   $$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{(1 + \alpha^2) \cdot d_{\text{in}}}}\right)$$

---

## 📁 Repository Structure

```
.
├── dataset.py                          # Data synthesis (make_moons), stratified splitting & scaling
├── model.py                            # PyTorch LinearClassifier and DeepMoonANN model architectures
├── train.py                            # Training loop with Adam, ReduceLROnPlateau, and Early Stopping
├── evaluate.py                         # Evaluation metrics: Accuracy, Precision, Recall, F1, ROC-AUC, CM
├── visualize.py                        # Plotting scripts for publication-grade PNG diagrams
├── main.py                             # Automated end-to-end execution pipeline
├── ann_classification_make_moons.ipynb # Interactive, runnable Jupyter Notebook
├── requirements.txt                    # Project Python dependencies
├── outputs/                            # Generated artifacts & model weights
│   ├── 01_dataset_distribution.png     # Scatter plot of raw crescent moons
│   ├── 02_learning_curves.png          # Train/Val Loss and Accuracy vs Epochs
│   ├── 03_linear_vs_ann_boundary.png   # Comparative decision boundary visualization
│   ├── 04_diagnostics_roc_cm.png       # Test Confusion Matrix heatmap and ROC curve
│   ├── deep_ann_model.pth              # Saved PyTorch model checkpoint weights
│   └── results_summary.json            # Machine-readable performance results
└── web/                                # Interactive browser studio
    ├── index.html                      # Modern visual dashboard with 2D interactive canvas
    ├── style.css                       # Premium glassmorphic dark-mode design system
    └── app.js                          # Real-time probe simulator and decision boundary renderer
```

---

## ⚡ Quick Start

### 1. Installation
Clone or navigate to this workspace and install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Execute Full Pipeline
Run the main script to train models, evaluate test partitions, and generate all visualization artifacts:
```bash
python main.py
```

### 3. Run Interactive Jupyter Notebook
Open [`ann_classification_make_moons.ipynb`](file:///c:/Users/abila/OneDrive/Desktop/Development%20of%20an%20Artificial%20Neural%20Network%20(ANN)%20for%20Binary%20Classification%20Using%20the%20make_moons%20Dataset%20from%20Scikit-learn/ann_classification_make_moons.ipynb) in VS Code or JupyterLab to experiment interactively cell-by-cell.

### 4. Launch Interactive Web Dashboard
Open [`web/index.html`](file:///c:/Users/abila/OneDrive/Desktop/Development%20of%20an%20Artificial%20Neural%20Network%20(ANN)%20for%20Binary%20Classification%20Using%20the%20make_moons%20Dataset%20from%20Scikit-learn/web/index.html) in any web browser to:
- Test custom coordinates $(X_1, X_2)$ with live classification probabilities.
- Toggle between Linear and Non-Linear ANN decision boundaries.
- Adjust decision thresholds $\tau$, dataset noise $\sigma$, and point counts in real time.

---

## 📊 Visual Diagnostics Preview

- **Comparative Decision Boundaries**: Displays how the linear hyperplane cuts indiscriminately across the moons (clamping accuracy at 88.4%), whereas the ANN's non-linear manifold smoothly winds between both half-circles (95.56%).
- **Learning Curves**: Demonstrates rapid convergence within ~40 epochs, stabilized by `BatchNorm1d` and early stopping.
- **ROC Curve**: Demonstrates an Area Under the Curve (AUC) of **0.9907**.
