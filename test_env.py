import torch
import sklearn
from sklearn.datasets import make_moons
import matplotlib
import seaborn
import pandas

print("PyTorch version:", torch.__version__)
print("Scikit-learn version:", sklearn.__version__)
print("Testing make_moons generation...")
X, y = make_moons(n_samples=100, noise=0.2, random_state=42)
print("Data shape:", X.shape, y.shape)
print("Imports and synthetic dataset generation successful!")
