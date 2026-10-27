"""
dataset.py - Dataset generation, preprocessing, and PyTorch DataLoader utilities
for binary classification using the Scikit-learn make_moons dataset.
"""

from typing import Tuple, Dict, Any
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def generate_moons_data(
    n_samples: int = 1500,
    noise: float = 0.20,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate the two interleaving half-circles (make_moons) synthetic dataset.

    Args:
        n_samples (int): Total number of samples generated.
        noise (float): Standard deviation of Gaussian noise added to the data.
        random_state (int): Seed for reproducible random number generation.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Feature matrix X of shape (n_samples, 2)
                                       and binary labels y of shape (n_samples,).
    """
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    return X, y


def prepare_data_splits(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Split the dataset into Train, Validation, and Test sets, and apply standard scaling.
    To prevent data leakage, StandardScaler is fitted strictly on the training partition.

    Args:
        X (np.ndarray): Input 2D feature matrix (shape: [N, 2]).
        y (np.ndarray): Binary target labels (0 or 1).
        test_size (float): Fraction of data reserved for the final test set.
        val_size (float): Fraction of data reserved for validation.
        random_state (int): Random seed.

    Returns:
        Dict[str, Any]: Dictionary containing raw and scaled arrays for train/val/test,
                        as well as the fitted scaler.
    """
    # First split off the test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Adjust val_size relative to the remaining temp set
    adjusted_val_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=adjusted_val_size, random_state=random_state, stratify=y_temp
    )

    # Fit scaler only on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return {
        "X_train_raw": X_train,
        "X_val_raw": X_val,
        "X_test_raw": X_test,
        "X_train": X_train_scaled,
        "X_val": X_val_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "scaler": scaler
    }


def create_dataloaders(
    data_dict: Dict[str, Any],
    batch_size: int = 32
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Convert NumPy partitions into PyTorch TensorDatasets and DataLoaders.

    Args:
        data_dict (Dict[str, Any]): Preprocessed data split dictionary.
        batch_size (int): Batch size for mini-batch stochastic gradient descent.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: (train_loader, val_loader, test_loader)
    """
    # Convert features to torch.FloatTensor and labels to torch.FloatTensor (shape: [N, 1])
    train_x = torch.tensor(data_dict["X_train"], dtype=torch.float32)
    train_y = torch.tensor(data_dict["y_train"], dtype=torch.float32).unsqueeze(1)

    val_x = torch.tensor(data_dict["X_val"], dtype=torch.float32)
    val_y = torch.tensor(data_dict["y_val"], dtype=torch.float32).unsqueeze(1)

    test_x = torch.tensor(data_dict["X_test"], dtype=torch.float32)
    test_y = torch.tensor(data_dict["y_test"], dtype=torch.float32).unsqueeze(1)

    train_dataset = TensorDataset(train_x, train_y)
    val_dataset = TensorDataset(val_x, val_y)
    test_dataset = TensorDataset(test_x, test_y)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    X, y = generate_moons_data(n_samples=1500, noise=0.20, random_state=42)
    splits = prepare_data_splits(X, y)
    train_loader, val_loader, test_loader = create_dataloaders(splits, batch_size=32)

    print(f"Total samples: {len(X)}")
    print(f"Train samples: {len(splits['X_train'])} ({len(splits['X_train'])/len(X):.1%})")
    print(f"Val samples:   {len(splits['X_val'])} ({len(splits['X_val'])/len(X):.1%})")
    print(f"Test samples:  {len(splits['X_test'])} ({len(splits['X_test'])/len(X):.1%})")
    print(f"Features: {X.shape[1]}, Classes: {len(np.unique(y))}")
