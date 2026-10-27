"""
train.py - Training engine with Adam optimizer, BCEWithLogitsLoss, learning rate
scheduling, and early stopping for the make_moons binary classification model.
"""

from typing import Dict, Any, Tuple
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """
    Perform one forward & backward pass over the training data.

    Returns:
        Tuple[float, float]: (average_loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == batch_y).sum().item()
        total += batch_y.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Evaluate the model on validation or test DataLoader without gradient tracking.

    Returns:
        Tuple[float, float]: (average_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            total_loss += loss.item() * batch_x.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 150,
    lr: float = 0.01,
    weight_decay: float = 1e-4,
    patience: int = 20,
    device: str = "cpu"
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Execute full training routine with Adam optimizer, ReduceLROnPlateau, and Early Stopping.

    Args:
        model (nn.Module): PyTorch model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        epochs (int): Maximum number of training epochs.
        lr (float): Initial learning rate for Adam.
        weight_decay (float): L2 regularization factor.
        patience (int): Number of epochs to wait for val loss improvement before early stopping.
        device (str): Compute device ('cpu' or 'cuda').

    Returns:
        Tuple[nn.Module, Dict[str, Any]]: Trained model loaded with best weights, and history dictionary.
    """
    dev = torch.device(device)
    model.to(dev)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-5
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "lr": [],
        "best_epoch": 0,
        "best_val_loss": float("inf"),
        "best_val_acc": 0.0
    }

    best_weights = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, dev)
        val_loss, val_acc = evaluate_epoch(model, val_loader, criterion, dev)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            history["best_val_loss"] = best_val_loss
            history["best_val_acc"] = val_acc
            history["best_epoch"] = epoch
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch:03d}/{epochs:03d}] | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                  f"LR: {current_lr:.6f}")

        # Early stopping trigger
        if patience_counter >= patience:
            print(f"[*] Early stopping triggered at epoch {epoch}. Best Val Loss: {best_val_loss:.4f}")
            break

    # Restore best model weights
    model.load_state_dict(best_weights)
    return model, history
