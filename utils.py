"""
Utility functions for bat audio classification system.
"""

import os
import random
import numpy as np
import torch
import yaml
import wandb
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_device(device_str: str = "cuda") -> torch.device:
    """
    Get torch device based on availability.
    
    Args:
        device_str: Preferred device ("cuda", "cpu", "mps")
        
    Returns:
        torch.device object
    """
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif device_str == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def init_wandb(config: dict):
    """
    Initialize Weights & Biases logging.
    
    Args:
        config: Configuration dictionary
    """
    if config.get('wandb', {}).get('enabled', False):
        wandb.init(
            project=config['wandb'].get('project', 'bat-classification'),
            entity=config['wandb'].get('entity', None),
            name=config['wandb'].get('run_name', None),
            tags=config['wandb'].get('tags', []),
            config=config,
        )


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    metrics: dict,
    filepath: str,
):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch number
        loss: Current loss value
        metrics: Dictionary of metrics
        filepath: Path to save checkpoint
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics,
    }
    
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    filepath: str,
    device: torch.device,
):
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        filepath: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        Tuple of (epoch, loss, metrics)
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', 0.0)
    metrics = checkpoint.get('metrics', {})
    
    print(f"Checkpoint loaded from {filepath}")
    return epoch, loss, metrics


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5):
    """
    Compute classification metrics for multi-label classification.
    
    Args:
        y_true: Ground truth labels (N, C)
        y_pred: Predicted probabilities (N, C)
        threshold: Classification threshold
        
    Returns:
        Dictionary containing various metrics
    """
    # Convert predictions to binary using threshold
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    # Compute per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred_binary, average=None, zero_division=0
    )
    
    # Compute macro-averaged metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred_binary, average='macro', zero_division=0
    )
    
    # Compute micro-averaged metrics
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred_binary, average='micro', zero_division=0
    )
    
    # Compute weighted-averaged metrics
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred_binary, average='weighted', zero_division=0
    )
    
    metrics = {
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'per_class_precision': precision,
        'per_class_recall': recall,
        'per_class_f1': f1,
        'per_class_support': support,
    }
    
    # Compute ROC-AUC if possible
    try:
        # Macro-averaged ROC-AUC
        roc_auc_macro = roc_auc_score(y_true, y_pred, average='macro')
        # Micro-averaged ROC-AUC
        roc_auc_micro = roc_auc_score(y_true, y_pred, average='micro')
        # Per-class ROC-AUC
        roc_auc_per_class = roc_auc_score(y_true, y_pred, average=None)
        
        metrics['roc_auc_macro'] = roc_auc_macro
        metrics['roc_auc_micro'] = roc_auc_micro
        metrics['roc_auc_per_class'] = roc_auc_per_class
    except ValueError:
        # Handle cases where ROC-AUC cannot be computed
        pass
    
    # Compute Average Precision (AP) if possible
    try:
        ap_macro = average_precision_score(y_true, y_pred, average='macro')
        ap_micro = average_precision_score(y_true, y_pred, average='micro')
        ap_per_class = average_precision_score(y_true, y_pred, average=None)
        
        metrics['ap_macro'] = ap_macro
        metrics['ap_micro'] = ap_micro
        metrics['ap_per_class'] = ap_per_class
    except ValueError:
        pass
    
    return metrics


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve.
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change in monitored value to qualify as improvement
            mode: One of 'min' or 'max' - whether lower or higher is better
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.early_stop = False
        
    def __call__(self, current_value: float) -> bool:
        """
        Check if training should be stopped.
        
        Args:
            current_value: Current metric value
            
        Returns:
            True if training should be stopped, False otherwise
        """
        if self.best_value is None:
            self.best_value = current_value
            return False
        
        if self.mode == 'min':
            improved = current_value < (self.best_value - self.min_delta)
        else:
            improved = current_value > (self.best_value + self.min_delta)
        
        if improved:
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False


def log_metrics_wandb(metrics: dict, step: int = None, prefix: str = ""):
    """
    Log metrics to Weights & Biases.
    
    Args:
        metrics: Dictionary of metrics to log
        step: Step number (epoch or batch)
        prefix: Prefix for metric names (e.g., "train/", "val/")
    """
    if wandb.run is not None:
        log_dict = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                log_dict[f"{prefix}{key}"] = value
            elif isinstance(value, np.ndarray) and value.size == 1:
                log_dict[f"{prefix}{key}"] = value.item()
        
        if step is not None:
            wandb.log(log_dict, step=step)
        else:
            wandb.log(log_dict)
