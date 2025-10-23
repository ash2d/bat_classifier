"""Project utilities shared across training and evaluation scripts."""
from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from sklearn import metrics


def set_seed(seed: int) -> None:
    """Seed random number generators for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_classification_metrics(
    targets: np.ndarray,
    probs: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Return per-class and aggregate metrics for multi-label classification."""

    preds = (probs >= threshold).astype(int)
    macro_precision = metrics.precision_score(targets, preds, average="macro", zero_division=0)
    macro_recall = metrics.recall_score(targets, preds, average="macro", zero_division=0)
    macro_f1 = metrics.f1_score(targets, preds, average="macro", zero_division=0)
    micro_precision = metrics.precision_score(targets, preds, average="micro", zero_division=0)
    micro_recall = metrics.recall_score(targets, preds, average="micro", zero_division=0)
    micro_f1 = metrics.f1_score(targets, preds, average="micro", zero_division=0)
    per_class_precision = metrics.precision_score(targets, preds, average=None, zero_division=0)
    per_class_recall = metrics.recall_score(targets, preds, average=None, zero_division=0)
    per_class_f1 = metrics.f1_score(targets, preds, average=None, zero_division=0)
    try:
        roc_auc_macro = metrics.roc_auc_score(targets, probs, average="macro")
    except ValueError:
        roc_auc_macro = float("nan")
    return {
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "per_class_precision": per_class_precision.tolist(),
        "per_class_recall": per_class_recall.tolist(),
        "per_class_f1": per_class_f1.tolist(),
        "roc_auc_macro": roc_auc_macro,
    }


@dataclass
class Checkpoint:
    """Container representing a model checkpoint."""

    epoch: int
    model_state: Dict[str, Any]
    optimizer_state: Dict[str, Any]
    scheduler_state: Optional[Dict[str, Any]]
    metrics: Dict[str, Any]


def save_checkpoint(checkpoint: Checkpoint, path: str | Path) -> None:
    """Persist a checkpoint to disk."""

    torch.save(asdict(checkpoint), path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> Checkpoint:
    """Load a checkpoint from disk."""

    payload = torch.load(path, map_location=map_location)
    return Checkpoint(**payload)


def init_wandb(project: str, config: Dict[str, Any], entity: Optional[str] = None) -> Any:
    """Initialise Weights & Biases logging.

    Set ``WANDB_API_KEY`` in your environment before calling this function or use
    ``wandb.login(key="YOUR_KEY_HERE")`` where indicated in ``train.py``.
    """

    import wandb  # Local import to keep dependency optional.

    run = wandb.init(project=project, entity=entity, config=config)
    return run


def dump_metrics(path: str | Path, metrics_dict: Dict[str, Any]) -> None:
    """Save metrics dictionary to JSON for later analysis."""

    with open(path, "w", encoding="utf-8") as fp:
        json.dump(metrics_dict, fp, indent=2)
