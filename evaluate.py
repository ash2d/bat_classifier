"""Evaluation script for the bat audio MIL classifier.

This script handles:
- Loading a trained model checkpoint
- Running inference on validation/test set
- Computing comprehensive evaluation metrics
- Generating per-class and aggregate metrics
- Optionally computing ROC-AUC curves
- Saving predictions and metrics
- Logging results to wandb

Before running:
1. Ensure you have a trained model checkpoint (e.g., checkpoints/best_model.pt)
2. Set the appropriate manifest path in config.yaml or via command line
3. Optionally set WANDB_API_KEY for result logging
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from data import BatClipDataset, BatDatasetConfig, bat_collate_fn
from model import CRNN, BatMILModel, CRNNConfig, MILPoolingHead
from utils import compute_classification_metrics, dump_metrics, init_wandb, load_checkpoint, set_seed


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def create_dataset_config(config: Dict[str, Any]) -> BatDatasetConfig:
    """Create BatDatasetConfig from full configuration."""
    audio_cfg = config["audio"]
    mil_cfg = config["mil"]
    return BatDatasetConfig(
        sample_rate=audio_cfg["sample_rate"],
        n_mels=audio_cfg["n_mels"],
        bag_size=mil_cfg["bag_size"],
        instance_frames=mil_cfg["instance_frames"],
        f_min=audio_cfg["f_min"],
        f_max=audio_cfg["f_max"],
        pcen_time_constant=audio_cfg["pcen"]["time_constant"],
        pcen_gain=audio_cfg["pcen"]["gain"],
        pcen_bias=audio_cfg["pcen"]["bias"],
        pcen_power=audio_cfg["pcen"]["power"],
        pcen_eps=audio_cfg["pcen"]["eps"],
        clip_duration=audio_cfg["clip_duration"],
        random_crop=False,  # No random cropping during evaluation
    )


def create_dataloader(
    config: Dict[str, Any],
    dataset_config: BatDatasetConfig,
    manifest_path: str,
) -> DataLoader:
    """Create evaluation dataloader."""
    data_cfg = config["data"]
    system_cfg = config["system"]
    eval_cfg = config["evaluation"]
    
    # Create label mapping
    label_to_index = {label: idx for idx, label in enumerate(data_cfg["classes"])}
    
    # Create dataset
    dataset = BatClipDataset(
        manifest_path=manifest_path,
        audio_root=data_cfg["audio_root"],
        label_to_index=label_to_index,
        config=dataset_config,
        label_delimiter=data_cfg["label_delimiter"],
        instance_hop_frames=config["mil"]["instance_hop_frames"],
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=eval_cfg["batch_size"],
        shuffle=False,
        num_workers=system_cfg["num_workers"],
        pin_memory=system_cfg["pin_memory"],
        collate_fn=lambda batch: bat_collate_fn(batch, pad_to_bag_size=dataset_config.bag_size),
    )
    
    return dataloader


def create_model(config: Dict[str, Any], num_classes: int, device: torch.device) -> BatMILModel:
    """Create the BatMILModel from configuration."""
    encoder_cfg = config["model"]["encoder"]
    pooling_cfg = config["model"]["pooling"]
    
    # Create CRNN config
    crnn_config = CRNNConfig(
        input_channels=encoder_cfg["input_channels"],
        conv_layers=encoder_cfg["conv_layers"],
        rnn_hidden_size=encoder_cfg["rnn_hidden_size"],
        rnn_num_layers=encoder_cfg["rnn_num_layers"],
        rnn_dropout=encoder_cfg["rnn_dropout"],
        bidirectional=encoder_cfg["bidirectional"],
    )
    
    # Create encoder
    encoder = CRNN(crnn_config)
    
    # Calculate feature dimension after encoder
    feature_dim = encoder_cfg["rnn_hidden_size"]
    if encoder_cfg["bidirectional"]:
        feature_dim *= 2
    
    # Create pooling head
    pooling_head = MILPoolingHead(
        input_dim=feature_dim,
        output_dim=num_classes,
        pooling_type=pooling_cfg["pooling_type"],
        attention_hidden=pooling_cfg["attention_hidden"],
    )
    
    # Create full model
    model = BatMILModel(encoder=encoder, pooling_head=pooling_head)
    model = model.to(device)
    
    return model


def evaluate_model(
    model: BatMILModel,
    dataloader: DataLoader,
    device: torch.device,
    config: Dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference and collect predictions.
    
    Returns:
        tuple: (all_targets, all_probs, all_logits)
    """
    model.eval()
    
    all_targets = []
    all_probs = []
    all_logits = []
    
    print("Running inference...")
    with torch.no_grad():
        for batch_idx, (instances, labels, lengths) in enumerate(dataloader):
            instances = instances.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)
            
            # Forward pass
            logits = model(instances, lengths)
            probs = torch.sigmoid(logits)
            
            # Store predictions
            all_targets.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_logits.append(logits.cpu().numpy())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1}/{len(dataloader)} batches")
    
    # Concatenate all batches
    all_targets = np.vstack(all_targets)
    all_probs = np.vstack(all_probs)
    all_logits = np.vstack(all_logits)
    
    return all_targets, all_probs, all_logits


def compute_detailed_metrics(
    targets: np.ndarray,
    probs: np.ndarray,
    config: Dict[str, Any],
    class_names: list[str],
) -> Dict[str, Any]:
    """Compute comprehensive evaluation metrics.
    
    Args:
        targets: Ground truth labels (N, C)
        probs: Predicted probabilities (N, C)
        config: Configuration dictionary
        class_names: List of class names
    
    Returns:
        Dictionary containing all metrics
    """
    eval_cfg = config["evaluation"]
    threshold = eval_cfg["threshold"]
    
    # Compute basic metrics
    metrics = compute_classification_metrics(targets, probs, threshold=threshold)
    
    # Add per-class metrics with names
    per_class_metrics = {}
    for idx, class_name in enumerate(class_names):
        per_class_metrics[class_name] = {
            "precision": metrics["per_class_precision"][idx],
            "recall": metrics["per_class_recall"][idx],
            "f1": metrics["per_class_f1"][idx],
            "support": int(targets[:, idx].sum()),  # Number of positive samples
        }
    
    # Compute additional ROC-AUC metrics if requested
    if eval_cfg["compute_roc_auc"]:
        try:
            from sklearn.metrics import roc_auc_score, roc_curve
            
            # Per-class ROC-AUC
            per_class_auc = []
            for idx in range(targets.shape[1]):
                if targets[:, idx].sum() > 0:  # Only if class has positive samples
                    auc = roc_auc_score(targets[:, idx], probs[:, idx])
                    per_class_auc.append(auc)
                else:
                    per_class_auc.append(float("nan"))
            
            metrics["per_class_auc"] = per_class_auc
            
            # Add AUC to per-class metrics
            for idx, class_name in enumerate(class_names):
                per_class_metrics[class_name]["auc"] = per_class_auc[idx]
        
        except Exception as e:
            print(f"Warning: Could not compute ROC-AUC: {e}")
    
    # Overall statistics
    preds = (probs >= threshold).astype(int)
    metrics["total_samples"] = len(targets)
    metrics["avg_labels_per_sample"] = float(targets.sum(axis=1).mean())
    metrics["avg_predictions_per_sample"] = float(preds.sum(axis=1).mean())
    
    # Compile full results
    results = {
        "aggregate_metrics": {
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "macro_f1": metrics["macro_f1"],
            "micro_precision": metrics["micro_precision"],
            "micro_recall": metrics["micro_recall"],
            "micro_f1": metrics["micro_f1"],
            "roc_auc_macro": metrics["roc_auc_macro"],
        },
        "per_class_metrics": per_class_metrics,
        "statistics": {
            "total_samples": metrics["total_samples"],
            "avg_labels_per_sample": metrics["avg_labels_per_sample"],
            "avg_predictions_per_sample": metrics["avg_predictions_per_sample"],
            "threshold": threshold,
        },
    }
    
    return results


def save_predictions(
    targets: np.ndarray,
    probs: np.ndarray,
    class_names: list[str],
    output_path: str,
    manifest_df: Optional[pd.DataFrame] = None,
) -> None:
    """Save predictions to CSV file.
    
    Args:
        targets: Ground truth labels (N, C)
        probs: Predicted probabilities (N, C)
        class_names: List of class names
        output_path: Path to save predictions
        manifest_df: Optional manifest dataframe to include file paths
    """
    # Create dataframe with predictions
    pred_df = pd.DataFrame(probs, columns=[f"prob_{name}" for name in class_names])
    
    # Add ground truth
    for idx, name in enumerate(class_names):
        pred_df[f"target_{name}"] = targets[:, idx]
    
    # Add file paths if available
    if manifest_df is not None and "filepath" in manifest_df.columns:
        pred_df.insert(0, "filepath", manifest_df["filepath"].values[:len(pred_df)])
    
    # Save to CSV
    pred_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")


def print_evaluation_summary(results: Dict[str, Any]) -> None:
    """Print a formatted summary of evaluation results."""
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    
    # Aggregate metrics
    print("\nAggregate Metrics:")
    print("-" * 80)
    agg = results["aggregate_metrics"]
    print(f"Macro Precision: {agg['macro_precision']:.4f}")
    print(f"Macro Recall:    {agg['macro_recall']:.4f}")
    print(f"Macro F1:        {agg['macro_f1']:.4f}")
    print(f"Micro Precision: {agg['micro_precision']:.4f}")
    print(f"Micro Recall:    {agg['micro_recall']:.4f}")
    print(f"Micro F1:        {agg['micro_f1']:.4f}")
    print(f"ROC-AUC (macro): {agg['roc_auc_macro']:.4f}")
    
    # Statistics
    print("\nDataset Statistics:")
    print("-" * 80)
    stats = results["statistics"]
    print(f"Total Samples:               {stats['total_samples']}")
    print(f"Avg Labels per Sample:       {stats['avg_labels_per_sample']:.2f}")
    print(f"Avg Predictions per Sample:  {stats['avg_predictions_per_sample']:.2f}")
    print(f"Classification Threshold:    {stats['threshold']:.2f}")
    
    # Per-class metrics (top 10 by F1)
    print("\nPer-Class Metrics (Top 10 by F1):")
    print("-" * 80)
    per_class = results["per_class_metrics"]
    sorted_classes = sorted(
        per_class.items(),
        key=lambda x: x[1]["f1"],
        reverse=True,
    )[:10]
    
    print(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 80)
    for class_name, metrics in sorted_classes:
        print(f"{class_name:<20} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} "
              f"{metrics['f1']:>10.4f} {metrics['support']:>10}")
    
    print("=" * 80)


def evaluate(
    config: Dict[str, Any],
    checkpoint_path: str,
    manifest_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Main evaluation function.
    
    Args:
        config: Configuration dictionary
        checkpoint_path: Path to model checkpoint
        manifest_path: Optional path to evaluation manifest (overrides config)
    
    Returns:
        Dictionary containing evaluation results
    """
    # Set seed for reproducibility
    set_seed(config["system"]["seed"])
    
    # Set device
    device_str = config["system"]["device"]
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device_str = "cpu"
    device = torch.device(device_str)
    
    # Use provided manifest or default to validation manifest
    if manifest_path is None:
        manifest_path = config["data"]["val_manifest"]
    
    print(f"Evaluating on: {manifest_path}")
    
    # Create dataset config and dataloader
    dataset_config = create_dataset_config(config)
    dataloader = create_dataloader(config, dataset_config, manifest_path)
    print(f"Loaded {len(dataloader.dataset)} samples")
    
    # Create model
    num_classes = len(config["data"]["classes"])
    class_names = config["data"]["classes"]
    model = create_model(config, num_classes, device)
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint.model_state)
    print(f"Loaded checkpoint from epoch {checkpoint.epoch}")
    
    # Run evaluation
    targets, probs, logits = evaluate_model(model, dataloader, device, config)
    
    # Compute metrics
    print("\nComputing metrics...")
    results = compute_detailed_metrics(targets, probs, config, class_names)
    
    # Print summary
    print_evaluation_summary(results)
    
    # Save predictions if requested
    eval_cfg = config["evaluation"]
    if eval_cfg["save_predictions"]:
        # Try to load manifest for file paths
        manifest_df = None
        try:
            manifest_df = pd.read_csv(manifest_path)
        except Exception as e:
            print(f"Warning: Could not load manifest for file paths: {e}")
        
        save_predictions(
            targets,
            probs,
            class_names,
            eval_cfg["predictions_path"],
            manifest_df,
        )
    
    # Save metrics to JSON
    metrics_output_path = Path(eval_cfg["predictions_path"]).parent / "evaluation_metrics.json"
    dump_metrics(metrics_output_path, results)
    print(f"Metrics saved to {metrics_output_path}")
    
    # Log to wandb if enabled
    if config["logging"]["use_wandb"]:
        try:
            wandb_run = init_wandb(
                project=config["logging"]["wandb_project"],
                entity=config["logging"]["wandb_entity"],
                config=config,
            )
            
            # Log aggregate metrics
            wandb_run.log(results["aggregate_metrics"])
            
            # Log per-class metrics as a table
            import wandb
            per_class_data = []
            for class_name, metrics in results["per_class_metrics"].items():
                per_class_data.append([class_name] + list(metrics.values()))
            
            columns = ["class", "precision", "recall", "f1", "support"]
            if eval_cfg["compute_roc_auc"]:
                columns.append("auc")
            
            table = wandb.Table(columns=columns, data=per_class_data)
            wandb_run.log({"per_class_metrics": table})
            
            wandb_run.finish()
            print("Results logged to wandb")
        
        except Exception as e:
            print(f"Warning: Failed to log to wandb: {e}")
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate bat audio MIL classifier")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to model checkpoint (default: checkpoints/best_model.pt)",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to evaluation manifest (default: use val_manifest from config)",
    )
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Run evaluation
    evaluate(config, args.checkpoint, args.manifest)


if __name__ == "__main__":
    main()
