"""
Evaluation script for bat audio classification model.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb

from utils import (
    set_seed,
    load_config,
    get_device,
    load_checkpoint,
    compute_metrics,
    log_metrics_wandb,
)
from data import create_dataloader
from model import create_model


def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple:
    """
    Evaluate model on dataset.
    
    Args:
        model: Neural network model
        dataloader: Data loader
        device: Device to evaluate on
        
    Returns:
        Tuple of (predictions, labels)
    """
    model.eval()
    
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        for instances, labels in pbar:
            # Move to device
            instances = instances.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits = model(instances)
            probs = torch.sigmoid(logits)
            
            # Collect predictions and labels
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # Concatenate all predictions and labels
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    return all_preds, all_labels


def evaluate_model(config: dict, checkpoint_path: str = None):
    """
    Main evaluation function.
    
    Args:
        config: Configuration dictionary
        checkpoint_path: Path to model checkpoint (optional)
    """
    # Set seed for reproducibility
    set_seed(config['seed'])
    
    # Get device
    device = get_device(config['device'])
    print(f"Using device: {device}")
    
    # Create test data loader
    print("Creating test data loader...")
    test_loader = create_dataloader(
        csv_path=config['paths']['test_csv'],
        data_dir=config['paths']['data_dir'],
        config=config,
        train=False,
        shuffle=False,
    )
    
    print(f"Test batches: {len(test_loader)}")
    
    # Create model
    print("Creating model...")
    model = create_model(config)
    model = model.to(device)
    
    # Load checkpoint if provided
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        optimizer = torch.optim.Adam(model.parameters())  # Dummy optimizer for loading
        epoch, loss, metrics = load_checkpoint(model, optimizer, checkpoint_path, device)
        print(f"Loaded checkpoint from epoch {epoch} with loss {loss:.4f}")
    else:
        print("No checkpoint provided or checkpoint not found. Using randomly initialized model.")
    
    # Evaluate
    print("\nEvaluating model...")
    predictions, labels = evaluate(model, test_loader, device)
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(labels, predictions, threshold=0.5)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    print("\nMacro-averaged metrics:")
    print(f"  Precision: {metrics['precision_macro']:.4f}")
    print(f"  Recall:    {metrics['recall_macro']:.4f}")
    print(f"  F1-score:  {metrics['f1_macro']:.4f}")
    
    print("\nMicro-averaged metrics:")
    print(f"  Precision: {metrics['precision_micro']:.4f}")
    print(f"  Recall:    {metrics['recall_micro']:.4f}")
    print(f"  F1-score:  {metrics['f1_micro']:.4f}")
    
    print("\nWeighted-averaged metrics:")
    print(f"  Precision: {metrics['precision_weighted']:.4f}")
    print(f"  Recall:    {metrics['recall_weighted']:.4f}")
    print(f"  F1-score:  {metrics['f1_weighted']:.4f}")
    
    if 'roc_auc_macro' in metrics:
        print("\nROC-AUC scores:")
        print(f"  Macro: {metrics['roc_auc_macro']:.4f}")
        print(f"  Micro: {metrics['roc_auc_micro']:.4f}")
    
    if 'ap_macro' in metrics:
        print("\nAverage Precision scores:")
        print(f"  Macro: {metrics['ap_macro']:.4f}")
        print(f"  Micro: {metrics['ap_micro']:.4f}")
    
    # Per-class metrics
    print("\nPer-class metrics:")
    print(f"{'Class':<8} {'Precision':<12} {'Recall':<12} {'F1-score':<12} {'Support':<10}")
    print("-" * 60)
    
    num_classes = config['data']['num_classes']
    for i in range(num_classes):
        precision = metrics['per_class_precision'][i]
        recall = metrics['per_class_recall'][i]
        f1 = metrics['per_class_f1'][i]
        support = metrics['per_class_support'][i]
        
        print(f"{i:<8} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {support:<10.0f}")
    
    print("="*50)
    
    # Log to wandb if enabled
    if config.get('wandb', {}).get('enabled', False):
        print("\nLogging metrics to Weights & Biases...")
        
        # Initialize wandb for evaluation
        wandb.init(
            project=config['wandb'].get('project', 'bat-classification'),
            entity=config['wandb'].get('entity', None),
            name=f"eval_{config['wandb'].get('run_name', 'test')}",
            tags=config['wandb'].get('tags', []) + ['evaluation'],
            config=config,
        )
        
        # Log metrics
        log_dict = {
            'test/precision_macro': metrics['precision_macro'],
            'test/recall_macro': metrics['recall_macro'],
            'test/f1_macro': metrics['f1_macro'],
            'test/precision_micro': metrics['precision_micro'],
            'test/recall_micro': metrics['recall_micro'],
            'test/f1_micro': metrics['f1_micro'],
            'test/precision_weighted': metrics['precision_weighted'],
            'test/recall_weighted': metrics['recall_weighted'],
            'test/f1_weighted': metrics['f1_weighted'],
        }
        
        if 'roc_auc_macro' in metrics:
            log_dict['test/roc_auc_macro'] = metrics['roc_auc_macro']
            log_dict['test/roc_auc_micro'] = metrics['roc_auc_micro']
        
        if 'ap_macro' in metrics:
            log_dict['test/ap_macro'] = metrics['ap_macro']
            log_dict['test/ap_micro'] = metrics['ap_micro']
        
        wandb.log(log_dict)
        
        # Log per-class metrics as table
        per_class_data = []
        for i in range(num_classes):
            per_class_data.append([
                i,
                metrics['per_class_precision'][i],
                metrics['per_class_recall'][i],
                metrics['per_class_f1'][i],
                int(metrics['per_class_support'][i]),
            ])
        
        table = wandb.Table(
            columns=['Class', 'Precision', 'Recall', 'F1-score', 'Support'],
            data=per_class_data,
        )
        wandb.log({'test/per_class_metrics': table})
        
        # Upload checkpoint to wandb if provided
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            wandb.save(checkpoint_path)
            print(f"Checkpoint uploaded to Weights & Biases")
        
        wandb.finish()
    
    # Save predictions
    output_dir = config['paths'].get('log_dir', './logs')
    os.makedirs(output_dir, exist_ok=True)
    
    predictions_path = os.path.join(output_dir, 'test_predictions.npz')
    np.savez(
        predictions_path,
        predictions=predictions,
        labels=labels,
        **{k: v for k, v in metrics.items() if isinstance(v, (int, float, np.ndarray))}
    )
    print(f"\nPredictions saved to {predictions_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate bat audio classifier")
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file',
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to model checkpoint',
    )
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Use best checkpoint if not specified
    if args.checkpoint is None:
        checkpoint_path = os.path.join(
            config['paths']['checkpoint_dir'],
            'best_model.pt',
        )
    else:
        checkpoint_path = args.checkpoint
    
    # Evaluate model
    evaluate_model(config, checkpoint_path)


if __name__ == '__main__':
    main()
