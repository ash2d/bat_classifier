"""
Training script for bat audio classification model.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from tqdm import tqdm

from utils import (
    set_seed,
    load_config,
    get_device,
    init_wandb,
    save_checkpoint,
    EarlyStopping,
    log_metrics_wandb,
)
from data import create_dataloader
from model import create_model


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> dict:
    """
    Train for one epoch.
    
    Args:
        model: Neural network model
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        
    Returns:
        Dictionary of training metrics
    """
    model.train()
    
    total_loss = 0.0
    all_labels = []
    all_preds = []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, (instances, labels) in enumerate(pbar):
        # Move to device
        instances = instances.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(instances)
        
        # Compute loss
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        
        # Collect predictions and labels
        probs = torch.sigmoid(logits)
        all_preds.append(probs.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item()})
    
    # Compute average loss
    avg_loss = total_loss / len(dataloader)
    
    # Concatenate all predictions and labels
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    metrics = {
        'loss': avg_loss,
        'predictions': all_preds,
        'labels': all_labels,
    }
    
    return metrics


def validate_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> dict:
    """
    Validate for one epoch.
    
    Args:
        model: Neural network model
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        epoch: Current epoch number
        
    Returns:
        Dictionary of validation metrics
    """
    model.eval()
    
    total_loss = 0.0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
        for batch_idx, (instances, labels) in enumerate(pbar):
            # Move to device
            instances = instances.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits = model(instances)
            
            # Compute loss
            loss = criterion(logits, labels)
            
            # Track metrics
            total_loss += loss.item()
            
            # Collect predictions and labels
            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
    
    # Compute average loss
    avg_loss = total_loss / len(dataloader)
    
    # Concatenate all predictions and labels
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    metrics = {
        'loss': avg_loss,
        'predictions': all_preds,
        'labels': all_labels,
    }
    
    return metrics


def train(config: dict):
    """
    Main training function.
    
    Args:
        config: Configuration dictionary
    """
    # Set seed for reproducibility
    set_seed(config['seed'])
    
    # Get device
    device = get_device(config['device'])
    print(f"Using device: {device}")
    
    # Initialize wandb
    init_wandb(config)
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader = create_dataloader(
        csv_path=config['paths']['train_csv'],
        data_dir=config['paths']['data_dir'],
        config=config,
        train=True,
        shuffle=True,
    )
    
    val_loader = create_dataloader(
        csv_path=config['paths']['val_csv'],
        data_dir=config['paths']['data_dir'],
        config=config,
        train=False,
        shuffle=False,
    )
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create model
    print("Creating model...")
    model = create_model(config)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create loss function
    class_weights = config['training'].get('class_weights', None)
    if class_weights is not None:
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
    )
    
    # Create learning rate scheduler
    scheduler_type = config['training'].get('scheduler', 'step')
    if scheduler_type == 'step':
        scheduler = StepLR(
            optimizer,
            step_size=config['training']['step_size'],
            gamma=config['training']['gamma'],
        )
    elif scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config['training']['epochs'],
        )
    else:
        scheduler = None
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config['training']['patience'],
        mode='min',
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(1, config['training']['epochs'] + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{config['training']['epochs']}")
        print(f"{'='*50}")
        
        # Train
        train_metrics = train_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
        )
        
        print(f"Train Loss: {train_metrics['loss']:.4f}")
        
        # Validate
        val_metrics = validate_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
        )
        
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        
        # Log metrics to wandb
        log_metrics_wandb(
            {'loss': train_metrics['loss']},
            step=epoch,
            prefix='train/',
        )
        log_metrics_wandb(
            {'loss': val_metrics['loss']},
            step=epoch,
            prefix='val/',
        )
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            checkpoint_path = os.path.join(
                config['paths']['checkpoint_dir'],
                'best_model.pt',
            )
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=val_metrics['loss'],
                metrics={'val_loss': val_metrics['loss']},
                filepath=checkpoint_path,
            )
            print(f"New best model saved (val_loss: {best_val_loss:.4f})")
        
        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            print(f"Learning rate: {current_lr:.6f}")
            log_metrics_wandb({'learning_rate': current_lr}, step=epoch)
        
        # Early stopping
        if early_stopping(val_metrics['loss']):
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break
        
        # Save periodic checkpoint
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(
                config['paths']['checkpoint_dir'],
                f'checkpoint_epoch_{epoch}.pt',
            )
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=val_metrics['loss'],
                metrics={'val_loss': val_metrics['loss']},
                filepath=checkpoint_path,
            )
    
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train bat audio classifier")
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file',
    )
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create directories
    os.makedirs(config['paths']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['paths']['log_dir'], exist_ok=True)
    
    # Train model
    train(config)


if __name__ == '__main__':
    main()
