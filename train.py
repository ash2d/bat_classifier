"""Training script for the bat audio MIL classifier.

This script handles:
- Loading configuration from config.yaml
- Initializing datasets and dataloaders
- Building the model architecture
- Setting up optimizer, scheduler, and loss function
- Training loop with validation
- Logging to Weights & Biases (wandb)
- Checkpointing best models

Before running:
1. Set dataset paths in config.yaml (train_manifest, val_manifest, audio_root)
2. Update the classes list in config.yaml with your actual species names
3. Set WANDB_API_KEY environment variable or uncomment wandb.login() below
4. Adjust hyperparameters in config.yaml as needed
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml
from torch import nn, optim
from torch.utils.data import DataLoader

from data import BatClipDataset, BatDatasetConfig, bat_collate_fn
from model import CRNN, BatMILModel, CRNNConfig, MILPoolingHead
from utils import (
    Checkpoint,
    compute_classification_metrics,
    init_wandb,
    save_checkpoint,
    set_seed,
)


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
        random_crop=mil_cfg["random_crop"],
    )


def create_dataloaders(
    config: Dict[str, Any],
    dataset_config: BatDatasetConfig,
) -> tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders."""
    data_cfg = config["data"]
    system_cfg = config["system"]
    training_cfg = config["training"]
    
    # Create label mapping
    label_to_index = {label: idx for idx, label in enumerate(data_cfg["classes"])}
    
    # Create datasets
    train_dataset = BatClipDataset(
        manifest_path=data_cfg["train_manifest"],
        audio_root=data_cfg["audio_root"],
        label_to_index=label_to_index,
        config=dataset_config,
        label_delimiter=data_cfg["label_delimiter"],
        instance_hop_frames=config["mil"]["instance_hop_frames"],
    )
    
    val_dataset = BatClipDataset(
        manifest_path=data_cfg["val_manifest"],
        audio_root=data_cfg["audio_root"],
        label_to_index=label_to_index,
        config=dataset_config,
        label_delimiter=data_cfg["label_delimiter"],
        instance_hop_frames=config["mil"]["instance_hop_frames"],
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_cfg["batch_size"],
        shuffle=True,
        num_workers=system_cfg["num_workers"],
        pin_memory=system_cfg["pin_memory"],
        collate_fn=lambda batch: bat_collate_fn(batch, pad_to_bag_size=dataset_config.bag_size),
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["evaluation"]["batch_size"],
        shuffle=False,
        num_workers=system_cfg["num_workers"],
        pin_memory=system_cfg["pin_memory"],
        collate_fn=lambda batch: bat_collate_fn(batch, pad_to_bag_size=dataset_config.bag_size),
    )
    
    return train_loader, val_loader


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


def create_optimizer_and_scheduler(
    model: nn.Module,
    config: Dict[str, Any],
) -> tuple[optim.Optimizer, Optional[Any]]:
    """Create optimizer and learning rate scheduler."""
    training_cfg = config["training"]
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=training_cfg["learning_rate"],
        weight_decay=training_cfg["weight_decay"],
    )
    
    scheduler_cfg = training_cfg["scheduler"]
    scheduler = None
    
    if scheduler_cfg["type"] == "ReduceLROnPlateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=scheduler_cfg["factor"],
            patience=scheduler_cfg["patience"],
            min_lr=scheduler_cfg["min_lr"],
            verbose=True,
        )
    elif scheduler_cfg["type"] == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_cfg.get("step_size", 30),
            gamma=scheduler_cfg["factor"],
        )
    elif scheduler_cfg["type"] == "CosineAnnealingLR":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=training_cfg["num_epochs"],
            eta_min=scheduler_cfg["min_lr"],
        )
    
    return optimizer, scheduler


def create_loss_function(config: Dict[str, Any], device: torch.device) -> nn.Module:
    """Create loss function with optional class weights."""
    loss_cfg = config["training"]["loss"]
    
    if loss_cfg["class_weights"] is not None:
        class_weights = torch.tensor(loss_cfg["class_weights"], dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    return criterion


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    config: Dict[str, Any],
    wandb_run: Optional[Any] = None,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    log_interval = config["logging"]["log_interval"]
    
    all_targets = []
    all_probs = []
    
    for batch_idx, (instances, labels, lengths) in enumerate(train_loader):
        instances = instances.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(instances, lengths)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Store predictions for metrics
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        targets = labels.detach().cpu().numpy()
        all_targets.append(targets)
        all_probs.append(probs)
        
        # Log batch metrics
        if (batch_idx + 1) % log_interval == 0 and config["logging"]["verbose"]:
            avg_loss = running_loss / (batch_idx + 1)
            print(f"Epoch {epoch} [{batch_idx + 1}/{len(train_loader)}] Loss: {avg_loss:.4f}")
    
    # Compute epoch metrics
    epoch_loss = running_loss / len(train_loader)
    all_targets = np.vstack(all_targets)
    all_probs = np.vstack(all_probs)
    metrics = compute_classification_metrics(all_targets, all_probs)
    
    results = {
        "train_loss": epoch_loss,
        "train_macro_f1": metrics["macro_f1"],
        "train_micro_f1": metrics["micro_f1"],
        "train_macro_precision": metrics["macro_precision"],
        "train_macro_recall": metrics["macro_recall"],
    }
    
    # Log to wandb
    if wandb_run is not None:
        wandb_run.log({"epoch": epoch, **results})
    
    return results


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    config: Dict[str, Any],
    wandb_run: Optional[Any] = None,
) -> Dict[str, float]:
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for instances, labels, lengths in val_loader:
            instances = instances.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)
            
            # Forward pass
            logits = model(instances, lengths)
            loss = criterion(logits, labels)
            
            running_loss += loss.item()
            
            # Store predictions for metrics
            probs = torch.sigmoid(logits).cpu().numpy()
            targets = labels.cpu().numpy()
            all_targets.append(targets)
            all_probs.append(probs)
    
    # Compute epoch metrics
    epoch_loss = running_loss / len(val_loader)
    all_targets = np.vstack(all_targets)
    all_probs = np.vstack(all_probs)
    metrics = compute_classification_metrics(all_targets, all_probs)
    
    results = {
        "val_loss": epoch_loss,
        "val_macro_f1": metrics["macro_f1"],
        "val_micro_f1": metrics["micro_f1"],
        "val_macro_precision": metrics["macro_precision"],
        "val_macro_recall": metrics["macro_recall"],
        "val_roc_auc": metrics["roc_auc_macro"],
    }
    
    # Log to wandb
    if wandb_run is not None:
        wandb_run.log({"epoch": epoch, **results})
    
    if config["logging"]["verbose"]:
        print(f"Validation - Loss: {epoch_loss:.4f}, Macro F1: {metrics['macro_f1']:.4f}, "
              f"Micro F1: {metrics['micro_f1']:.4f}")
    
    return results


def train(config: Dict[str, Any]) -> None:
    """Main training function."""
    # Set seed for reproducibility
    set_seed(config["system"]["seed"])
    
    # Set device
    device_str = config["system"]["device"]
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device_str = "cpu"
    device = torch.device(device_str)
    
    # Initialize wandb
    # TODO: Set WANDB_API_KEY environment variable or uncomment the line below:
    # import wandb
    # wandb.login(key="YOUR_WANDB_API_KEY_HERE")
    
    wandb_run = None
    if config["logging"]["use_wandb"]:
        try:
            wandb_run = init_wandb(
                project=config["logging"]["wandb_project"],
                entity=config["logging"]["wandb_entity"],
                config=config,
            )
            print(f"Initialized wandb run: {wandb_run.name}")
        except Exception as e:
            print(f"Failed to initialize wandb: {e}")
            print("Continuing without wandb logging...")
    
    # Create dataset config
    dataset_config = create_dataset_config(config)
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(config, dataset_config)
    print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")
    
    # Create model
    num_classes = len(config["data"]["classes"])
    print(f"Creating model for {num_classes} classes...")
    model = create_model(config, num_classes, device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(model, config)
    
    # Create loss function
    criterion = create_loss_function(config, device)
    
    # Create checkpoint directory
    checkpoint_dir = Path(config["training"]["checkpoint_dir"])
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Training loop
    best_val_f1 = 0.0
    epochs_without_improvement = 0
    num_epochs = config["training"]["num_epochs"]
    early_stopping_cfg = config["training"]["early_stopping"]
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print("=" * 80)
    
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 80)
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, config, wandb_run
        )
        
        # Validate
        val_metrics = validate(
            model, val_loader, criterion, device, epoch, config, wandb_run
        )
        
        # Learning rate scheduler step
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics["val_macro_f1"])
            else:
                scheduler.step()
        
        # Print epoch summary
        print(f"Train Loss: {train_metrics['train_loss']:.4f}, "
              f"Train Macro F1: {train_metrics['train_macro_f1']:.4f}")
        print(f"Val Loss: {val_metrics['val_loss']:.4f}, "
              f"Val Macro F1: {val_metrics['val_macro_f1']:.4f}")
        
        # Save checkpoint
        current_val_f1 = val_metrics["val_macro_f1"]
        improved = current_val_f1 > best_val_f1 + early_stopping_cfg["min_delta"]
        
        if improved:
            best_val_f1 = current_val_f1
            epochs_without_improvement = 0
            
            if config["training"]["save_best_only"]:
                checkpoint_path = checkpoint_dir / "best_model.pt"
                checkpoint = Checkpoint(
                    epoch=epoch,
                    model_state=model.state_dict(),
                    optimizer_state=optimizer.state_dict(),
                    scheduler_state=scheduler.state_dict() if scheduler is not None else None,
                    metrics={**train_metrics, **val_metrics},
                )
                save_checkpoint(checkpoint, checkpoint_path)
                print(f"✓ Saved best model (Val Macro F1: {best_val_f1:.4f})")
                
                # Log checkpoint to wandb
                if wandb_run is not None:
                    try:
                        import wandb
                        wandb.save(str(checkpoint_path))
                    except Exception as e:
                        print(f"Failed to upload checkpoint to wandb: {e}")
        else:
            epochs_without_improvement += 1
        
        # Early stopping
        if early_stopping_cfg["enabled"]:
            if epochs_without_improvement >= early_stopping_cfg["patience"]:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                print(f"Best Val Macro F1: {best_val_f1:.4f}")
                break
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Best Val Macro F1: {best_val_f1:.4f}")
    
    if wandb_run is not None:
        wandb_run.finish()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train bat audio MIL classifier")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)",
    )
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Run training
    train(config)


if __name__ == "__main__":
    main()
