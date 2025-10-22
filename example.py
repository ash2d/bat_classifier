"""
Example script demonstrating how to use the bat classifier.

This script shows:
1. How to load configuration
2. How to create a model
3. How to create dataloaders (with dummy data)
4. Basic training and evaluation workflow
"""

import torch
import numpy as np
from utils import load_config, get_device, set_seed
from model import create_model


def main():
    """Main example function."""
    print("="*60)
    print("Bat Audio Classifier - Example Usage")
    print("="*60)
    
    # 1. Load configuration
    print("\n1. Loading configuration from config.yaml...")
    config = load_config('config.yaml')
    print(f"   Configuration loaded successfully!")
    print(f"   - Sample rate: {config['data']['sample_rate']} Hz")
    print(f"   - Number of classes: {config['data']['num_classes']}")
    print(f"   - Pooling type: {config['model']['pooling_type']}")
    
    # 2. Set random seed for reproducibility
    print("\n2. Setting random seed for reproducibility...")
    set_seed(config['seed'])
    print(f"   Random seed set to: {config['seed']}")
    
    # 3. Get device
    print("\n3. Detecting available device...")
    device = get_device(config['device'])
    print(f"   Using device: {device}")
    
    # 4. Create model
    print("\n4. Creating BatMILModel...")
    model = create_model(config)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Model created successfully!")
    print(f"   - Total parameters: {total_params:,}")
    print(f"   - Trainable parameters: {trainable_params:,}")
    
    # 5. Create dummy input to test model
    print("\n5. Testing model with dummy input...")
    batch_size = 2
    bag_size = config['data']['bag_size']
    feature_dim = 1000  # Dummy feature dimension
    
    # Create dummy instances
    dummy_instances = torch.randn(batch_size, bag_size, feature_dim).to(device)
    print(f"   Input shape: {dummy_instances.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(dummy_instances)
        probs = torch.sigmoid(logits)
    
    print(f"   Output logits shape: {logits.shape}")
    print(f"   Output probabilities shape: {probs.shape}")
    print(f"   Example probabilities (first sample): {probs[0, :5].cpu().numpy()}")
    
    # 6. Show configuration options
    print("\n6. Configuration Options:")
    print("\n   Pooling Types Available:")
    print("   - 'linear_softmax': Linear layer with softmax attention")
    print("   - 'attention': Learnable attention mechanism")
    print("   - 'max': Max pooling across instances")
    print(f"\n   Current pooling type: {config['model']['pooling_type']}")
    
    print("\n   To change pooling type, edit config.yaml:")
    print("   model:")
    print("     pooling_type: 'attention'  # Change this")
    
    # 7. Show training workflow
    print("\n7. Training Workflow:")
    print("   To train the model on your data:")
    print("   a. Prepare your data:")
    print("      - WAV files in a directory (e.g., ./data/audio/)")
    print("      - CSV files with labels (train.csv, val.csv, test.csv)")
    print("      - Format: filename,species1,species2,...,species20")
    print("   b. Update paths in config.yaml:")
    print("      paths:")
    print("        data_dir: './data/audio'")
    print("        train_csv: './data/train.csv'")
    print("   c. Run training:")
    print("      python train.py --config config.yaml")
    
    # 8. Show evaluation workflow
    print("\n8. Evaluation Workflow:")
    print("   To evaluate a trained model:")
    print("   python evaluate.py --config config.yaml --checkpoint checkpoints/best_model.pt")
    
    print("\n" + "="*60)
    print("Example completed successfully!")
    print("="*60)
    
    # 9. Additional info
    print("\nFor more information:")
    print("- See README.md for detailed documentation")
    print("- Check config.yaml for all configuration options")
    print("- Review data.py for custom dataset implementation")
    print("- Review model.py for CRNN and MIL architecture")


if __name__ == '__main__':
    main()
