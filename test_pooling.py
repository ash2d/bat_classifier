"""
Test script to verify all pooling types work correctly.
"""

import torch
from utils import load_config, set_seed
from model import create_model


def test_pooling_type(pooling_type: str):
    """Test a specific pooling type."""
    print(f"\nTesting pooling type: {pooling_type}")
    print("-" * 50)
    
    # Load config and modify pooling type
    config = load_config('config.yaml')
    config['model']['pooling_type'] = pooling_type
    
    # Set seed
    set_seed(42)
    
    # Create model
    model = create_model(config)
    model.eval()
    
    # Create dummy input
    batch_size = 2
    bag_size = 10
    feature_dim = 1000
    dummy_instances = torch.randn(batch_size, bag_size, feature_dim)
    
    # Forward pass
    with torch.no_grad():
        logits = model(dummy_instances)
        probs = torch.sigmoid(logits)
    
    print(f"✓ Model created successfully")
    print(f"✓ Input shape: {dummy_instances.shape}")
    print(f"✓ Output shape: {logits.shape}")
    print(f"✓ Output range: [{probs.min().item():.4f}, {probs.max().item():.4f}]")
    print(f"✓ Mean probability: {probs.mean().item():.4f}")
    
    return True


def main():
    """Test all pooling types."""
    print("=" * 60)
    print("Testing All Pooling Types")
    print("=" * 60)
    
    pooling_types = ['linear_softmax', 'attention', 'max']
    
    results = {}
    for pooling_type in pooling_types:
        try:
            success = test_pooling_type(pooling_type)
            results[pooling_type] = 'PASS' if success else 'FAIL'
        except Exception as e:
            print(f"✗ Error: {e}")
            results[pooling_type] = 'FAIL'
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for pooling_type, result in results.items():
        status = "✓" if result == "PASS" else "✗"
        print(f"{status} {pooling_type}: {result}")
    
    all_passed = all(result == 'PASS' for result in results.values())
    if all_passed:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed!")
    
    return all_passed


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
