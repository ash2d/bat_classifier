# Quick Start Guide - Bat Audio Classifier

This guide provides a step-by-step checklist to get started quickly.

## Prerequisites Checklist

- [ ] Python >= 3.12 installed
- [ ] Audio dataset prepared (WAV files, ~384 kHz, 5 seconds each)
- [ ] Labels prepared for each audio clip
- [ ] (Optional) Weights & Biases account created

## Setup Steps

### 1. Install Dependencies

```bash
# Option A: Using pip
pip install torch torchaudio librosa numpy scipy pyyaml wandb scikit-learn soundfile pandas

# Option B: If you have a requirements.txt
pip install -r requirements.txt
```

### 2. Prepare Your Dataset

Create two CSV files: `train.csv` and `val.csv`

**Required columns:**
- `filepath`: Path to WAV file (relative to audio_root or absolute)
- `labels`: Comma-separated species names

**Example:**
```csv
filepath,labels
recordings/bat_001.wav,species_1,species_3
recordings/bat_002.wav,species_2
```

See `example_manifest.csv` for more details.

### 3. Configure the System

Edit `config.yaml`:

**Essential configurations:**
```yaml
data:
  train_manifest: "path/to/train.csv"        # ← Change this
  val_manifest: "path/to/val.csv"            # ← Change this
  audio_root: "path/to/audio/directory"      # ← Change this
  
  classes:                                    # ← Update with your species
    - "Myotis_daubentonii"
    - "Pipistrellus_pipistrellus"
    # ... add all your species
```

**Audio processing (adjust for your recordings):**
```yaml
audio:
  sample_rate: 384000      # Match your recording sample rate
  f_min: 10000.0           # Minimum frequency in Hz
  f_max: 100000.0          # Maximum frequency in Hz
```

### 4. Set Up Wandb (Optional but Recommended)

```bash
# Get your API key from https://wandb.ai/settings
export WANDB_API_KEY="your_api_key_here"
```

Or disable wandb in `config.yaml`:
```yaml
logging:
  use_wandb: false
```

### 5. Train the Model

```bash
python train.py
```

The script will:
- Load and validate your dataset
- Initialize the model
- Train for the specified epochs
- Save the best model to `checkpoints/best_model.pt`
- Log metrics to wandb (if enabled)

### 6. Evaluate the Model

```bash
python evaluate.py --checkpoint checkpoints/best_model.pt
```

Results will be saved to:
- `predictions.csv` - Per-sample predictions
- `evaluation_metrics.json` - Detailed metrics

## Common Adjustments

### If Training is Slow
```yaml
training:
  batch_size: 4          # Reduce from default 8
  
system:
  num_workers: 2         # Reduce from default 4
```

### If GPU Memory is Low
```yaml
mil:
  bag_size: 16           # Reduce from default 32

model:
  encoder:
    rnn_hidden_size: 128  # Reduce from default 256
```

### For Imbalanced Datasets

Calculate class weights:
```python
import numpy as np
import pandas as pd

# Count samples per class in your training set
df = pd.read_csv("train.csv")
class_counts = {}  # Count positive samples for each class

# Calculate weights (inverse frequency)
weights = [1.0 / count for count in class_counts.values()]
```

Add to `config.yaml`:
```yaml
training:
  loss:
    class_weights: [1.2, 0.8, 1.5, ...]  # Your calculated weights
```

## Validation Checklist

Before training, verify:

- [ ] All audio files can be loaded (check with `torchaudio.load()`)
- [ ] All labels in CSV exist in `config.yaml` classes list
- [ ] File paths are correct (relative to audio_root or absolute)
- [ ] Audio sample rate matches `config.yaml`
- [ ] Sufficient disk space for checkpoints (~100-500 MB per checkpoint)
- [ ] GPU available (if using CUDA): `torch.cuda.is_available()`

## Monitoring Training

### Console Output
Training progress is printed every 10 batches (configurable):
```
Epoch 1/100
--------------------------------------------------------------------------------
Epoch 1 [10/50] Loss: 0.6234
Epoch 1 [20/50] Loss: 0.5987
...
Train Loss: 0.5432, Train Macro F1: 0.3456
Val Loss: 0.5876, Val Macro F1: 0.3012
✓ Saved best model (Val Macro F1: 0.3012)
```

### Wandb Dashboard
If enabled, view real-time metrics at: https://wandb.ai/your-username/bat-classifier

Metrics logged:
- Training/validation loss
- Macro/micro F1, precision, recall
- Learning rate
- Per-epoch progress

## Troubleshooting Quick Fixes

### "FileNotFoundError: [Errno 2] No such file or directory"
- Check `data.train_manifest` path in config.yaml
- Verify `data.audio_root` is correct
- Ensure file paths in CSV are correct

### "ValueError: Manifest must include a 'filepath' column"
- Add `filepath,labels` header as first line of CSV
- Check for typos in column names

### "CUDA out of memory"
- Reduce `training.batch_size` in config.yaml
- Reduce `mil.bag_size` in config.yaml
- Use smaller model (fewer conv layers)

### "RuntimeError: Expected all tensors to be on the same device"
- Set `system.device: "cpu"` in config.yaml if GPU issues persist

### No improvement during training
- Check that labels are correct (not all zeros)
- Verify frequency range covers bat vocalizations
- Increase model capacity (more layers/hidden units)
- Adjust learning rate
- Try different pooling strategy (attention often works well)

## Next Steps

1. **Hyperparameter Tuning**: Experiment with different:
   - Learning rates (0.0001, 0.0005, 0.001)
   - Pooling types (linear_softmax, attention, max)
   - Model architectures (add/remove conv layers)
   - Batch sizes

2. **Data Augmentation**: Add audio augmentations for better generalization

3. **Ensemble Models**: Train multiple models and average predictions

4. **Error Analysis**: Examine misclassified samples in predictions.csv

## Support

- Check README.md for detailed documentation
- Open an issue on GitHub for bugs
- Review config.yaml comments for parameter descriptions

## Quick Command Reference

```bash
# Train with default config
python train.py

# Train with custom config
python train.py --config my_config.yaml

# Evaluate on validation set
python evaluate.py --checkpoint checkpoints/best_model.pt

# Evaluate on test set
python evaluate.py --checkpoint checkpoints/best_model.pt --manifest test.csv

# Check Python/package versions
python --version
python -c "import torch; print('PyTorch:', torch.__version__)"
```
