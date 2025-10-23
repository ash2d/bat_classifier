# Bat Audio Classifier - Multi-label MIL System

A PyTorch-based multi-label audio tagging system for ultrasonic bat recordings using weak labels and Multiple-Instance Learning (MIL).

## Overview

This system classifies 5-second ultrasonic bat recordings (~384 kHz sample rate) using a CRNN (Convolutional Recurrent Neural Network) architecture combined with MIL pooling strategies. Each audio clip is labeled with a multi-hot vector indicating presence/absence of approximately 20 bat species.

## Project Structure

```
bat_classifier/
├── config.yaml         # Main configuration file
├── data.py            # Dataset and data loading utilities
├── model.py           # CRNN and MIL model architectures
├── utils.py           # Utility functions (metrics, checkpointing, wandb)
├── train.py           # Training script
├── evaluate.py        # Evaluation script
├── main.py            # Simple entry point
└── pyproject.toml     # Project dependencies
```

## Features

### Data Processing
- **BatClipDataset**: Custom dataset class for loading WAV files
- **PCEN Spectrograms**: Per-Channel Energy Normalization for ultrasonic signals
- **MIL Bag Creation**: Splits each clip into multiple instances (frames)
- **Configurable Audio Processing**: Adjustable FFT parameters, mel bins, frequency range

### Model Architecture
- **CRNN Encoder**: Convolutional layers → BatchNorm → ReLU → Pooling → BiGRU
- **MIL Pooling Head**: Three pooling strategies:
  - `linear_softmax`: Weighted pooling using softmax attention
  - `attention`: Learnable attention mechanism
  - `max`: Max pooling over instances
- **Multi-label Output**: Sigmoid activation for independent class predictions

### Training Features
- Adam optimizer with configurable weight decay
- Learning rate scheduling (ReduceLROnPlateau, StepLR, CosineAnnealing)
- BCEWithLogitsLoss with optional class weights for imbalanced data
- Early stopping based on validation F1 score
- Automatic checkpointing (best model only or all epochs)
- Weights & Biases integration for experiment tracking

### Evaluation Features
- Comprehensive metrics: precision, recall, F1 (macro/micro)
- Per-class performance metrics
- ROC-AUC computation (optional)
- Prediction export to CSV
- Results logging to wandb

## Installation

### Requirements
- Python >= 3.12
- PyTorch >= 2.0.0
- See `pyproject.toml` for full dependencies

### Setup

1. Clone the repository:
```bash
git clone https://github.com/ash2d/bat_classifier.git
cd bat_classifier
```

2. Install dependencies:
```bash
pip install -e .
```

Or install packages individually:
```bash
pip install torch torchaudio librosa numpy scipy pyyaml wandb scikit-learn soundfile pandas
```

## Configuration

### Dataset Setup

Before training, you need to prepare your dataset:

1. **Audio Files**: WAV files of ultrasonic bat recordings (5s duration, ~384 kHz sample rate)

2. **Manifest CSV Files**: Create CSV files for train/val splits with the following columns:
   - `filepath`: Relative or absolute path to WAV file
   - `labels`: Comma-separated list of species names/IDs
   
   Example `train.csv`:
   ```csv
   filepath,labels
   recordings/bat_001.wav,species_1,species_3
   recordings/bat_002.wav,species_2
   recordings/bat_003.wav,species_1,species_2,species_5
   ```

3. **Update config.yaml**:
   ```yaml
   data:
     train_manifest: "path/to/train.csv"
     val_manifest: "path/to/val.csv"
     audio_root: "path/to/audio/clips"
     classes:
       - "species_1"
       - "species_2"
       - "species_3"
       # ... add all your species
   ```

### Audio Processing Parameters

Adjust the audio processing parameters in `config.yaml` for your ultrasonic recordings:

```yaml
audio:
  sample_rate: 384000    # Match your recording sample rate
  n_mels: 128           # Number of mel frequency bins
  n_fft: 2048           # FFT size (larger = better freq resolution, worse time resolution)
  hop_length: 512       # Hop length for STFT
  f_min: 10000.0        # Minimum frequency in Hz (ultrasonic range start)
  f_max: 100000.0       # Maximum frequency in Hz (ultrasonic range end)
```

**Tips for ultrasonic audio**:
- Use larger `n_fft` (2048-4096) for better frequency resolution
- Adjust `f_min` and `f_max` to cover the bat vocalization range (typically 10-100 kHz)
- Tune PCEN parameters based on your recording conditions

### Model Configuration

Adjust model architecture in `config.yaml`:

```yaml
model:
  encoder:
    conv_layers:
      - out_channels: 32
        kernel_size: [3, 3]
        pool_size: [2, 2]
      # Add more layers for deeper architecture
    rnn_hidden_size: 256
    rnn_num_layers: 2
  
  pooling:
    pooling_type: "linear_softmax"  # or "attention" or "max"
```

### Weights & Biases Setup

To enable experiment tracking:

1. Create a Weights & Biases account at https://wandb.ai

2. Set your API key:
   ```bash
   export WANDB_API_KEY="your_api_key_here"
   ```
   
   Or add it in `train.py`:
   ```python
   import wandb
   wandb.login(key="YOUR_API_KEY_HERE")
   ```

3. Update config.yaml:
   ```yaml
   logging:
     use_wandb: true
     wandb_project: "bat-classifier"
     wandb_entity: "your-username"  # or null for default
   ```

## Usage

### Training

Run training with default config:
```bash
python train.py
```

With custom config:
```bash
python train.py --config my_config.yaml
```

The training script will:
1. Load and prepare datasets
2. Initialize the model architecture
3. Train for the specified number of epochs
4. Validate after each epoch
5. Save the best model checkpoint
6. Log metrics to wandb (if enabled)
7. Apply early stopping if no improvement

### Evaluation

Evaluate a trained model:
```bash
python evaluate.py --checkpoint checkpoints/best_model.pt
```

With custom config or manifest:
```bash
python evaluate.py --config my_config.yaml --checkpoint checkpoints/best_model.pt --manifest path/to/test.csv
```

The evaluation script will:
1. Load the trained model
2. Run inference on the evaluation set
3. Compute comprehensive metrics
4. Save predictions to CSV
5. Save metrics to JSON
6. Log results to wandb (if enabled)

## Customization Points

### Required Customizations

1. **Dataset Paths** (`config.yaml`):
   - `data.train_manifest`: Path to training CSV
   - `data.val_manifest`: Path to validation CSV
   - `data.audio_root`: Root directory for audio files

2. **Species List** (`config.yaml`):
   - `data.classes`: List of all species names (must match labels in CSV)

3. **Weights & Biases** (optional but recommended):
   - Set `WANDB_API_KEY` environment variable
   - Update `logging.wandb_project` and `logging.wandb_entity`

### Optional Customizations

1. **Audio Processing** (`config.yaml`):
   - Adjust `audio.n_fft`, `audio.hop_length` for different time/frequency resolution
   - Tune `audio.pcen.*` parameters for your recording characteristics
   - Modify `mil.bag_size` and `mil.instance_frames` for different clip segmentation

2. **Model Architecture** (`config.yaml`):
   - Add/remove convolutional layers in `model.encoder.conv_layers`
   - Change `model.encoder.rnn_hidden_size` and `model.encoder.rnn_num_layers`
   - Try different pooling strategies: "linear_softmax", "attention", "max"

3. **Training Hyperparameters** (`config.yaml`):
   - Adjust `training.learning_rate`, `training.weight_decay`
   - Modify `training.batch_size` based on GPU memory
   - Set `training.loss.class_weights` for imbalanced datasets

4. **Data Loading** (`data.py`):
   - Modify `_parse_labels()` if your label format differs
   - Adjust `_prepare_waveform()` for different audio preprocessing
   - Update path resolution in `_resolve_audio_path()` if needed

## Output Files

### During Training
- `checkpoints/best_model.pt`: Best model checkpoint based on validation F1
- `wandb/`: Wandb run logs and artifacts (if enabled)

### During Evaluation
- `predictions.csv`: Per-sample predictions with probabilities
- `evaluation_metrics.json`: Detailed metrics in JSON format

## Model Checkpoint Format

Checkpoints contain:
```python
{
    "epoch": int,
    "model_state": Dict[str, Tensor],
    "optimizer_state": Dict[str, Any],
    "scheduler_state": Optional[Dict[str, Any]],
    "metrics": Dict[str, float]
}
```

Load a checkpoint:
```python
from utils import load_checkpoint

checkpoint = load_checkpoint("checkpoints/best_model.pt")
model.load_state_dict(checkpoint.model_state)
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce `training.batch_size` in config.yaml
   - Reduce `mil.bag_size` to decrease instances per clip
   - Use smaller model architecture (fewer conv layers or smaller RNN)

2. **Audio Loading Errors**:
   - Verify audio files are valid WAV format
   - Check `data.audio_root` path is correct
   - Ensure manifest CSV has correct file paths

3. **Label Mismatch**:
   - Ensure all labels in CSV exist in `data.classes` list
   - Check label delimiter matches `data.label_delimiter`
   - Verify no trailing/leading whitespace in labels

4. **Wandb Connection Issues**:
   - Set `WANDB_API_KEY` environment variable
   - Or set `logging.use_wandb: false` to disable
   - Check internet connectivity

5. **Poor Model Performance**:
   - Verify ultrasonic frequency range (`f_min`, `f_max`) is appropriate
   - Try different pooling strategies ("attention" often works well)
   - Increase model capacity (more conv layers, larger RNN)
   - Adjust learning rate and scheduler settings
   - Add class weights if dataset is imbalanced

## Example Workflow

```bash
# 1. Prepare your dataset
# - Create train.csv and val.csv with columns: filepath, labels
# - Place audio files in a directory

# 2. Update config.yaml
# - Set data.train_manifest, data.val_manifest, data.audio_root
# - Update data.classes with your species list
# - Adjust audio processing parameters for ultrasonic range

# 3. Set up wandb (optional)
export WANDB_API_KEY="your_key_here"

# 4. Train the model
python train.py

# 5. Evaluate on validation set
python evaluate.py --checkpoint checkpoints/best_model.pt

# 6. Evaluate on test set (if available)
python evaluate.py --checkpoint checkpoints/best_model.pt --manifest path/to/test.csv
```

## Advanced Usage

### Custom Augmentations

Add audio augmentations by modifying the `BatDatasetConfig.augmentations` field:

```python
# In data.py, add your augmentation pipeline
def my_augmentation_pipeline(waveform):
    # Add noise, time stretching, pitch shifting, etc.
    return augmented_waveform

# Pass to dataset
dataset_config.augmentations = my_augmentation_pipeline
```

### Class Weights for Imbalanced Data

Calculate and set class weights in config.yaml:

```python
# Calculate weights based on class frequencies
class_counts = np.array([...])  # Count of samples per class
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum() * len(class_weights)
```

Then update config.yaml:
```yaml
training:
  loss:
    class_weights: [1.2, 0.8, 1.5, ...]  # Your calculated weights
```

### Ensemble Predictions

Train multiple models with different configurations and ensemble predictions:

```python
models = [load_model(ckpt) for ckpt in checkpoint_paths]
ensemble_probs = np.mean([predict(model, data) for model in models], axis=0)
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{bat_classifier,
  title = {Bat Audio Classifier: Multi-label MIL System},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/ash2d/bat_classifier}
}
```

## License

See LICENSE file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Contact

For questions or issues, please open an issue on GitHub or contact [your-email@example.com].
