# Bat Audio Classifier

Multi-label audio tagging system for ultrasonic bat recordings using PyTorch and Multiple-Instance Learning (MIL).

## Overview

This system classifies ultrasonic bat recordings (5-second clips at ~384 kHz sample rate) into ~20 species using weak multi-label supervision. Each audio clip is treated as a "bag" of instances (frames) using Multiple-Instance Learning, where the model learns to identify which instances contain species-specific patterns.

## Features

- **Custom Dataset**: BatClipDataset with PCEN spectrogram computation for ultrasonic frequencies (10-100 kHz)
- **CRNN Architecture**: Convolutional layers + BatchNorm + ReLU + Pooling + BiGRU for feature extraction
- **MIL Pooling**: Supports three pooling strategies:
  - `linear_softmax`: Linear layer with softmax attention weights
  - `attention`: Learnable attention mechanism
  - `max`: Max pooling across instances
- **Training**: Complete training loop with Adam optimizer, learning rate scheduling, early stopping, and wandb logging
- **Evaluation**: Comprehensive metrics including precision, recall, F1-score, ROC-AUC, and Average Precision

## Installation

Install dependencies using pip:

```bash
pip install -e .
```

Or install from pyproject.toml:

```bash
pip install torch torchaudio librosa numpy scipy pyyaml wandb scikit-learn soundfile
```

## Project Structure

```
bat_classifier/
├── config.yaml       # Configuration file with all parameters
├── data.py          # BatClipDataset class and data loading
├── model.py         # CRNN, MILPoolingHead, and BatMILModel
├── train.py         # Training script with optimizer and logging
├── evaluate.py      # Evaluation script with metrics
├── utils.py         # Utility functions (seed, metrics, checkpoints, wandb)
└── main.py          # Original hello world (can be removed or repurposed)
```

## Configuration

Edit `config.yaml` to customize:

### Data Parameters
- `sample_rate`: 384000 Hz (384 kHz for ultrasonic recordings)
- `n_mels`: Number of mel filterbanks (default: 128)
- `fmin`, `fmax`: Frequency range for ultrasonic band (10-100 kHz)
- `bag_size`: Number of instances per clip (default: 10)
- `num_classes`: Number of bat species (default: 20)

### Model Parameters
- `pooling_type`: "linear_softmax", "attention", or "max"
- `conv_layers`: List of convolutional layer configurations
- `rnn_hidden_size`: Hidden size for BiGRU (default: 128)
- `dropout`: Dropout rate (default: 0.5)

### Training Parameters
- `batch_size`: Batch size (default: 8)
- `epochs`: Number of training epochs (default: 100)
- `learning_rate`: Initial learning rate (default: 0.001)
- `weight_decay`: L2 regularization (default: 0.0001)
- `class_weights`: Optional class weights for imbalanced data

### Data Paths
Update paths in config.yaml:
- `data_dir`: Directory containing WAV files
- `train_csv`, `val_csv`, `test_csv`: CSV files with format:
  ```
  filename,species1,species2,...,species20
  clip001.wav,1,0,1,...,0
  clip002.wav,0,1,0,...,1
  ```

## Usage

### Training

```bash
python train.py --config config.yaml
```

The training script will:
1. Load data from CSV files and audio directory
2. Initialize model with configured architecture
3. Train with Adam optimizer and learning rate scheduling
4. Log metrics to Weights & Biases (if enabled)
5. Save best model checkpoint to `checkpoints/best_model.pt`
6. Save periodic checkpoints every 10 epochs

### Evaluation

```bash
python evaluate.py --config config.yaml --checkpoint checkpoints/best_model.pt
```

The evaluation script will:
1. Load trained model from checkpoint
2. Evaluate on test set
3. Compute comprehensive metrics:
   - Macro/micro/weighted precision, recall, F1-score
   - Per-class metrics
   - ROC-AUC scores (if applicable)
   - Average Precision scores
4. Log results to Weights & Biases
5. Save predictions to `logs/test_predictions.npz`

## Data Format

### Audio Files
- Format: WAV files
- Sample rate: ~384 kHz (ultrasonic)
- Duration: 5 seconds
- Location: Organized in `data_dir`

### CSV Files
CSV files should have the following structure:
```csv
filename,species1,species2,species3,...,species20
audio/clip001.wav,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
audio/clip002.wav,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
```

- First column: `filename` - path to WAV file (relative to `data_dir`)
- Remaining columns: Binary labels (0 or 1) for each species

## Customization

### PCEN Computation
The `_compute_pcen()` method in `data.py` currently uses simple normalization. For full PCEN:

```python
import librosa
pcen_spec = librosa.pcen(mel_spec.numpy(), sr=self.sample_rate)
```

### Spectrogram Parameters
Adjust ultrasonic processing parameters in `config.yaml`:
- `n_fft`: FFT window size (affects frequency resolution)
- `hop_length`: Hop length (affects time resolution)
- `n_mels`: Number of mel bands
- `fmin`, `fmax`: Frequency range for bat calls

### Convolutional Layers
Modify `conv_layers` in `config.yaml` to add/remove layers:
```yaml
conv_layers:
  - filters: 64
    kernel_size: [3, 3]
    stride: [1, 1]
    padding: [1, 1]
  - filters: 128
    kernel_size: [3, 3]
    stride: [1, 1]
    padding: [1, 1]
```

### Augmentation
Enable/disable augmentation in `config.yaml`:
```yaml
augmentation:
  time_masking: true
  time_mask_param: 20
  freq_masking: true
  freq_mask_param: 10
  noise_injection: true
  noise_std: 0.01
```

## Weights & Biases Integration

To enable wandb logging:

1. Install wandb: `pip install wandb`
2. Login: `wandb login`
3. Configure in `config.yaml`:
```yaml
wandb:
  enabled: true
  project: "bat-classification"
  entity: "your-username"  # Your wandb username or team
  tags: ["mil", "audio", "bat", "ultrasonic"]
```

## Model Architecture

```
Input: Audio waveform (5s @ 384kHz)
  ↓
Mel Spectrogram (10-100 kHz, 128 mel bands)
  ↓
PCEN Normalization
  ↓
Split into T instances (frames)
  ↓
CRNN Feature Extractor:
  - Convolutional layers (64→128→256 filters)
  - BatchNorm + ReLU + MaxPooling
  - BiGRU (128 hidden units, 2 layers)
  ↓
Instance Features (T × 256)
  ↓
MIL Pooling Head:
  - Linear Softmax / Attention / Max pooling
  - Aggregates instance features
  ↓
Output: Class logits (20 species)
```

## Multiple-Instance Learning

The model treats each audio clip as a "bag" containing multiple instances (time frames). This is useful for:
- Handling weak labels (clip-level rather than frame-level)
- Learning which temporal segments contain species-specific calls
- Dealing with variable-length or sparse patterns within clips

Three pooling strategies are supported:
1. **Linear Softmax**: Learns class-specific attention over instances
2. **Attention**: Learns generic attention weights, then classifies aggregated features
3. **Max**: Takes maximum activation across instances (assumes at least one positive instance per bag)

## License

Add your license here.

## Citation

If you use this code, please cite appropriately.
