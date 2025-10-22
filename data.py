"""
Data loading and preprocessing for bat audio classification.
"""

import os
import pandas as pd
import numpy as np
import torch
import torchaudio
import librosa
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, List


class BatClipDataset(Dataset):
    """
    Custom Dataset class for bat audio clips with weak multi-label supervision.
    
    Each audio clip is treated as a bag of instances (frames) for Multiple-Instance Learning.
    """
    
    def __init__(
        self,
        csv_path: str,
        data_dir: str,
        config: dict,
        transform: Optional[callable] = None,
        train: bool = True,
    ):
        """
        Args:
            csv_path: Path to CSV file with columns [filename, species1, species2, ...]
            data_dir: Root directory containing audio files
            config: Configuration dictionary
            transform: Optional transform to apply to spectrograms
            train: Whether this is training data (for augmentation)
        """
        self.data_dir = data_dir
        self.config = config
        self.transform = transform
        self.train = train
        
        # Load CSV with file paths and labels
        self.df = pd.read_csv(csv_path)
        
        # Extract configuration parameters
        self.sample_rate = config['data']['sample_rate']
        self.n_mels = config['data']['n_mels']
        self.n_fft = config['data']['n_fft']
        self.hop_length = config['data']['hop_length']
        self.fmin = config['data']['fmin']
        self.fmax = config['data']['fmax']
        self.clip_duration = config['data']['clip_duration']
        self.bag_size = config['data']['bag_size']
        self.num_classes = config['data']['num_classes']
        
        # Augmentation parameters
        self.augmentation = config.get('augmentation', {})
        
    def __len__(self) -> int:
        """Return the number of clips in the dataset."""
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single clip and its labels.
        
        Args:
            idx: Index of the clip
            
        Returns:
            Tuple of (instances_tensor, label_tensor)
            - instances_tensor: (T, F) where T is number of instances, F is feature dimension
            - label_tensor: (C,) multi-hot label vector
        """
        # Get file path and labels
        row = self.df.iloc[idx]
        filename = row['filename']
        filepath = os.path.join(self.data_dir, filename)
        
        # Extract labels (assuming columns after 'filename' are species labels)
        label_columns = [col for col in self.df.columns if col != 'filename']
        labels = row[label_columns].values.astype(np.float32)
        label_tensor = torch.from_numpy(labels)
        
        # Load audio
        waveform, sr = torchaudio.load(filepath)
        
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Ensure fixed duration
        target_length = int(self.clip_duration * self.sample_rate)
        if waveform.shape[1] < target_length:
            # Pad if too short
            padding = target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif waveform.shape[1] > target_length:
            # Truncate if too long
            waveform = waveform[:, :target_length]
        
        # Compute mel spectrogram
        mel_spec = self._compute_mel_spectrogram(waveform)
        
        # Compute PCEN (Per-Channel Energy Normalization)
        pcen_spec = self._compute_pcen(mel_spec)
        
        # Apply augmentation if training
        if self.train:
            pcen_spec = self._apply_augmentation(pcen_spec)
        
        # Split into instances (frames)
        instances = self._split_into_instances(pcen_spec)
        
        # Apply optional transform
        if self.transform is not None:
            instances = self.transform(instances)
        
        return instances, label_tensor
    
    def _compute_mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Compute mel spectrogram from waveform.
        
        Args:
            waveform: Audio waveform tensor (1, L)
            
        Returns:
            Mel spectrogram tensor (n_mels, T)
        """
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_min=self.fmin,
            f_max=self.fmax,
        )
        
        mel_spec = mel_transform(waveform)
        mel_spec = mel_spec.squeeze(0)  # Remove channel dimension
        
        # Convert to log scale
        mel_spec = torch.log(mel_spec + 1e-9)
        
        return mel_spec
    
    def _compute_pcen(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        Compute Per-Channel Energy Normalization (PCEN).
        
        PCEN is a robust frontend for neural network audio models.
        For simplicity, this is a placeholder - you can implement full PCEN here.
        
        Args:
            mel_spec: Mel spectrogram (n_mels, T)
            
        Returns:
            PCEN-normalized spectrogram (n_mels, T)
        """
        # Placeholder: Simple normalization
        # Full PCEN involves adaptive gain control and compression
        # You can implement librosa.pcen or torchaudio.functional.pcen here
        
        # For now, apply per-frequency normalization
        mean = mel_spec.mean(dim=1, keepdim=True)
        std = mel_spec.std(dim=1, keepdim=True)
        pcen_spec = (mel_spec - mean) / (std + 1e-9)
        
        return pcen_spec
    
    def _apply_augmentation(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Apply data augmentation to spectrogram.
        
        Args:
            spec: Spectrogram tensor (F, T)
            
        Returns:
            Augmented spectrogram (F, T)
        """
        # Time masking
        if self.augmentation.get('time_masking', False):
            time_mask_param = self.augmentation.get('time_mask_param', 20)
            time_mask = torchaudio.transforms.TimeMasking(time_mask_param)
            spec = spec.unsqueeze(0)  # Add channel dimension
            spec = time_mask(spec)
            spec = spec.squeeze(0)
        
        # Frequency masking
        if self.augmentation.get('freq_masking', False):
            freq_mask_param = self.augmentation.get('freq_mask_param', 10)
            freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param)
            spec = spec.unsqueeze(0)  # Add channel dimension
            spec = freq_mask(spec)
            spec = spec.squeeze(0)
        
        # Noise injection
        if self.augmentation.get('noise_injection', False):
            noise_std = self.augmentation.get('noise_std', 0.01)
            noise = torch.randn_like(spec) * noise_std
            spec = spec + noise
        
        return spec
    
    def _split_into_instances(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Split spectrogram into fixed number of instances (frames).
        
        Args:
            spec: Spectrogram tensor (F, T)
            
        Returns:
            Instances tensor (bag_size, F * instance_width)
        """
        freq_bins, time_bins = spec.shape
        
        # Calculate instance width
        instance_width = time_bins // self.bag_size
        
        instances = []
        for i in range(self.bag_size):
            start = i * instance_width
            end = start + instance_width
            
            # Handle last instance
            if i == self.bag_size - 1:
                end = time_bins
            
            instance = spec[:, start:end]
            
            # Flatten to 1D feature vector
            instance_flat = instance.flatten()
            instances.append(instance_flat)
        
        # Stack instances
        instances_tensor = torch.stack(instances)  # (bag_size, F * instance_width)
        
        return instances_tensor


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Custom collate function to handle variable-length instances.
    
    Pads or truncates instances to ensure uniform batch size.
    
    Args:
        batch: List of (instances, labels) tuples
        
    Returns:
        Tuple of (batched_instances, batched_labels)
        - batched_instances: (B, T, F)
        - batched_labels: (B, C)
    """
    instances_list, labels_list = zip(*batch)
    
    # Find maximum feature dimension in batch
    max_features = max(inst.shape[1] for inst in instances_list)
    bag_size = instances_list[0].shape[0]
    
    # Pad instances to same feature dimension
    padded_instances = []
    for instances in instances_list:
        if instances.shape[1] < max_features:
            padding = max_features - instances.shape[1]
            instances = torch.nn.functional.pad(instances, (0, padding))
        elif instances.shape[1] > max_features:
            instances = instances[:, :max_features]
        padded_instances.append(instances)
    
    # Stack into batch
    batched_instances = torch.stack(padded_instances)  # (B, T, F)
    batched_labels = torch.stack(labels_list)  # (B, C)
    
    return batched_instances, batched_labels


def create_dataloader(
    csv_path: str,
    data_dir: str,
    config: dict,
    train: bool = True,
    shuffle: bool = True,
) -> DataLoader:
    """
    Create DataLoader for bat audio dataset.
    
    Args:
        csv_path: Path to CSV file
        data_dir: Root directory for audio files
        config: Configuration dictionary
        train: Whether this is training data
        shuffle: Whether to shuffle data
        
    Returns:
        DataLoader object
    """
    dataset = BatClipDataset(
        csv_path=csv_path,
        data_dir=data_dir,
        config=config,
        train=train,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=shuffle,
        num_workers=config.get('num_workers', 4),
        collate_fn=collate_fn,
        pin_memory=True if config['device'] == 'cuda' else False,
    )
    
    return dataloader
