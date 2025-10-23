"""Data loading utilities for the bat audio MIL project.

Fill in dataset-specific parameters (paths, metadata schema, augmentation pipeline)
as noted in docstrings before running training.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
import torchaudio
from torch import nn
from torch.utils.data import Dataset


@dataclass
class BatDatasetConfig:
    """Lightweight container for dataset-related settings.

    Add any additional fields that must mirror values stored in ``config.yaml``
    (e.g., ``instance_hop_frames``) so that the dataset stays in sync with the
    experiment configuration.
    """

    sample_rate: int
    n_mels: int
    bag_size: int
    instance_frames: int
    f_min: float
    f_max: float
    pcen_time_constant: float
    pcen_gain: float
    pcen_bias: float
    pcen_power: float
    pcen_eps: float
    clip_duration: float
    random_crop: bool = False
    augmentations: Optional[Callable] = None


class BatClipDataset(Dataset):
    """Dataset returning MIL bags of PCEN spectrogram instances and multi-hot labels.

    Expected manifest CSV columns (adjust as needed):
        - ``filepath``: relative or absolute path to a WAV file.
        - ``labels``: a string containing comma-separated class ids or names.

    Update ``_parse_labels`` if your label storage differs, and set the paths to
    ``manifest_path`` and ``audio_root`` in ``train.py`` / ``evaluate.py``.
    """

    def __init__(
        self,
        manifest_path: str,
        audio_root: str,
        label_to_index: Dict[str, int],
        config: BatDatasetConfig,
        label_delimiter: str = ",",
        instance_hop_frames: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.manifest_path = manifest_path  # TODO: Point to your train/val CSV.
        self.audio_root = audio_root  # TODO: Point to directory with WAV clips.
        self.label_to_index = label_to_index
        self.config = config
        self.label_delimiter = label_delimiter
        self.instance_hop_frames = instance_hop_frames or config.instance_frames
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_mels=config.n_mels,
            n_fft=2048,  # TODO: Adjust FFT size for ultrasonic range resolution.
            hop_length=512,  # TODO: Match hop_length to time resolution needs.
            f_min=config.f_min,
            f_max=config.f_max,
            center=True,
            pad_mode="reflect",
            power=1.0,
        )
        self.pcen = torchaudio.transforms.PCEN(
            sr=config.sample_rate,
            trainable=False,
            time_constant=config.pcen_time_constant,
            gain=config.pcen_gain,
            bias=config.pcen_bias,
            power=config.pcen_power,
            eps=config.pcen_eps,
        )
        self.df = pd.read_csv(self.manifest_path)
        if "filepath" not in self.df.columns:
            raise ValueError("Manifest must include a 'filepath' column.")
        if "labels" not in self.df.columns:
            raise ValueError("Manifest must include a 'labels' column.")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[index]
        audio_path = self._resolve_audio_path(row["filepath"])
        waveform, sr = torchaudio.load(audio_path)
        waveform = self._prepare_waveform(waveform, sr)
        spec = self._compute_pcen_spectrogram(waveform)
        instances = self._split_into_instances(spec)
        labels = self._parse_labels(row["labels"])
        return instances, labels

    def _resolve_audio_path(self, relative_path: str) -> str:
        # Replace with dataset-specific path handling (e.g., pathlib joining).
        return f"{self.audio_root}/{relative_path}"

    def _prepare_waveform(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if waveform.ndim > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != self.config.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=sample_rate,
                new_freq=self.config.sample_rate,
            )
        target_num_samples = int(self.config.sample_rate * self.config.clip_duration)
        if waveform.size(-1) < target_num_samples:
            pad_amount = target_num_samples - waveform.size(-1)
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        elif waveform.size(-1) > target_num_samples:
            if self.config.random_crop:
                start = torch.randint(0, waveform.size(-1) - target_num_samples + 1, (1,)).item()
                waveform = waveform[..., start : start + target_num_samples]
            else:
                waveform = waveform[..., :target_num_samples]
        return waveform

    def _compute_pcen_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        spec = self.mel_spec(waveform)
        spec = self.pcen(spec)
        spec = torch.log1p(spec)
        return spec.squeeze(0)

    def _split_into_instances(self, spec: torch.Tensor) -> torch.Tensor:
        frames = spec.unfold(dimension=-1, size=self.config.instance_frames, step=self.instance_hop_frames)
        frames = frames.transpose(0, 1)  # (time, freq, instance_frames)
        if frames.size(0) == 0:
            frames = spec[:, : self.config.instance_frames].unsqueeze(0)
        if frames.size(0) > self.config.bag_size:
            indices = torch.randperm(frames.size(0))[: self.config.bag_size]
            frames = frames.index_select(0, indices)
        elif frames.size(0) < self.config.bag_size:
            pad_frames = self.config.bag_size - frames.size(0)
            padding = torch.zeros((pad_frames, frames.size(1), frames.size(2)))
            frames = torch.cat([frames, padding], dim=0)
        return frames

    def _parse_labels(self, label_entry: str) -> torch.Tensor:
        multi_hot = torch.zeros(len(self.label_to_index), dtype=torch.float32)
        for label in label_entry.split(self.label_delimiter):
            label = label.strip()
            if not label:
                continue
            if label not in self.label_to_index:
                continue  # TODO: Decide how to handle unknown labels.
            multi_hot[self.label_to_index[label]] = 1.0
        return multi_hot


def bat_collate_fn(
    batch: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    pad_to_bag_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate variable-length MIL bags, padding as necessary.

    Returns ``(instances, labels, lengths)`` where ``instances`` has shape
    ``(batch, bag_size, freq, frames)``.
    """

    instances, labels = zip(*batch)
    lengths = torch.tensor([inst.size(0) for inst in instances], dtype=torch.long)
    target_bag = pad_to_bag_size or max(lengths.max().item(), 1)
    padded_instances: List[torch.Tensor] = []
    for bag in instances:
        if bag.size(0) < target_bag:
            pad_amount = target_bag - bag.size(0)
            padding = torch.zeros((pad_amount, bag.size(1), bag.size(2)))
            bag = torch.cat([bag, padding], dim=0)
        elif bag.size(0) > target_bag:
            bag = bag[:target_bag]
        padded_instances.append(bag)
    batch_instances = torch.stack(padded_instances)
    batch_labels = torch.stack(labels)
    return batch_instances, batch_labels, lengths
