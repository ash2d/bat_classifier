"""Model architectures for the bat audio MIL system."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn


@dataclass
class CRNNConfig:
    """Configuration parameters for the convolutional + recurrent encoder."""

    input_channels: int
    conv_layers: Iterable[Dict[str, object]]
    rnn_hidden_size: int
    rnn_num_layers: int
    rnn_dropout: float
    bidirectional: bool


class ConvBlock(nn.Module):
    """Simple conv → batch norm → ReLU → pooling block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        pool_size: Tuple[int, int],
    ) -> None:
        super().__init__()
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CRNN(nn.Module):
    """Convolutional Recurrent encoder mapping spectrogram instances to features."""

    def __init__(self, config: CRNNConfig) -> None:
        super().__init__()
        conv_layers: List[nn.Module] = []
        in_channels = config.input_channels
        for layer_cfg in config.conv_layers:
            out_channels = int(layer_cfg.get("out_channels", 32))
            kernel_size = layer_cfg.get("kernel_size", (3, 3))
            pool_size = layer_cfg.get("pool_size", (2, 2))
            conv_layers.append(
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=tuple(kernel_size),  # type: ignore[arg-type]
                    pool_size=tuple(pool_size),  # type: ignore[arg-type]
                )
            )
            in_channels = out_channels
        self.conv = nn.Sequential(*conv_layers)
        self.temporal_projection = nn.AdaptiveAvgPool2d((None, 1))
        rnn_input_size = in_channels
        self.rnn = nn.GRU(
            input_size=rnn_input_size,
            hidden_size=config.rnn_hidden_size,
            num_layers=config.rnn_num_layers,
            dropout=config.rnn_dropout if config.rnn_num_layers > 1 else 0.0,
            batch_first=False,
            bidirectional=config.bidirectional,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor with shape ``(batch, bag, freq, frames)``.

        Returns:
            Instance embeddings with shape ``(batch, bag, feature_dim)``.
        """

        batch_size, bag_size, freq_bins, time_steps = x.shape
        x = x.view(batch_size * bag_size, 1, freq_bins, time_steps)
        x = self.conv(x)
        x = self.temporal_projection(x)
        x = x.squeeze(-1)  # (batch * bag, channels, time)
        x = x.permute(2, 0, 1)  # (time, batch * bag, channels)
        rnn_out, _ = self.rnn(x)
        features = rnn_out[-1]
        features = features.view(batch_size, bag_size, -1)
        return features


class MILPoolingHead(nn.Module):
    """Multiple instance pooling head supporting several aggregation types."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        pooling_type: str = "linear_softmax",
        attention_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.pooling_type = pooling_type
        self.classifier = nn.Linear(input_dim, output_dim)
        if pooling_type == "attention":
            self.attention = nn.Sequential(
                nn.Linear(input_dim, attention_hidden),
                nn.Tanh(),
                nn.Linear(attention_hidden, output_dim),
            )
        elif pooling_type == "linear_softmax":
            self.linear = nn.Linear(input_dim, output_dim)
        elif pooling_type == "max":
            pass
        else:
            raise ValueError(f"Unsupported pooling_type: {pooling_type}")
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Aggregate instance embeddings into bag-level logits."""

        x = self.dropout(x)
        if self.pooling_type == "attention":
            attn_logits = self.attention(x)
            attn_weights = torch.softmax(attn_logits, dim=1)
            pooled = torch.sum(attn_weights * self.classifier(x), dim=1)
        elif self.pooling_type == "linear_softmax":
            scores = self.linear(x)
            numerator = torch.sum(scores * scores, dim=1)
            denominator = torch.sum(scores + 1e-6, dim=1)
            pooled = numerator / denominator
        else:  # max pooling
            logits = self.classifier(x)
            pooled, _ = torch.max(logits, dim=1)
        return pooled


class BatMILModel(nn.Module):
    """Full MIL model combining CRNN encoder and pooling head."""

    def __init__(
        self,
        encoder: CRNN,
        pooling_head: MILPoolingHead,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.pooling_head = pooling_head

    def forward(self, instances: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return bag-level logits for multi-label classification."""

        features = self.encoder(instances)
        logits = self.pooling_head(features)
        return logits
