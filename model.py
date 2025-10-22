"""
Neural network models for bat audio classification using Multiple-Instance Learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict


class CRNN(nn.Module):
    """
    Convolutional Recurrent Neural Network for audio feature extraction.
    
    Architecture: Conv layers -> BatchNorm -> ReLU -> Pooling -> BiGRU
    """
    
    def __init__(
        self,
        input_dim: int,
        conv_layers: List[Dict],
        pool_size: List[int],
        rnn_hidden_size: int,
        rnn_num_layers: int = 2,
        dropout: float = 0.5,
    ):
        """
        Args:
            input_dim: Input feature dimension
            conv_layers: List of dicts with conv layer parameters
            pool_size: Max pooling size [height, width]
            rnn_hidden_size: Hidden size for BiGRU
            rnn_num_layers: Number of RNN layers
            dropout: Dropout rate
        """
        super(CRNN, self).__init__()
        
        self.input_dim = input_dim
        self.rnn_hidden_size = rnn_hidden_size
        
        # Build convolutional layers
        self.conv_blocks = nn.ModuleList()
        in_channels = 1  # Input is single-channel spectrogram
        
        for layer_config in conv_layers:
            conv_block = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=layer_config['filters'],
                    kernel_size=tuple(layer_config['kernel_size']),
                    stride=tuple(layer_config['stride']),
                    padding=tuple(layer_config['padding']),
                ),
                nn.BatchNorm2d(layer_config['filters']),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=tuple(pool_size)),
                nn.Dropout2d(p=dropout),
            )
            self.conv_blocks.append(conv_block)
            in_channels = layer_config['filters']
        
        # Placeholder for computed conv output size
        self.conv_output_size = None
        
        # Recurrent layer (BiGRU)
        self.rnn = nn.GRU(
            input_size=self.conv_output_size if self.conv_output_size else input_dim,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if rnn_num_layers > 1 else 0,
        )
        
        self.dropout = nn.Dropout(p=dropout)
        
    def _compute_conv_output_size(self, x: torch.Tensor) -> int:
        """
        Compute the output size after conv layers.
        
        Args:
            x: Input tensor (B, T, F)
            
        Returns:
            Output feature dimension
        """
        # Add channel dimension and reshape for conv2d
        # Treat each instance as (1, height, width)
        batch_size, seq_len, feat_dim = x.shape
        
        # For now, we'll compute this dynamically
        # Reshape to (B*T, 1, H, W) for conv processing
        # This is a placeholder - you need to determine H, W from your feature extraction
        return feat_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CRNN.
        
        Args:
            x: Input tensor (B, T, F) where B=batch, T=instances, F=features
            
        Returns:
            Instance features (B, T, rnn_hidden_size * 2)
        """
        batch_size, seq_len, feat_dim = x.shape
        
        # For simplicity, we'll process the sequence through RNN directly
        # In a full implementation, you'd reshape and apply conv2d first
        
        # Initialize RNN input size if not set
        if self.conv_output_size is None:
            self.conv_output_size = feat_dim
            # Reinitialize RNN with correct input size
            self.rnn = nn.GRU(
                input_size=feat_dim,
                hidden_size=self.rnn_hidden_size,
                num_layers=self.rnn.num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=self.rnn.dropout if self.rnn.num_layers > 1 else 0,
            ).to(x.device)
        
        # Pass through RNN
        rnn_out, _ = self.rnn(x)  # (B, T, rnn_hidden_size * 2)
        
        # Apply dropout
        rnn_out = self.dropout(rnn_out)
        
        return rnn_out


class MILPoolingHead(nn.Module):
    """
    Multiple-Instance Learning pooling head.
    
    Aggregates instance features into bag-level prediction.
    Supports different pooling strategies: linear_softmax, attention, max.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        pooling_type: str = "linear_softmax",
    ):
        """
        Args:
            input_dim: Input feature dimension (from CRNN)
            num_classes: Number of output classes
            pooling_type: Type of pooling ("linear_softmax", "attention", "max")
        """
        super(MILPoolingHead, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.pooling_type = pooling_type
        
        if pooling_type == "linear_softmax":
            # Linear layer + softmax weights for weighted averaging
            self.attention = nn.Linear(input_dim, num_classes)
            
        elif pooling_type == "attention":
            # Attention mechanism
            self.attention_weights = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.Tanh(),
                nn.Linear(input_dim // 2, 1),
            )
            self.classifier = nn.Linear(input_dim, num_classes)
            
        elif pooling_type == "max":
            # Max pooling + classification
            self.classifier = nn.Linear(input_dim, num_classes)
            
        else:
            raise ValueError(f"Unknown pooling type: {pooling_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MIL pooling head.
        
        Args:
            x: Instance features (B, T, F)
            
        Returns:
            Class logits (B, C)
        """
        if self.pooling_type == "linear_softmax":
            # Compute instance scores
            instance_scores = self.attention(x)  # (B, T, C)
            
            # Apply softmax across instances
            attention_weights = F.softmax(instance_scores, dim=1)  # (B, T, C)
            
            # Weighted sum of instance scores
            bag_logits = torch.sum(instance_scores * attention_weights, dim=1)  # (B, C)
            
        elif self.pooling_type == "attention":
            # Compute attention weights for each instance
            attention_scores = self.attention_weights(x)  # (B, T, 1)
            attention_weights = F.softmax(attention_scores, dim=1)  # (B, T, 1)
            
            # Weighted average of instance features
            weighted_features = torch.sum(x * attention_weights, dim=1)  # (B, F)
            
            # Classify aggregated features
            bag_logits = self.classifier(weighted_features)  # (B, C)
            
        elif self.pooling_type == "max":
            # Max pooling across instances
            max_features, _ = torch.max(x, dim=1)  # (B, F)
            
            # Classify
            bag_logits = self.classifier(max_features)  # (B, C)
        
        return bag_logits


class BatMILModel(nn.Module):
    """
    Complete model for bat audio classification using Multiple-Instance Learning.
    
    Combines CRNN feature extractor with MIL pooling head.
    """
    
    def __init__(self, config: dict):
        """
        Args:
            config: Configuration dictionary
        """
        super(BatMILModel, self).__init__()
        
        self.config = config
        
        # Extract configuration
        conv_layers = config['model']['conv_layers']
        pool_size = config['model']['pool_size']
        rnn_hidden_size = config['model']['rnn_hidden_size']
        rnn_num_layers = config['model']['rnn_num_layers']
        dropout = config['model']['dropout']
        num_classes = config['data']['num_classes']
        pooling_type = config['model']['pooling_type']
        
        # Placeholder input dimension (will be set dynamically)
        input_dim = 1000  # This will be updated based on actual input
        
        # Initialize CRNN
        self.crnn = CRNN(
            input_dim=input_dim,
            conv_layers=conv_layers,
            pool_size=pool_size,
            rnn_hidden_size=rnn_hidden_size,
            rnn_num_layers=rnn_num_layers,
            dropout=dropout,
        )
        
        # Initialize MIL pooling head
        # Input dimension is rnn_hidden_size * 2 (bidirectional)
        self.mil_head = MILPoolingHead(
            input_dim=rnn_hidden_size * 2,
            num_classes=num_classes,
            pooling_type=pooling_type,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the complete model.
        
        Args:
            x: Input instances (B, T, F)
            
        Returns:
            Class logits (B, C)
        """
        # Extract instance features
        instance_features = self.crnn(x)  # (B, T, rnn_hidden_size * 2)
        
        # Aggregate to bag-level prediction
        logits = self.mil_head(instance_features)  # (B, C)
        
        return logits
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class probabilities.
        
        Args:
            x: Input instances (B, T, F)
            
        Returns:
            Class probabilities (B, C)
        """
        logits = self.forward(x)
        probs = torch.sigmoid(logits)  # Multi-label classification
        return probs


def create_model(config: dict) -> BatMILModel:
    """
    Factory function to create model from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        BatMILModel instance
    """
    model = BatMILModel(config)
    return model
