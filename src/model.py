"""
Hybrid 3D CNN-Transformer Deep Learning Architecture for Lung Cancer Detection.
Features multi-task heads for primary malignancy prediction and auxiliary radiological trait extraction.
Supports Monte Carlo (MC) Dropout for epistemic uncertainty estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional



class ConvBlock3D(nn.Module):
    """ 3D Convolutional Block with BatchNorm, ReLU, and Dropout """
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.2):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(p=dropout_rate)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.relu(self.bn(self.conv(x))))


class SpatialTransformerEncoder(nn.Module):
    """ Transformer Encoder for sequence of 3D Patch Embeddings """
    def __init__(self, embed_dim: int = 128, num_heads: int = 4, num_layers: int = 2, dim_feedforward: int = 256):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=0.2
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_length, embed_dim)
        return self.transformer(x)


class HybridLungCancerModel(nn.Module):
    """
    Adaptive Hybrid 3D CNN-Transformer Framework with Multi-Task Output Heads.
    """
    def __init__(self, in_channels: int = 1, mc_dropout_rate: float = 0.25):
        super().__init__()
        self.mc_dropout_rate = mc_dropout_rate
        
        # 3D CNN Feature Extractor
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),  # (B, 32, D/2, H/2, W/2)
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)  # (B, 32, D/4, H/4, W/4)
        )
        
        self.block1 = ConvBlock3D(32, 64, dropout_rate=mc_dropout_rate)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)  # (B, 64, D/8, H/8, W/8)
        
        self.block2 = ConvBlock3D(64, 128, dropout_rate=mc_dropout_rate)
        self.pool2 = nn.AdaptiveAvgPool3d((4, 4, 4))  # Fixed spatial feature grid (B, 128, 4, 4, 4)
        
        # Patch Projection for Transformer
        self.embed_dim = 128
        self.num_patches = 4 * 4 * 4  # 64 patch tokens
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches, self.embed_dim))
        nn.init.normal_(self.pos_embedding, std=0.02)
        
        # Transformer Encoder Layer
        self.transformer = SpatialTransformerEncoder(embed_dim=self.embed_dim, num_heads=4, num_layers=2)
        
        # Global Feature Aggregator
        self.fc_shared = nn.Sequential(
            nn.Linear(self.embed_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=mc_dropout_rate)
        )
        
        # Head 1: Primary Malignancy Classification (Binary: 0=Benign, 1=Cancerous)
        self.primary_head = nn.Linear(128, 2)
        
        # Head 2: Auxiliary Radiological Characteristics Prediction
        # [Spiculation, Lobulation, Calcification, Subtlety, Margin, Sphericity]
        self.auxiliary_radiology_head = nn.Linear(128, 6)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 3D CNN Feature Extraction
        x = self.stem(x)
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))  # (B, 128, 4, 4, 4)
        
        # 2. Reshape to sequence of 3D patch tokens: (B, 64, 128)
        B, C, D, H, W = x.shape
        x_patches = x.permute(0, 2, 3, 4, 1).contiguous().view(B, D * H * W, C)
        
        # 3. Add positional embeddings & Transformer Attention
        x_patches = x_patches + self.pos_embedding
        x_trans = self.transformer(x_patches)  # (B, 64, 128)
        
        # 4. Global Average Pooling over patch sequence
        global_repr = x_trans.mean(dim=1)  # (B, 128)
        shared_features = self.fc_shared(global_repr)  # (B, 128)
        return shared_features

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shared_features = self.forward_features(x)
        
        # Primary Malignancy Logits
        primary_logits = self.primary_head(shared_features)  # (B, 2)
        
        # Auxiliary Radiological Trait Scores
        radiology_traits = self.auxiliary_radiology_head(shared_features)  # (B, 6)
        
        return primary_logits, radiology_traits

    def enable_mc_dropout(self):
        """ Force dropout layers to stay active during inference for Monte Carlo sampling """
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout3d)):
                m.train()
