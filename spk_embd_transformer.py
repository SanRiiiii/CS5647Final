#!/usr/bin/env python3
"""
Speaker Embedding Transformer (Linear Version)
Takes pre-trained spk_embd as input, applies linear transformation, and predicts F0 for adversarial training
"""

import torch
import torch.nn as nn
from torch import Tensor
from grl import GradientReversal


class LinearTransformer(nn.Module):
    """Linear Transformer"""
    
    def __init__(self, spk_embd_dim: int, output_dim: int):
        super().__init__()
        self.weight_matrix = nn.Linear(spk_embd_dim, output_dim, bias=False)
        nn.init.xavier_uniform_(self.weight_matrix.weight)
    
    def forward(self, spk_embd: Tensor) -> Tensor:
        return self.weight_matrix(spk_embd)


class SpeakerEmbeddingTransformer(nn.Module):
    """
    Speaker Embedding Transformer (Linear Version)
    
    Processes speaker embeddings using linear transformation
    
    Args:
        spk_embd_dim: Input speaker embedding dimension (default: 256)
        output_dim: Output dimension (default: 256, aligned with n_hidden)
        f0_pred_dim: F0 prediction dimension (default: 1)
        dropout_rate: Dropout rate (default: 0.1)
    """
    
    def __init__(
        self,
        spk_embd_dim: int = 256,
        output_dim: int = 256,
        f0_pred_dim: int = 1,
        dropout_rate: float = 0.1,
    ):
        super(SpeakerEmbeddingTransformer, self).__init__()
        
        self.spk_embd_dim = spk_embd_dim
        self.output_dim = output_dim
        self.f0_pred_dim = f0_pred_dim
        
        # Linear transformer
        self.transformer = LinearTransformer(spk_embd_dim, output_dim)
        
        # F0 distribution prediction head: for adversarial training (consistent with spk_encoder's F0 distribution classification)
        self.f0_dist_predictor = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(output_dim, 5 * 5),  # 5 quantiles x 5 classes = 25 dimensions
        )
    
    def forward(self, spk_embd: Tensor, return_f0_dist_pred: bool = False):
        """
        Forward pass
        
        Args:
            spk_embd: Pre-trained speaker embedding [B, spk_embd_dim]
            return_f0_dist_pred: Whether to return F0 distribution prediction
            
        Returns:
            transformed_spk_embd: Transformed speaker embedding [B, output_dim]
            f0_dist_pred: F0 distribution prediction [B, 5, 5] (only returned when return_f0_dist_pred=True)
        """
        # Check input dimensions
        if spk_embd.dim() != 2 or spk_embd.size(-1) != self.spk_embd_dim:
            raise ValueError(f"spk_embd shape must be [B, {self.spk_embd_dim}], got {tuple(spk_embd.shape)}")
        
        # Transform through transformer
        transformed_spk_embd = self.transformer(spk_embd)  # [B, output_dim]
        
        if not return_f0_dist_pred:
            return transformed_spk_embd
        
        # F0 distribution prediction (for adversarial training)
        f0_dist_logits = self.f0_dist_predictor(transformed_spk_embd)  # [B, 25]
        f0_dist_pred = f0_dist_logits.view(-1, 5, 5)  # [B, 5, 5]: 5 quantiles, each with 5 classes
        
        return transformed_spk_embd, f0_dist_pred
    
    def get_transform_info(self):
        """Get transformer information (for analysis)"""
        info = {
            'transform_type': 'linear',
            'spk_embd_dim': self.spk_embd_dim,
            'output_dim': self.output_dim,
            'weight_matrix_shape': self.transformer.weight_matrix.weight.shape
        }
        
        return info


class SpeakerEmbeddingTransformerWithGRL(nn.Module):
    """
    Speaker Embedding Transformer with GRL (Linear Version)
    
    Used for adversarial training: prevents transformed spk_embd from predicting F0
    """
    
    def __init__(
        self,
        spk_embd_dim: int = 256,
        output_dim: int = 256,
        f0_pred_dim: int = 1,
        dropout_rate: float = 0.1,
    ):
        super(SpeakerEmbeddingTransformerWithGRL, self).__init__()
        
        self.transformer = SpeakerEmbeddingTransformer(
            spk_embd_dim=spk_embd_dim,
            output_dim=output_dim,
            f0_pred_dim=f0_pred_dim,
            dropout_rate=dropout_rate,
        )
        
        # GRL for adversarial training
        self.grl = GradientReversal()
    
    def forward(self, spk_embd: Tensor, grl_lambda: float = 1.0):
        """
        Forward pass (with GRL)
        
        Args:
            spk_embd: Pre-trained speaker embedding [B, spk_embd_dim]
            grl_lambda: GRL strength
            
        Returns:
            transformed_spk_embd: Transformed speaker embedding [B, output_dim]
            f0_dist_pred_with_grl: F0 distribution prediction with GRL [B, 5, 5]
        """
        # Get transformed spk_embd
        transformed_spk_embd = self.transformer(spk_embd, return_f0_dist_pred=False)
        
        # Apply GRL to transformed spk_embd, then predict F0 distribution
        spk_embd_with_grl = self.grl(transformed_spk_embd, grl_lambda)
        f0_dist_logits = self.transformer.f0_dist_predictor(spk_embd_with_grl)
        f0_dist_pred_with_grl = f0_dist_logits.view(-1, 5, 5)  # [B, 5, 5]
        
        return transformed_spk_embd, f0_dist_pred_with_grl
