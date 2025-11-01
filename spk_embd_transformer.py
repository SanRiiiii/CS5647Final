#!/usr/bin/env python3
"""
Speaker Embedding Transformer (Multi-Modal Version)
以预训练的spk_embd作为输入，支持多种变换形式：
1. 线性变换 (linear)
2. 逐层门控 (gated)
3. 条件调制 (conditional)
并预测F0进行对抗训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Any
from torch import Tensor
from grl import GradientReversal


class LinearTransformer(nn.Module):
    """线性变换器"""
    
    def __init__(self, spk_embd_dim: int, output_dim: int):
        super().__init__()
        self.weight_matrix = nn.Linear(spk_embd_dim, output_dim, bias=False)
        nn.init.xavier_uniform_(self.weight_matrix.weight)
    
    def forward(self, spk_embd: Tensor) -> Tensor:
        return self.weight_matrix(spk_embd)


class GatedTransformer(nn.Module):
    """逐层门控变换器"""
    
    def __init__(self, spk_embd_dim: int, output_dim: int, num_layers: int = 3, hidden_dim: int = 512, dropout_rate: float = 0.1):
        super().__init__()
        self.spk_embd_dim = spk_embd_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # 逐层门控网络
        self.gate_layers = nn.ModuleList()
        self.transform_layers = nn.ModuleList()
        
        current_dim = spk_embd_dim
        for i in range(num_layers):
            # 门控层
            gate_layer = nn.Sequential(
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, current_dim),
                nn.Sigmoid()
            )
            
            # 变换层
            transform_layer = nn.Sequential(
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, current_dim),
            )
            
            self.gate_layers.append(gate_layer)
            self.transform_layers.append(transform_layer)
        
        # 输出投影
        if output_dim != spk_embd_dim:
            self.output_proj = nn.Linear(spk_embd_dim, output_dim)
        else:
            self.output_proj = nn.Identity()
    
    def forward(self, spk_embd: Tensor) -> Tensor:
        x = spk_embd
        
        for i in range(self.num_layers):
            # 计算门控权重
            gate_weights = self.gate_layers[i](x)
            
            # 计算变换特征
            transformed_features = self.transform_layers[i](x)
            
            # 门控操作：gate * transformed + (1 - gate) * original
            x = gate_weights * transformed_features + (1 - gate_weights) * x
        
        return self.output_proj(x)


class FiLMTransformer(nn.Module):
    """FiLM (Feature-wise Linear Modulation) 变换器"""
    
    def __init__(self, spk_embd_dim: int, output_dim: int, condition_dim: int = 128, num_film_layers: int = 2, dropout_rate: float = 0.1):
        super().__init__()
        self.spk_embd_dim = spk_embd_dim
        self.output_dim = output_dim
        self.condition_dim = condition_dim
        self.num_film_layers = num_film_layers
        
        # 条件编码器：将spk_embd编码为条件特征
        self.condition_encoder = nn.Sequential(
            nn.Linear(spk_embd_dim, condition_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(condition_dim, condition_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # FiLM层
        self.film_layers = nn.ModuleList()
        for i in range(num_film_layers):
            film_layer = nn.ModuleDict({
                # 特征变换层
                'feature_transform': nn.Sequential(
                    nn.Linear(spk_embd_dim, spk_embd_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                ),
                # FiLM调制参数生成器
                'film_generator': nn.Sequential(
                    nn.Linear(condition_dim, spk_embd_dim * 2),  # 生成scale和shift参数
                )
            })
            self.film_layers.append(film_layer)
        
        # 输出投影
        if output_dim != spk_embd_dim:
            self.output_proj = nn.Linear(spk_embd_dim, output_dim)
        else:
            self.output_proj = nn.Identity()
    
    def forward(self, spk_embd: Tensor) -> Tensor:
        # 编码条件特征
        condition = self.condition_encoder(spk_embd)  # [B, condition_dim]
        
        # 逐层FiLM调制
        x = spk_embd  # [B, spk_embd_dim]
        
        for film_layer in self.film_layers:
            # 特征变换
            transformed_features = film_layer['feature_transform'](x)  # [B, spk_embd_dim]
            
            # 生成FiLM参数
            film_params = film_layer['film_generator'](condition)  # [B, spk_embd_dim * 2]
            scale = film_params[:, :self.spk_embd_dim]  # [B, spk_embd_dim]
            shift = film_params[:, self.spk_embd_dim:]   # [B, spk_embd_dim]
            
            # 应用FiLM调制：scale * features + shift
            x = scale * transformed_features + shift
        
        return self.output_proj(x)


class SpeakerEmbeddingTransformer(nn.Module):
    """
    Speaker Embedding Transformer (Multi-Modal Version)
    
    支持多种变换形式：
    1. linear: 简单线性变换
    2. gated: 逐层门控变换
    3. film: FiLM (Feature-wise Linear Modulation) 变换
    
    Args:
        spk_embd_dim: 输入说话人嵌入维度 (默认256)
        output_dim: 输出维度 (默认256，与n_hidden对齐)
        f0_pred_dim: F0预测维度 (默认1)
        transform_type: 变换类型 ('linear', 'gated', 'film')
        transform_config: 变换器配置参数
        dropout_rate: Dropout率 (默认0.1)
    """
    
    def __init__(
        self,
        spk_embd_dim: int = 256,
        output_dim: int = 256,
        f0_pred_dim: int = 1,
        transform_type: str = 'linear',
        transform_config: Optional[Dict[str, Any]] = None,
        dropout_rate: float = 0.1,
    ):
        super(SpeakerEmbeddingTransformer, self).__init__()
        
        self.spk_embd_dim = spk_embd_dim
        self.output_dim = output_dim
        self.f0_pred_dim = f0_pred_dim
        self.transform_type = transform_type
        
        # 默认配置
        if transform_config is None:
            transform_config = {}
        
        # 选择变换器
        if transform_type == 'linear':
            self.transformer = LinearTransformer(spk_embd_dim, output_dim)
        elif transform_type == 'gated':
            num_layers = transform_config.get('num_layers', 3)
            hidden_dim = transform_config.get('hidden_dim', 512)
            self.transformer = GatedTransformer(spk_embd_dim, output_dim, num_layers, hidden_dim, dropout_rate)
        elif transform_type == 'film':
            condition_dim = transform_config.get('condition_dim', 128)
            num_film_layers = transform_config.get('num_film_layers', 2)
            self.transformer = FiLMTransformer(spk_embd_dim, output_dim, condition_dim, num_film_layers, dropout_rate)
        else:
            raise ValueError(f"Unsupported transform_type: {transform_type}")
        
        # F0分布预测头：用于对抗训练（与spk_encoder的F0分布分类一致）
        self.f0_dist_predictor = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(output_dim, 5 * 5),  # 5个分位数 x 5个类别 = 25维
        )
        
        # 初始化完成（移除调试打印）
    
    def forward(self, spk_embd: Tensor, return_f0_dist_pred: bool = False):
        """
        前向传播
        
        Args:
            spk_embd: 预训练的说话人嵌入 [B, spk_embd_dim]
            return_f0_dist_pred: 是否返回F0分布预测结果
            
        Returns:
            transformed_spk_embd: 变换后的说话人嵌入 [B, output_dim]
            f0_dist_pred: F0分布预测结果 [B, 5, 5] (仅在return_f0_dist_pred=True时返回)
        """
        # 检查输入维度
        if spk_embd.dim() != 2 or spk_embd.size(-1) != self.spk_embd_dim:
            raise ValueError(f"spk_embd shape must be [B, {self.spk_embd_dim}], got {tuple(spk_embd.shape)}")
        
        # 通过变换器变换
        transformed_spk_embd = self.transformer(spk_embd)  # [B, output_dim]
        
        if not return_f0_dist_pred:
            return transformed_spk_embd
        
        # F0分布预测（用于对抗训练）
        f0_dist_logits = self.f0_dist_predictor(transformed_spk_embd)  # [B, 25]
        f0_dist_pred = f0_dist_logits.view(-1, 5, 5)  # [B, 5, 5]: 5个分位数，每个5分类
        
        return transformed_spk_embd, f0_dist_pred
    
    def get_transform_info(self):
        """获取变换器信息（用于分析）"""
        info = {
            'transform_type': self.transform_type,
            'spk_embd_dim': self.spk_embd_dim,
            'output_dim': self.output_dim,
        }
        
        if self.transform_type == 'linear':
            info['weight_matrix_shape'] = self.transformer.weight_matrix.weight.shape
        elif self.transform_type == 'gated':
            info['num_layers'] = self.transformer.num_layers
        elif self.transform_type == 'film':
            info['condition_dim'] = self.transformer.condition_dim
            info['num_film_layers'] = self.transformer.num_film_layers
        
        return info


class SpeakerEmbeddingTransformerWithGRL(nn.Module):
    """
    带GRL的Speaker Embedding Transformer (Multi-Modal Version)
    
    用于对抗训练：让变换后的spk_embd无法预测F0
    """
    
    def __init__(
        self,
        spk_embd_dim: int = 256,
        output_dim: int = 256,
        f0_pred_dim: int = 1,
        transform_type: str = 'linear',
        transform_config: Optional[Dict[str, Any]] = None,
        dropout_rate: float = 0.1,
    ):
        super(SpeakerEmbeddingTransformerWithGRL, self).__init__()
        
        self.transformer = SpeakerEmbeddingTransformer(
            spk_embd_dim=spk_embd_dim,
            output_dim=output_dim,
            f0_pred_dim=f0_pred_dim,
            transform_type=transform_type,
            transform_config=transform_config,
            dropout_rate=dropout_rate,
        )
        
        # GRL用于对抗训练
        self.grl = GradientReversal()
    
    def forward(self, spk_embd: Tensor, grl_lambda: float = 1.0):
        """
        前向传播（带GRL）
        
        Args:
            spk_embd: 预训练的说话人嵌入 [B, spk_embd_dim]
            grl_lambda: GRL强度
            
        Returns:
            transformed_spk_embd: 变换后的说话人嵌入 [B, output_dim]
            f0_dist_pred_with_grl: 带GRL的F0分布预测结果 [B, 5, 5]
        """
        # 获取变换后的spk_embd
        transformed_spk_embd = self.transformer(spk_embd, return_f0_dist_pred=False)
        
        # 对变换后的spk_embd应用GRL，然后预测F0分布
        spk_embd_with_grl = self.grl(transformed_spk_embd, grl_lambda)
        f0_dist_logits = self.transformer.f0_dist_predictor(spk_embd_with_grl)
        f0_dist_pred_with_grl = f0_dist_logits.view(-1, 5, 5)  # [B, 5, 5]
        
        return transformed_spk_embd, f0_dist_pred_with_grl
