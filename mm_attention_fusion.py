import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * x / rms


class MultiModalCrossAttention(nn.Module):
    """
    多模态交叉注意力机制 (增强Speaker依赖版本)
    
    设计：
    - Query: 内容特征 C_Q
    - 一路 Cross Attention 用于融合音色：音色 -> S_K, S_V，与内容做 attention
    - 另一路 Cross Attention 用于融合旋律：F0+Volume -> P_K, P_V，与内容做 attention
    - 两个路径结果相加，然后额外concat speaker embedding增强speaker依赖
    - 经投影层 + 残差 + FFN
    
    增强Speaker依赖的改进：
    1. 门控固定为1.0，最大化speaker attention影响
    2. Speaker特征在融合时获得2倍权重
    3. 额外concat speaker embedding
    4. 添加speaker残差连接
    5. 多重路径确保speaker信息传递
    """
    def __init__(self, d_model, num_heads=8, dropout=0.1, init_alpha=1.0):  # 门控固定为1.0
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 内容 Query 规范化与投影
        self.content_rmsnorm = RMSNorm(d_model)
        self.content_q_proj = nn.Linear(d_model, d_model)
        self.content_q_norm = nn.LayerNorm(d_model)

        # 音色路径：Speaker Key/Value 映射
        self.speaker_k_proj = nn.Linear(d_model, d_model)
        self.speaker_v_proj = nn.Linear(d_model, d_model)
        self.speaker_k_norm = nn.LayerNorm(d_model)
        
        # 旋律路径：Pitch Key/Value 映射  
        self.pitch_k_proj = nn.Linear(d_model, d_model)
        self.pitch_v_proj = nn.Linear(d_model, d_model)
        self.pitch_k_norm = nn.LayerNorm(d_model)
        
        # 交叉注意力层
        self.speaker_cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.pitch_cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        
        # 门控系数（可学习参数，sigmoid 保证在[0,1]之间）。初始化约为 0.3。
        # 令 gate = sigmoid(gate_alpha_raw)，为使 gate≈init_alpha，解 raw = logit(init_alpha)
        self.gate_alpha_raw = nn.Parameter(torch.logit(torch.tensor(init_alpha, dtype=torch.get_default_dtype())))
    

        # 融合层
        self.fusion_linear = nn.Linear(d_model, d_model)
        # 额外的投影层用于处理concat后的2*D维度
        self.fusion_proj = nn.Linear(d_model * 2, d_model)

        self.volume_linear = nn.Linear(d_model, d_model)
        self.volume_norm = nn.LayerNorm(d_model)

        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, content_features, speaker_features, pitch_volume_features):
        """
        多模态交叉注意力前向传播
        
        Args:
            content_features: [batch_size, seq_len, d_model] - 内容特征 (Query)
            speaker_features: [batch_size, 1, d_model] - 说话人特征（音色）
            pitch_volume_features: [batch_size, seq_len, d_model] - 音高+音量拼接特征
        """
        batch_size, seq_len, d_model = content_features.shape
        
        # 1. 音色路径：Speaker Cross Attention
        # 音色特征映射为 Key/Value 并归一化
        speaker_k = self.speaker_k_norm(self.speaker_k_proj(speaker_features))  # [B, 1, D]
        speaker_v = self.speaker_v_proj(speaker_features)  # [B, 1, D]

        # 内容作为 Query：RMSNorm -> 线性投影 -> LayerNorm 得到 CQ
        content_q = self.content_q_norm(self.content_q_proj(self.content_rmsnorm(content_features)))

        # 内容作为 Query，音色作为 Key/Value
        speaker_attended, speaker_attn_weights = self.speaker_cross_attn(
            query=content_q,             # [B, T, D] - 内容 Query (CQ)
            key=speaker_k,              # [B, 1, D] - 音色 Key
            value=speaker_v             # [B, 1, D] - 音色 Value
        )
        
        # 门控控制音色注入强度（最大化speaker依赖）
        gate = torch.tensor(1.0, device=speaker_attended.device)  # 固定为1，最大化speaker影响
        # 直接使用attention结果，不混合原始speaker特征
        # speaker_attended = gate * speaker_attended + (1 - gate) * speaker_features.expand(-1, seq_len, -1)
        
        # 2. 音高+音量路径：Pitch-Volume Cross Attention
        # 音高+音量拼接特征映射为 Key/Value 并归一化
        pitch_volume_k = self.pitch_k_norm(self.pitch_k_proj(pitch_volume_features))  # [B, T, D]
        pitch_volume_v = self.pitch_v_proj(pitch_volume_features)  # [B, T, D]
        
        # 内容作为 Query，音高+音量作为 Key/Value
        pitch_volume_attended, pitch_volume_attn_weights = self.pitch_cross_attn(
            query=content_q,                    # [B, T, D] - 内容 Query (CQ)
            key=pitch_volume_k,                 # [B, T, D] - 音高+音量 Key
            value=pitch_volume_v                # [B, T, D] - 音高+音量 Value
        )
        
   
        fused_features = speaker_attended + pitch_volume_attended 
        fused_features = self.fusion_linear(fused_features) 
        

        # 将speaker_features扩展到与content相同的序列长度
        speaker_features_expanded = speaker_features.expand(-1, seq_len, -1)  # [B, T, D]
        fused_features = torch.cat([fused_features, speaker_features_expanded], dim=-1)  # [B, T, 2*D]
        
        # 投影回原始维度
        fused_features = self.fusion_proj(fused_features)  # [B, T, D]
        
        # 5. 残差连接和层归一化（额外加入speaker残差）
        attended_features = self.norm1(fused_features)
        
        # 6. 前馈网络
        ffn_output = self.ffn(attended_features)

        # 7. 残差连接和层归一化
        final_output = self.norm2(attended_features + ffn_output)
    
        
        return final_output, gate

class StackedMultiModalCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads=8, dropout=0.1, init_alpha=0.3, num_layers: int = 3):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            MultiModalCrossAttention(d_model, num_heads, dropout, init_alpha) for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        
        # 添加concat后的线性变换层：从2*d_model维度转换为d_model维度
        self.concat_linear = nn.Linear(d_model * 2, d_model)

    def forward(self, content_features, speaker_features, pitch_volume_features):
        x = content_features
        gate_values = []  # 收集每层的门控因子
        for i in range(self.num_layers):
            tmp, gate = self.layers[i](x, speaker_features, pitch_volume_features)
            gate_values.append(gate)
            x = self.layer_norms[i](x + tmp)
        
        # 返回最终输出和平均门控因子
        avg_gate = torch.stack(gate_values).mean()
        return x, avg_gate



