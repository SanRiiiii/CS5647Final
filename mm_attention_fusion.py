import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RMSNorm(nn.Module):

    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * x / rms


class MultiModalCrossAttention(nn.Module):

    def __init__(self, d_model, num_heads=8, dropout=0.1, init_alpha=1.0):  # 门控固定为1.0
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
    
        self.content_rmsnorm = RMSNorm(d_model)
        self.content_q_proj = nn.Linear(d_model, d_model)
        self.content_q_norm = nn.LayerNorm(d_model)

        self.speaker_k_proj = nn.Linear(d_model, d_model)
        self.speaker_v_proj = nn.Linear(d_model, d_model)
        self.speaker_k_norm = nn.LayerNorm(d_model)
        
        self.pitch_k_proj = nn.Linear(d_model, d_model)
        self.pitch_v_proj = nn.Linear(d_model, d_model)
        self.pitch_k_norm = nn.LayerNorm(d_model)
        
        self.speaker_cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.pitch_cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        
        self.gate_alpha_raw = nn.Parameter(torch.logit(torch.tensor(init_alpha, dtype=torch.get_default_dtype())))
    

        self.fusion_linear = nn.Linear(d_model, d_model)
        self.fusion_proj = nn.Linear(d_model * 2, d_model)

        self.volume_linear = nn.Linear(d_model, d_model)
        self.volume_norm = nn.LayerNorm(d_model)

        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
        
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
        
    
        speaker_k = self.speaker_k_norm(self.speaker_k_proj(speaker_features))  # [B, 1, D]
        speaker_v = self.speaker_v_proj(speaker_features)  # [B, 1, D]

        content_q = self.content_q_norm(self.content_q_proj(self.content_rmsnorm(content_features)))

        speaker_attended, speaker_attn_weights = self.speaker_cross_attn(
            query=content_q,             # [B, T, D] -  Query (CQ)
            key=speaker_k,              # [B, 1, D] - Speaker Key
            value=speaker_v             # [B, 1, D] - Speaker Value
        )
        
        gate = torch.tensor(1.0, device=speaker_attended.device) 
        
        pitch_volume_k = self.pitch_k_norm(self.pitch_k_proj(pitch_volume_features))  # [B, T, D]
        pitch_volume_v = self.pitch_v_proj(pitch_volume_features)  # [B, T, D]
        
        pitch_volume_attended, pitch_volume_attn_weights = self.pitch_cross_attn(
            query=content_q,                    # [B, T, D] -  Query (CQ)
            key=pitch_volume_k,                 # [B, T, D] - Pitch+Volume Key
            value=pitch_volume_v                # [B, T, D] - Pitch+Volume Value
        )
        
   
        fused_features = speaker_attended + pitch_volume_attended 
        fused_features = self.fusion_linear(fused_features) 
        

        speaker_features_expanded = speaker_features.expand(-1, seq_len, -1) 
        fused_features = torch.cat([fused_features, speaker_features_expanded], dim=-1) 
        
        fused_features = self.fusion_proj(fused_features) 
        
        attended_features = self.norm1(fused_features)
        
        ffn_output = self.ffn(attended_features)

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
        
        self.concat_linear = nn.Linear(d_model * 2, d_model)

    def forward(self, content_features, speaker_features, pitch_volume_features):
        x = content_features
        gate_values = []  
        for i in range(self.num_layers):
            tmp, gate = self.layers[i](x, speaker_features, pitch_volume_features)
            gate_values.append(gate)
            x = self.layer_norms[i](x + tmp)
        
        avg_gate = torch.stack(gate_values).mean()
        return x, avg_gate



