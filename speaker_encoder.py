# # 03_models/speaker_encoder.py
# '''
# Implement of zero-shot speaker encoder
# Use mel spectruam as input

# '''
# from typing import List
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch import Tensor


# class GradientReversal(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x: Tensor, lambda_: float):
#         ctx.lambda_ = lambda_
#         ctx.input_dtype = x.dtype
#         return x.view_as(x)

#     @staticmethod
#     def backward(ctx, grad_output: Tensor):
#         # Ensure output dtype matches input dtype for mixed precision
#         return (-ctx.lambda_ * grad_output).to(ctx.input_dtype), None


# class GRL(nn.Module):
#     def __init__(self, lambda_: float = 1.0):
#         super().__init__()
#         self.lambda_ = lambda_

#     def forward(self, x: Tensor) -> Tensor:
#         return GradientReversal.apply(x, self.lambda_)


# def get_act(act: str) -> nn.Module:
#     if act == "lrelu":
#         return nn.LeakyReLU()
#     return nn.ReLU()


# class ConvBank(nn.Module):
#     def __init__(self, c_in: int, c_out: int, n_bank: int, bank_scale: int, act: str):
#         super(ConvBank, self).__init__()
#         self.conv_bank = nn.ModuleList(
#             [
#                 nn.Sequential(
#                     nn.ReflectionPad1d((k // 2, k // 2 - 1 + k % 2)),
#                     nn.Conv1d(c_in, c_out, kernel_size=k),
#                 )
#                 for k in range(bank_scale, n_bank + 1, bank_scale)
#             ]
#         )
#         self.act = get_act(act)

#     def forward(self, x: Tensor) -> Tensor:
#         outs = [self.act(layer(x)) for layer in self.conv_bank]
#         out = torch.cat(outs + [x], dim=1)
#         return out


# class SpeakerEncoder(nn.Module):
#     """
#     Zero-shot Speaker Encoder
#     输入: mel [B, 80, T]
#     输出: speaker embedding [B, c_out]，用于与 n_hidden 对齐
#     """
#     def __init__(
#         self,
#         c_in: int = 80,         # mel bins
#         c_h: int = 128,         # hidden channels for conv
#         c_out: int = 256,       # output embedding dim (match n_hidden)
#         kernel_size: int = 5,
#         bank_size: int = 8,
#         bank_scale: int = 1,
#         c_bank: int = 128,
#         n_conv_blocks: int = 6,
#         n_dense_blocks: int = 2,
#         subsample: List[int] = (1, 2, 1, 2, 1, 2),
#         act: str = "lrelu",
#         dropout_rate: float = 0.1,
#     ):
#         super(SpeakerEncoder, self).__init__()
#         self.c_in = c_in
#         self.c_h = c_h
#         self.c_out = c_out
#         self.kernel_size = kernel_size
#         self.n_conv_blocks = n_conv_blocks
#         self.n_dense_blocks = n_dense_blocks
#         self.subsample = list(subsample)
#         self.act = get_act(act)

#         # ConvBank
#         self.conv_bank = ConvBank(c_in, c_bank, bank_size, bank_scale, act)
#         in_channels = c_bank * (bank_size // bank_scale) + c_in

#         # 1x1 conv to c_h
#         self.in_conv_layer = nn.Conv1d(in_channels, c_h, kernel_size=1)

#         # residual conv blocks (two convs with optional downsample)
#         self.first_conv_layers = nn.ModuleList(
#             [
#                 nn.Sequential(
#                     nn.ReflectionPad1d((kernel_size // 2, kernel_size // 2 - 1 + kernel_size % 2)),
#                     nn.Conv1d(c_h, c_h, kernel_size=kernel_size),
#                 )
#                 for _ in range(n_conv_blocks)
#             ]
#         )
#         self.second_conv_layers = nn.ModuleList(
#             [
#                 nn.Sequential(
#                     nn.ReflectionPad1d((kernel_size // 2, kernel_size // 2 - 1 + kernel_size % 2)),
#                     nn.Conv1d(c_h, c_h, kernel_size=kernel_size, stride=sub),
#                 )
#                 for sub, _ in zip(self.subsample, range(n_conv_blocks))
#             ]
#         )

#         self.pooling_layer = nn.AdaptiveAvgPool1d(1)

#         # dense residual blocks
#         self.first_dense_layers = nn.ModuleList([nn.Linear(c_h, c_h) for _ in range(n_dense_blocks)])
#         self.second_dense_layers = nn.ModuleList([nn.Linear(c_h, c_h) for _ in range(n_dense_blocks)])

#         self.output_layer = nn.Linear(c_h, c_out)
#         self.dropout_layer = nn.Dropout(p=dropout_rate)

#         # --- Auxiliary heads for DAT ---
#         # Domain classifier (binary: speech vs singing)
#         self.domain_head = nn.Sequential(
#             nn.Linear(c_out, c_out),
#             get_act(act),
#             nn.Dropout(p=dropout_rate),
#             nn.Linear(c_out, 2),
#         )
#         # F0 distribution classifier: predict [5, 5] (5 percentiles, each with 5 classes)
#         # 输出维度：[B, 5, 5]，表示5个分位数(p10,p30,p50,p70,p90)在5个类别的概率分布
#         self.f0_dist_head = nn.Sequential(
#             nn.Linear(c_out, c_out),
#             get_act(act),
#             nn.Dropout(p=dropout_rate),
#             nn.Linear(c_out, 5 * 5),  # 5个分位数 x 5个类别 = 25维
#         )
        
#         # --- MFCC prediction head ---
#         # MFCC特征预测头：预测mfcc_2_mean, mfcc_4_mean, mfcc_10_mean
#         self.mfcc_head = nn.Sequential(
#             nn.Linear(c_out, c_out),
#             get_act(act),
#             nn.Dropout(p=dropout_rate),
#             nn.Linear(c_out, 3),  # 3个MFCC特征：mfcc_2_mean, mfcc_4_mean, mfcc_10_mean
#         )

#     def conv_blocks(self, inp: Tensor) -> Tensor:
#         out = inp
#         for idx, (first_layer, second_layer) in enumerate(zip(self.first_conv_layers, self.second_conv_layers)):
#             y = first_layer(out)
#             y = self.act(y)
#             y = self.dropout_layer(y)
#             y = second_layer(y)
#             y = self.act(y)
#             y = self.dropout_layer(y)
#             if self.subsample[idx] > 1:
#                 out = F.avg_pool1d(out, kernel_size=self.subsample[idx], ceil_mode=True)
#             out = y + out
#         return out

#     def dense_blocks(self, inp: Tensor) -> Tensor:
#         out = inp
#         for first_layer, second_layer in zip(self.first_dense_layers, self.second_dense_layers):
#             y = first_layer(out)
#             y = self.act(y)
#             y = self.dropout_layer(y)
#             y = second_layer(y)
#             y = self.act(y)
#             y = self.dropout_layer(y)
#             out = y + out
#         return out

#     def forward(self, mel: Tensor, return_aux: bool = False, grl_lambda: float = 1.0, return_mfcc: bool = False):
#         """
#         mel: [B, 80, T] 或 [B, T, 80] (会自动检测和转换)
#         return_aux: whether to return domain logits and f0 stats predictions
#         grl_lambda: gradient reversal strength for domain adversarial training
#         return_mfcc: whether to return MFCC predictions
#         """
#         # 自动检测并转换维度：如果第二维不是80，则转置
#         if mel.dim() == 3 and mel.shape[1] != 80:
#             print(f"[SpeakerEncoder] Auto-transposing mel from {mel.shape} to ", end="")
#             mel = mel.transpose(1, 2)
#             print(f"{mel.shape}")
        
#         out = self.conv_bank(mel)
#         out = self.in_conv_layer(out)
#         out = self.act(out)
#         out = self.conv_blocks(out)
#         out = self.pooling_layer(out).squeeze(-1)  # [B, c_h]
#         out = self.dense_blocks(out)               # [B, c_h]
#         spk_embed = self.output_layer(out)         # [B, c_out]

#         # 如果只需要speaker embedding
#         if not return_aux and not return_mfcc:
#             return spk_embed
        
#         # 准备返回值
#         results = [spk_embed]
        
#         # Domain classification with GRL
#         if return_aux:
#             grl_domain = GRL(lambda_=grl_lambda)
#             grl_f0 = GRL(lambda_=grl_lambda)
        
#             domain_logits = self.domain_head(grl_domain(spk_embed))  # [B, 2]

#             # F0 distribution classification with GRL
#             # 输出[B, 25]，然后reshape为[B, 5, 5]
#             f0_dist_logits = self.f0_dist_head(grl_f0(spk_embed))  # [B, 25]
#             f0_dist_logits = f0_dist_logits.view(-1, 5, 5)  # [B, 5, 5]: 5个分位数，每个5分类
            
#             results.extend([domain_logits, f0_dist_logits])
        
#         # MFCC prediction
#         if return_mfcc:
#             mfcc_pred = self.mfcc_head(spk_embed)  # [B, 3]: mfcc_2_mean, mfcc_4_mean, mfcc_10_mean
#             results.append(mfcc_pred)
        
#         return tuple(results) if len(results) > 1 else results[0]