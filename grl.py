#!/usr/bin/env python3
"""
Gradient Reversal Layer (GRL) for Adversarial Training
梯度反转层，用于对抗训练
"""

import torch
import torch.nn as nn
import numpy as np
from torch import Tensor


class GradientReversalFunction(torch.autograd.Function):
    """
    梯度反转函数
    前向传播时保持输入不变，反向传播时反转梯度
    """
    
    @staticmethod
    def forward(ctx, x: Tensor, lambda_val: float) -> Tensor:
        """
        前向传播：直接返回输入
        
        Args:
            x: 输入张量
            lambda_val: GRL强度参数
            
        Returns:
            输入张量（不变）
        """
        ctx.lambda_val = lambda_val
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple:
        """
        反向传播：反转梯度
        
        Args:
            grad_output: 输出梯度
            
        Returns:
            反转后的梯度
        """
        return grad_output.neg() * ctx.lambda_val, None


def get_grl_lambda(current_step: int, start_step: int, total_steps: int, gamma: float = 10.0, max_lambda: float = 0.12) -> float:
    """
    Dynamic GRL lambda scheduling (DANN paper)
    
    Formula: λ(p) = max_lambda * (2 / (1 + exp(-γ * p)) - 1)
    
    Args:
        current_step: current training step
        start_step: step to start increasing lambda (before this, lambda=0)
        total_steps: total training steps for lambda to reach max_lambda
        gamma: controls the speed of lambda increase (default: 10.0)
        max_lambda: maximum lambda value (default: 0.12)
        
    Returns:
        lambda value in [0, max_lambda]
    """
    if current_step < start_step:
        return 0.0
    
    p = (current_step - start_step) / (total_steps - start_step)
    p = np.clip(p, 0.0, 1.0)  # ensure p in [0, 1]
    
    # DANN formula: smooth sigmoid curve from 0 to max_lambda
    lambda_val = max_lambda * (2.0 / (1.0 + np.exp(-gamma * p)) - 1.0)
    
    return lambda_val


class GradientReversal(nn.Module):
    """
    梯度反转层模块
    用于对抗训练，让模型无法区分不同域的特征
    """
    
    def __init__(self, lambda_val: float = 1.0):
        """
        初始化GRL层
        
        Args:
            lambda_val: GRL强度参数（默认1.0）
        """
        super(GradientReversal, self).__init__()
        self.lambda_val = lambda_val
    
    def forward(self, x: Tensor, lambda_val: float = None) -> Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量
            lambda_val: 可选的GRL强度参数（覆盖默认值）
            
        Returns:
            经过GRL处理的张量
        """
        if lambda_val is None:
            lambda_val = self.lambda_val
        
        return GradientReversalFunction.apply(x, lambda_val)


class DynamicGRL(nn.Module):
    """
    动态GRL层
    根据训练步数自动计算lambda值
    """
    
    def __init__(self, start_step: int = 20000, total_steps: int = 100000, 
                 gamma: float = 10.0, max_lambda: float = 0.12):
        """
        初始化动态GRL层
        
        Args:
            start_step: 开始GRL的步数
            total_steps: GRL达到最大值的总步数
            gamma: GRL增长速度控制参数
            max_lambda: GRL lambda的最大值
        """
        super(DynamicGRL, self).__init__()
        self.start_step = start_step
        self.total_steps = total_steps
        self.gamma = gamma
        self.max_lambda = max_lambda
        self.current_step = 0
    
    def update_step(self, step: int):
        """更新当前训练步数"""
        self.current_step = step
    
    def get_current_lambda(self) -> float:
        """获取当前的lambda值"""
        return get_grl_lambda(
            current_step=self.current_step,
            start_step=self.start_step,
            total_steps=self.total_steps,
            gamma=self.gamma,
            max_lambda=self.max_lambda
        )
    
    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            经过GRL处理的张量
        """
        lambda_val = self.get_current_lambda()
        return GradientReversalFunction.apply(x, lambda_val)
