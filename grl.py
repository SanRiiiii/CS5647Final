#!/usr/bin/env python3
"""
Gradient Reversal Layer (GRL) for Adversarial Training
"""

import torch
import torch.nn as nn
import numpy as np
from torch import Tensor


class GradientReversalFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: Tensor, lambda_val: float) -> Tensor:
        ctx.lambda_val = lambda_val
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple:

        return grad_output.neg() * ctx.lambda_val, None


def get_grl_lambda(current_step: int, start_step: int, total_steps: int, gamma: float = 10.0, max_lambda: float = 0.12) -> float:
    """
    Dynamic GRL lambda scheduling 
    
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

    
    def __init__(self, lambda_val: float = 1.0):

        super(GradientReversal, self).__init__()
        self.lambda_val = lambda_val
    
    def forward(self, x: Tensor, lambda_val: float = None) -> Tensor:
        if lambda_val is None:
            lambda_val = self.lambda_val
        
        return GradientReversalFunction.apply(x, lambda_val)


class DynamicGRL(nn.Module):
    
    def __init__(self, start_step: int = 20000, total_steps: int = 100000, 
                 gamma: float = 10.0, max_lambda: float = 0.12):
        super(DynamicGRL, self).__init__()
        self.start_step = start_step
        self.total_steps = total_steps
        self.gamma = gamma
        self.max_lambda = max_lambda
        self.current_step = 0
    
    def update_step(self, step: int):
        self.current_step = step
    
    def get_current_lambda(self) -> float:
        return get_grl_lambda(
            current_step=self.current_step,
            start_step=self.start_step,
            total_steps=self.total_steps,
            gamma=self.gamma,
            max_lambda=self.max_lambda
        )
    
    def forward(self, x: Tensor) -> Tensor:
        lambda_val = self.get_current_lambda()
        return GradientReversalFunction.apply(x, lambda_val)
