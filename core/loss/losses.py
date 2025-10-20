"""
损失函数定义
"""

import torch
import math
from typing import Tuple


def compute_energy_loss(output: torch.Tensor, grad_output: torch.Tensor, 
                       source_term: torch.Tensor, radius: float) -> torch.Tensor:
    """
    计算能量泛函损失
    
    Args:
        output: 模型输出 [batch_size, 1]
        grad_output: 梯度 [batch_size, 2]
        source_term: 源项 [batch_size, 1]
        radius: 域半径
        
    Returns:
        能量损失
    """
    dfdx = grad_output[:, 0:1]
    dfdy = grad_output[:, 1:2]
    loss_body = torch.mean(0.5 * (dfdx ** 2 + dfdy ** 2) - source_term * output) * math.pi * radius**2
    return loss_body


def compute_boundary_loss(output: torch.Tensor, target: torch.Tensor, 
                         penalty: float, radius: float) -> torch.Tensor:
    """
    计算边界条件损失
    
    Args:
        output: 模型边界输出 [batch_size, 1]
        target: 目标边界值 [batch_size, 1]
        penalty: 惩罚系数
        radius: 域半径
        
    Returns:
        边界损失
    """
    loss_boundary = torch.mean((output - target) ** 2) * penalty * 2 * math.pi * radius
    return loss_boundary


def compute_total_loss(energy_loss: torch.Tensor, boundary_loss: torch.Tensor) -> torch.Tensor:
    """
    计算总损失
    
    Args:
        energy_loss: 能量损失
        boundary_loss: 边界损失
        
    Returns:
        总损失
    """
    return energy_loss + boundary_loss