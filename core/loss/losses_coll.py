"""
配点法损失函数定义 - 专用于数值积分方法
"""

import torch
import math
from typing import Tuple


def compute_energy_loss_quadrature(output_body: torch.Tensor, grad_output: torch.Tensor, 
                               source_term: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    通过数值积分 (配点法) 计算能量损失。
    0.5 * integral(|grad(u)|^2) - integral(f*u)
    
    Args:
        output_body: 模型输出 [batch_size, 1]
        grad_output: 梯度 [batch_size, 2]
        source_term: 源项 [batch_size, 1]
        weights: 积分权重 [batch_size, 1]
        
    Returns:
        能量损失
    """
    # 被积函数: 0.5 * |grad(u)|^2 - f*u
    integrand = 0.5 * torch.sum(grad_output ** 2, dim=1, keepdim=True) - source_term * output_body

    # 数值积分: sum(integrand * weights)
    loss = torch.sum(integrand * weights)
    return loss


def compute_boundary_loss_quadrature(output_boundary: torch.Tensor, target_boundary: torch.Tensor, 
                                 penalty: float, radius: float) -> torch.Tensor:
    """
    计算边界条件损失 (配点法版本)
    
    Args:
        output_boundary: 模型边界输出 [batch_size, 1]
        target_boundary: 目标边界值 [batch_size, 1]
        penalty: 惩罚系数
        radius: 域半径
        
    Returns:
        边界损失
    """
    loss_boundary = penalty * torch.mean((output_boundary - target_boundary) ** 2) * (2 * math.pi * radius)
    return loss_boundary


def compute_total_loss_quadrature(energy_loss: torch.Tensor, boundary_loss: torch.Tensor) -> torch.Tensor:
    """
    计算总损失 (配点法版本)
    
    Args:
        energy_loss: 能量损失
        boundary_loss: 边界损失
        
    Returns:
        总损失
    """
    return energy_loss + boundary_loss