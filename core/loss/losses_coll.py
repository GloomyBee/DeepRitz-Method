"""
配点法损失函数定义 - 专用于数值积分方法
"""

import torch
import math
from typing import Tuple
from .base_loss import BaseLoss, EnergyLossMixin, BoundaryLossMixin


class QuadratureEnergyLoss(BaseLoss, EnergyLossMixin, BoundaryLossMixin):
    """配点法数值积分能量损失计算类"""

    def compute_energy_loss(self, output_body: torch.Tensor, grad_output: torch.Tensor,
                           source_term: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        通过数值积分（配点法）计算能量损失

        Args:
            output_body: 模型输出
            grad_output: 梯度
            source_term: 源项
            weights: 积分权重

        Returns:
            能量损失
        """
        self.validate_inputs(output_body, grad_output, source_term, weights)

        # 能量项：0.5 * |grad(u)|^2
        energy_term = self.compute_energy_from_gradient(grad_output)

        # 源项：-f * u
        source_term_integral = self.compute_source_term_integral(
            output_body, source_term, weights=weights, sign=-1.0
        )

        # 数值积分：加权求和
        return energy_term + source_term_integral

    def compute_boundary_loss(self, output_boundary: torch.Tensor, target_boundary: torch.Tensor,
                             penalty: float, radius: float) -> torch.Tensor:
        """
        计算边界条件损失（配点法）

        Args:
            output_boundary: 模型边界输出
            target_boundary: 目标边界值
            penalty: 惩罚系数
            radius: 域半径

        Returns:
            边界损失
        """
        self.validate_inputs(output_boundary, target_boundary)

        # Dirichlet边界条件惩罚
        boundary_penalty = self.compute_dirichlet_penalty(output_boundary, target_boundary, method='l2')

        # 边界积分：乘以边界长度
        boundary_length = 2 * math.pi * radius
        return boundary_penalty * penalty * boundary_length / output_boundary.shape[0]

    def compute_total_loss(self, energy_loss: torch.Tensor, boundary_loss: torch.Tensor,
                        **kwargs) -> torch.Tensor:
        """
        计算总损失

        Args:
            energy_loss: 能量损失
            boundary_loss: 边界损失
            **kwargs: 其他参数

        Returns:
            总损失（标量）
        """
        # 确保能量损失和边界损失都是标量
        energy_scalar = torch.mean(energy_loss) if energy_loss.numel() > 1 else energy_loss
        boundary_scalar = torch.mean(boundary_loss) if boundary_loss.numel() > 1 else boundary_loss
        return super().compute_total_loss(energy_scalar, boundary_scalar)


# 为了向后兼容，保留原有的函数接口
_quadrature_loss = QuadratureEnergyLoss()


def compute_energy_loss_quadrature(output_body: torch.Tensor, grad_output: torch.Tensor,
                               source_term: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """向后兼容函数"""
    return _quadrature_loss.compute_energy_loss(output_body, grad_output, source_term, weights)


def compute_boundary_loss_quadrature(output_boundary: torch.Tensor, target_boundary: torch.Tensor,
                                 penalty: float, radius: float) -> torch.Tensor:
    """向后兼容函数"""
    return _quadrature_loss.compute_boundary_loss(output_boundary, target_boundary, penalty, radius)


def compute_total_loss_quadrature(energy_loss: torch.Tensor, boundary_loss: torch.Tensor) -> torch.Tensor:
    """向后兼容函数"""
    return _quadrature_loss.compute_total_loss(energy_loss, boundary_loss)