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
        通过加权求和（配点法）计算能量损失 → **标量**

        Args:
            output_body: 模型在内部点的输出 [N,1]
            grad_output: 梯度 [N,2]
            source_term: 源项 f [N,1]
            weights: 积分权重 [N,1]  (∑w_i ≈ area)

        Returns:
            能量积分标量
        """
        self.validate_inputs(output_body, grad_output, source_term, weights)

        # 1. 能量项：0.5 |∇u|² → [N,1] （Mixin 正确返回 per-point）
        energy_per_point = self.compute_energy_from_gradient(grad_output)  # [N,1]

        # 2. 加权求和 → 标量
        weighted_energy = torch.sum(energy_per_point * weights)  # 关键！

        # 3. 源项积分：调用 mixin，但强制汇总成标量（假设 mixin 返回 per-point）
        source_integral = self.compute_source_term_integral(
            output_body, source_term, weights=weights, sign=-1.0
        )  # 可能 [N,1] 或标量

        # 强制汇总（如果不是标量）
        source_integral = torch.sum(source_integral)  # 这确保总是标量

        # 4. 返回总能量损失标量
        total_energy = weighted_energy + source_integral
        return total_energy.squeeze()  # 额外防御：挤压任何剩余维度，确保标量

    def compute_boundary_loss(self, output_boundary: torch.Tensor, target_boundary: torch.Tensor,
                              penalty: float, radius: float) -> torch.Tensor:
        """
        边界 Dirichlet 惩罚（随机采样点 → 均匀平均）

        Returns:
            标量边界损失
        """
        self.validate_inputs(output_boundary, target_boundary)

        # L2 惩罚（均方误差）
        boundary_penalty = self.compute_dirichlet_penalty(
            output_boundary, target_boundary, method='l2'
        )  # 假设返回 mean((u-g)²) → 标量

        # 如果 mixin 返回 per-point，强制 mean 或 sum（取决于语义；这里用 mean，因为是平均误差）
        if boundary_penalty.dim() > 0:
            boundary_penalty = torch.mean(boundary_penalty)  # 或 sum，如果是积分形式

        # 边界长度 * 平均误差
        boundary_length = 2 * math.pi * radius
        return boundary_penalty * penalty * boundary_length  # 标量

    def compute_total_loss(self, energy_loss: torch.Tensor,
                           boundary_loss: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        两个标量相加 → 总损失标量
        """
        # 此时 energy_loss、boundary_loss 已经是标量
        return super().compute_total_loss(energy_loss, boundary_loss)


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