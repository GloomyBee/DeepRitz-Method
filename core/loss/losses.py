"""
损失函数定义 - DeepRitz方法（蒙特卡洛积分）
"""

import torch
import math
from typing import Tuple
from .base_loss import BaseLoss, EnergyLossMixin, BoundaryLossMixin


class MonteCarloEnergyLoss(BaseLoss, EnergyLossMixin, BoundaryLossMixin):
    """蒙特卡洛积分能量损失计算类"""

    def compute_energy_loss(self, output: torch.Tensor, grad_output: torch.Tensor,
                           source_term: torch.Tensor, radius: float) -> torch.Tensor:
        """
        计算能量泛函损失（蒙特卡洛积分）

        Args:
            output: 模型输出 [batch_size, 1]
            grad_output: 梯度 [batch_size, 2]
            source_term: 源项 [batch_size, 1]
            radius: 域半径

        Returns:
            能量损失
        """
        self.validate_inputs(output, grad_output, source_term)

        # 能量项：0.5 * |grad(u)|^2
        energy_term = self.compute_energy_from_gradient(grad_output)

        # 源项：-f * u
        source_term_integral = self.compute_source_term_integral(
            output, source_term, weights=None, sign=-1.0
        )

        # 修正：使用正确的蒙特卡洛积分
        area = math.pi * radius ** 2


        return torch.mean(energy_term) * area + torch.mean(source_term_integral) * area

    def compute_boundary_loss(self, output: torch.Tensor, target: torch.Tensor,
                             penalty: float, radius: float) -> torch.Tensor:
        """
        计算边界条件损失（蒙特卡洛积分）

        Args:
            output: 模型边界输出 [batch_size, 1]
            target: 目标边界值 [batch_size, 1]
            penalty: 惩罚系数
            radius: 域半径

        Returns:
            边界损失
        """
        self.validate_inputs(output, target)

        # Dirichlet边界条件惩罚
        boundary_penalty = self.compute_dirichlet_penalty(output, target, method='l2')

        # 边界积分：乘以边界长度
        boundary_length = 2 * math.pi * radius
        #return boundary_penalty * penalty * boundary_length / output.shape[0]

        return boundary_penalty * penalty * boundary_length

    def compute_total_loss(self, energy_loss: torch.Tensor,
                        boundary_loss: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        计算总损失

        Args:
            energy_loss: 能量损失
            boundary_loss: 边界损失
            **kwargs: 其他参数（未使用）

        Returns:
            总损失
        """
        return super().compute_total_loss(energy_loss, boundary_loss)


# 为了向后兼容，保留原有的函数接口
_monte_carlo_loss = MonteCarloEnergyLoss()


def compute_energy_loss(output: torch.Tensor, grad_output: torch.Tensor,
                       source_term: torch.Tensor, radius: float) -> torch.Tensor:
    """向后兼容函数"""
    return _monte_carlo_loss.compute_energy_loss(output, grad_output, source_term, radius)


def compute_boundary_loss(output: torch.Tensor, target: torch.Tensor,
                         penalty: float, radius: float) -> torch.Tensor:
    """向后兼容函数"""
    return _monte_carlo_loss.compute_boundary_loss(output, target, penalty, radius)


def compute_total_loss(energy_loss: torch.Tensor, boundary_loss: torch.Tensor) -> torch.Tensor:
    """向后兼容函数"""
    return _monte_carlo_loss.compute_total_loss(energy_loss, boundary_loss)