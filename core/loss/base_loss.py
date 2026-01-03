"""
损失函数基类定义
"""

from abc import ABC, abstractmethod
import torch
import math
from typing import Tuple, Union, Optional


class BaseLoss(ABC):
    """损失函数基类，定义统一接口和通用方法"""

    @abstractmethod
    def compute_energy_loss(self, *args, **kwargs) -> torch.Tensor:
        """
        计算能量损失（方法特定）

        Returns:
            能量损失张量
        """
        pass

    @abstractmethod
    def compute_boundary_loss(self, *args, **kwargs) -> torch.Tensor:
        """
        计算边界条件损失（方法特定）

        Returns:
            边界损失张量
        """
        pass

    def compute_total_loss(self, energy_loss: torch.Tensor,
                        boundary_loss: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        计算总损失（默认实现：简单相加）

        Args:
            energy_loss: 能量损失
            boundary_loss: 边界损失
            **kwargs: 其他参数（如惩罚系数等）

        Returns:
            总损失
        """
        return energy_loss + boundary_loss

    def validate_inputs(self, *tensors) -> None:
        """
        验证输入张量的有效性

        Args:
            *tensors: 要验证的张量列表

        Raises:
            ValueError: 当张量无效时
        """
        for i, tensor in enumerate(tensors):
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"Input {i} is not a tensor: {type(tensor)}")
            if tensor.numel() == 0:
                raise ValueError(f"Input {i} is empty")

    def compute_gradient_norm(self, grad_output: torch.Tensor,
                           method: str = 'l2') -> torch.Tensor:
        """
        计算梯度范数（通用方法）

        Args:
            grad_output: 梯度张量 [batch_size, dim]
            method: 范数类型 ('l2', 'l1', 'linf')

        Returns:
            梯度范数张量
        """
        if method == 'l2':
            return torch.sqrt(torch.sum(grad_output ** 2, dim=1, keepdim=True))
        elif method == 'l1':
            return torch.sum(torch.abs(grad_output), dim=1, keepdim=True)
        elif method == 'linf':
            return torch.max(torch.abs(grad_output), dim=1, keepdim=True)[0]
        else:
            raise ValueError(f"Unsupported norm method: {method}")

    def compute_domain_integral(self, values: torch.Tensor,
                             weights: Optional[torch.Tensor] = None,
                             radius: float = 1.0) -> torch.Tensor:
        """
        计算域积分（支持蒙特卡洛和配点法）

        Args:
            values: 被积函数值 [batch_size, 1]
            weights: 积分权重（配点法使用）
            radius: 域半径

        Returns:
            积分值
        """
        if weights is not None:
            # 配点法：加权求和
            integral = torch.sum(values * weights)
        else:
            # 蒙特卡洛积分：乘以面积
            area = math.pi * radius ** 2
            integral = torch.mean(values) * area

        return integral

    def compute_boundary_integral(self, values: torch.Tensor,
                              penalty: float = 1.0,
                              radius: float = 1.0) -> torch.Tensor:
        """
        计算边界积分

        Args:
            values: 边界函数值 [batch_size, 1]
            penalty: 惩罚系数
            radius: 域半径

        Returns:
            边界积分值
        """
        boundary_length = 2 * math.pi * radius
        return torch.mean(values ** 2) * penalty * boundary_length


class EnergyLossMixin:
    """能量损失计算的混入类，提供通用方法"""

    def compute_energy_from_gradient(self, grad_output: torch.Tensor) -> torch.Tensor:
        """
        从梯度计算能量项 0.5 * |grad(u)|^2

        Args:
            grad_output: 梯度 [batch_size, 2]

        Returns:
            能量项
        """
        return 0.5 * torch.sum(grad_output ** 2, dim=1, keepdim=True)

    def compute_source_term_integral(self, output: torch.Tensor,
                                  source_term: torch.Tensor,
                                  weights: Optional[torch.Tensor] = None,
                                  sign: float = -1.0) -> torch.Tensor:
        """
        计算源项积分 -sign * f * u

        Args:
            output: 模型输出 u
            source_term: 源项 f
            weights: 积分权重
            sign: 符号（通常为-1）
            radius: 域半径（用于蒙特卡洛积分）

        Returns:
            源项积分
        """
        integrand = sign * output * source_term
        if weights is not None:
            # 配点法：加权求和
            return torch.sum(integrand * weights)
        else:
            # 蒙特卡洛积分：返回平均值，面积因子在外部处理
            return torch.mean(integrand)


class BoundaryLossMixin:
    """边界损失计算的混入类，提供通用方法"""

    def compute_dirichlet_penalty(self, output: torch.Tensor,
                                 target: torch.Tensor,
                                 method: str = 'l2') -> torch.Tensor:
        """
        计算Dirichlet边界条件惩罚

        Args:
            output: 模型输出
            target: 目标值
            method: 损失方法 ('l2', 'l1')

        Returns:
            边界惩罚项
        """
        difference = output - target
        if method == 'l2':
            return torch.sum(difference ** 2, dim=1, keepdim=True)
        elif method == 'l1':
            return torch.sum(torch.abs(difference), dim=1, keepdim=True)
        else:
            raise ValueError(f"Unsupported boundary loss method: {method}")


class LossCombiner:
    """损失组合器，提供不同的损失组合策略"""

    @staticmethod
    def weighted_sum(losses: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        加权求和组合损失

        Args:
            losses: 损失列表 [num_losses, batch_size, 1]
            weights: 权重 [num_losses]

        Returns:
            加权总损失
        """
        total_loss = torch.zeros_like(losses[0])
        for loss, weight in zip(losses, weights):
            total_loss += weight * loss
        return total_loss

    @staticmethod
    def adaptive_weighted_sum(losses: torch.Tensor,
                           adaptive_weights: torch.Tensor) -> torch.Tensor:
        """
        自适应加权求和

        Args:
            losses: 损失列表
            adaptive_weights: 自适应权重

        Returns:
            自适应加权总损失
        """
        return torch.sum(losses * adaptive_weights.unsqueeze(-1), dim=0)