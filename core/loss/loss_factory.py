"""
损失函数工厂模块 - 提供统一的损失函数接口
"""

import torch
from typing import Union, Dict, Any
from .base_loss import BaseLoss
from .losses import MonteCarloEnergyLoss
from .losses_pinn import PINNLoss
from .losses_coll import QuadratureEnergyLoss


class LossFactory:
    """损失函数工厂类，提供统一的创建接口"""

    @staticmethod
    def create_loss_calculator(method: str, **kwargs) -> BaseLoss:
        """
        创建损失计算器

        Args:
            method: 损失计算方法
                - 'monte_carlo': 蒙特卡洛积分（DeepRitz）
                - 'pinn': 物理信息神经网络
                - 'quadrature': 配点法数值积分
            **kwargs: 方法特定参数

        Returns:
            损失计算器实例
        """
        if method == 'monte_carlo':
            return MonteCarloEnergyLoss()
        elif method == 'pinn':
            return PINNLoss()
        elif method == 'quadrature':
            return QuadratureEnergyLoss()
        else:
            raise ValueError(f"Unsupported loss method: {method}")

    @staticmethod
    def get_available_methods() -> list:
        """
        获取可用的损失计算方法

        Returns:
            方法列表
        """
        return ['monte_carlo', 'pinn', 'quadrature']

    @staticmethod
    def create_composite_loss(loss_calculators: list,
                          weights: torch.Tensor = None) -> 'LossCombiner':
        """
        创建复合损失函数

        Args:
            loss_calculators: 损失计算器列表
            weights: 权重张量

        Returns:
            损失组合器
        """
        from .base_loss import LossCombiner

        if weights is None:
            # 等权重组合
            weights = torch.ones(len(loss_calculators))

        # 返回一个简单的字典来避免metaclass冲突
        return {
            'combiner': LossCombiner(loss_calculators),
            'calculators': loss_calculators,
            'weights': weights
        }


class UnifiedLossInterface:
    """统一损失函数接口，封装不同方法的损失计算"""

    def __init__(self, method: str, **kwargs):
        """
        初始化统一损失接口

        Args:
            method: 损失计算方法
            **kwargs: 方法参数
        """
        self.method = method
        self.kwargs = kwargs
        self.loss_calculator = LossFactory.create_loss_calculator(method, **kwargs)

    def compute_energy_loss(self, *args, **kwargs) -> torch.Tensor:
        """计算能量损失"""
        return self.loss_calculator.compute_energy_loss(*args, **kwargs)

    def compute_boundary_loss(self, *args, **kwargs) -> torch.Tensor:
        """计算边界损失"""
        return self.loss_calculator.compute_boundary_loss(*args, **kwargs)

    def compute_total_loss(self, *args, **kwargs) -> torch.Tensor:
        """计算总损失"""
        return self.loss_calculator.compute_total_loss(*args, **kwargs)

    def validate_and_compute(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """
        验证输入并计算所有损失

        Returns:
            包含所有损失的字典
        """
        self.loss_calculator.validate_inputs(*args)

        energy_loss = self.compute_energy_loss(*args, **kwargs)
        boundary_loss = self.compute_boundary_loss(*args, **kwargs)
        total_loss = self.compute_total_loss(energy_loss, boundary_loss, **kwargs)

        return {
            'energy_loss': energy_loss,
            'boundary_loss': boundary_loss,
            'total_loss': total_loss
        }


# 便利函数
def create_loss_interface(method: str, **kwargs) -> UnifiedLossInterface:
    """
    创建统一损失接口的便利函数

    Args:
        method: 损失计算方法
        **kwargs: 方法参数

    Returns:
        统一损失接口实例
    """
    return UnifiedLossInterface(method, **kwargs)


def compute_all_losses(model: torch.nn.Module, data: torch.Tensor,
                       target: torch.Tensor, source_func,
                       methods: list = None) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    计算所有可用方法的损失

    Args:
        model: 神经网络模型
        data: 输入数据
        target: 目标数据
        source_func: 源项函数
        methods: 方法列表（None表示使用所有方法）

    Returns:
        方法到损失的映射
    """
    if methods is None:
        methods = LossFactory.get_available_methods()

    results = {}

    for method in methods:
        try:
            interface = create_loss_interface(method)
            if method == 'pinn':
                losses = interface.validate_and_compute(model, data, source_func=source_func)
            else:
                grad = torch.autograd.grad(
                    outputs=model(data),
                    inputs=data,
                    grad_outputs=torch.ones_like(model(data)),
                    create_graph=True,
                    retain_graph=True
                )[0]

                losses = interface.validate_and_compute(
                    model(data), grad, source_term=source_func(data)
                )
                losses['boundary_loss'] = interface.compute_boundary_loss(
                    model(data), target, penalty=1.0, radius=1.0
                )

            results[method] = losses

        except Exception as e:
            print(f"Warning: Failed to compute {method} loss: {e}")
            results[method] = {'error': str(e)}

    return results