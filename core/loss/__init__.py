"""
损失函数定义模块
"""

from .base_loss import BaseLoss, EnergyLossMixin, BoundaryLossMixin, LossCombiner
from . import losses, losses_pinn, losses_coll

# 导出具体损失计算类
from .losses import MonteCarloEnergyLoss
from .losses_pinn import PINNLoss
from .losses_coll import QuadratureEnergyLoss

__all__ = [
    # 基类和混入类
    'BaseLoss',
    'EnergyLossMixin',
    'BoundaryLossMixin',
    'LossCombiner',

    # 具体损失计算类
    'MonteCarloEnergyLoss',
    'PINNLoss',
    'QuadratureEnergyLoss',

    # 向后兼容的模块
    'losses',
    'losses_pinn',
    'losses_coll'
]