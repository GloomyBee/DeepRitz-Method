"""
训练器 定义模块
"""

from .base_trainer import BaseTrainer
from . import trainer, trainer_pinn, trainer_coll
from .trainer import Trainer
from .trainer_pinn import PINNTrainer
from .trainer_coll import CollocationTrainer

__all__ = [
    'BaseTrainer',
    'Trainer',
    'PINNTrainer',
    'CollocationTrainer',
    'trainer',
    'trainer_pinn',
    'trainer_coll'
]