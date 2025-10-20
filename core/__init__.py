"""
DeepRitz核心包
"""

from .pdes import base_pde, poisson
from .models import base_model, mlp
from .data_utils import sampler
from .loss import losses_pinn,losses_coll,losses
from .trainer import trainer_coll,trainer_pinn,trainer
from . import  utils

__all__ = [
    'base_pde', 'poisson',
    'base_model', 'mlp', 
    'sampler', 'losses', 'trainer', 'utils'
]