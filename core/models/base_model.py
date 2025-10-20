"""
模型基类定义
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Any


class BaseModel(nn.Module, ABC):
    """神经网络模型的基础接口"""
    
    def __init__(self, params: Dict[str, Any]):
        super().__init__()
        self.params = params
        self.device = params.get("device", "cpu")
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, input_dim]
            
        Returns:
            输出张量 [batch_size, output_dim]
        """
        pass