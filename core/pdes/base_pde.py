"""
PDE基类定义
"""

from abc import ABC, abstractmethod
import torch
from typing import Dict, Any, Tuple


class BasePDE(ABC):
    """所有PDE的抽象基类接口"""
    
    @abstractmethod
    def source_term(self, data: torch.Tensor) -> torch.Tensor:
        """
        计算源项f(x)
        
        Args:
            data: 输入数据张量 [batch_size, dim]
            
        Returns:
            源项张量 [batch_size, 1]
        """
        pass
    
    @abstractmethod
    def exact_solution(self, data: torch.Tensor) -> torch.Tensor:
        """
        计算解析解
        
        Args:
            data: 输入数据张量 [batch_size, dim]
            
        Returns:
            解析解张量 [batch_size, 1]
        """
        pass
    
    @abstractmethod
    def boundary_condition(self, data: torch.Tensor) -> torch.Tensor:
        """
        计算边界条件
        
        Args:
            data: 边界数据张量 [batch_size, dim]
            
        Returns:
            边界条件值 [batch_size, 1]
        """
        pass