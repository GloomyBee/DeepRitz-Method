"""
泊松方程实现
"""

import torch
import numpy as np
from .base_pde import BasePDE


class Poisson2D(BasePDE):
    """二维泊松方程实现 -u'' = f"""
    
    def __init__(self, radius: float = 1.0):
        self.radius = radius
    
    def source_term(self, data: torch.Tensor) -> torch.Tensor:
        """
        源项 f(x, y) = 4
        
        Args:
            data: 输入数据张量 [batch_size, 2]
            
        Returns:
            源项张量 [batch_size, 1]
        """
        return 4.0 * torch.ones([data.shape[0], 1], dtype=torch.float)
    
    def exact_solution(self, data: torch.Tensor) -> torch.Tensor:
        """
        解析解 u(x, y) = 1 - (x^2 + y^2)
        
        Args:
            data: 输入数据张量 [batch_size, 2]
            
        Returns:
            解析解张量 [batch_size, 1]
        """
        return (1.0 - torch.sum(data * data, dim=1)).unsqueeze(1)
    
    def boundary_condition(self, data: torch.Tensor) -> torch.Tensor:
        """
        边界条件（Dirichlet边界条件）
        
        Args:
            data: 边界数据张量 [batch_size, 2]
            
        Returns:
            边界条件值 [batch_size, 1]
        """
        return self.exact_solution(data)


if __name__ == "__main__":
    # 初始化 Poisson2D 实例
    pde = Poisson2D(radius=1.0)

    # 创建测试输入：3 个固定点
    test_points = torch.tensor([[0.0, 0.0], [0.5, 0.5], [1.0, 0.0]], dtype=torch.float32)

    # 测试 source_term
    source = pde.source_term(test_points)
    expected_source = torch.tensor([[4.0], [4.0], [4.0]], dtype=torch.float32)
    assert torch.allclose(source, expected_source), "Source term should be 4"
    print("Source term test passed: output shape", source.shape, "values", source.tolist())

    # 测试 exact_solution
    u = pde.exact_solution(test_points)
    expected_u = torch.tensor([[1.0], [0.5], [0.0]], dtype=torch.float32)  # 1 - (x^2 + y^2)
    assert torch.allclose(u, expected_u, atol=1e-5), "Exact solution incorrect"
    print("Exact solution test passed: output shape", u.shape, "values", u.tolist())

    # 测试 boundary_condition（使用边界点 x^2 + y^2 = 1）
    boundary_points = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=torch.float32)
    bc = pde.boundary_condition(boundary_points)
    expected_bc = pde.exact_solution(boundary_points)
    assert torch.allclose(bc, expected_bc, atol=1e-5), "Boundary condition does not match exact solution"
    print("Boundary condition test passed: output shape", bc.shape, "values", bc.tolist())

    print("All interface tests passed!")