"""
PDE模块的pytest测试文件
测试BasePDE抽象基类和Poisson2D具体实现
"""

import pytest
import torch
import numpy as np
from typing import Type

# 导入要测试的模块
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pdes.base_pde import BasePDE
from core.pdes.poisson import Poisson2D


class TestBasePDE:
    """测试BasePDE抽象基类"""
    
    def test_base_pde_is_abstract(self):
        """测试BasePDE是抽象类，不能直接实例化"""
        with pytest.raises(TypeError):
            BasePDE()
    
    def test_base_pde_has_abstract_methods(self):
        """测试BasePDE定义了必要的抽象方法"""
        abstract_methods = BasePDE.__abstractmethods__
        expected_methods = {'source_term', 'exact_solution', 'boundary_condition'}
        
        assert abstract_methods == expected_methods, \
            f"BasePDE应该有抽象方法: {expected_methods}, 实际有: {abstract_methods}"


class TestPoisson2D:
    """测试Poisson2D类"""
    
    @pytest.fixture
    def poisson_pde(self):
        """创建Poisson2D实例的fixture"""
        return Poisson2D(radius=1.0)
    
    @pytest.fixture
    def test_data(self):
        """创建测试数据的fixture"""
        return torch.tensor([
            [0.0, 0.0],  # 中心点
            [0.5, 0.5],  # 内部点
            [1.0, 0.0],  # 边界点
            [0.0, 1.0],  # 边界点
            [-0.5, -0.5] # 内部点
        ], dtype=torch.float32)
    
    @pytest.fixture
    def boundary_data(self):
        """创建边界测试数据的fixture"""
        return torch.tensor([
            [1.0, 0.0],   # 右边界
            [0.0, 1.0],   # 上边界
            [-1.0, 0.0],  # 左边界
            [0.0, -1.0],  # 下边界
            [0.7071, 0.7071]  # 对角线边界点
        ], dtype=torch.float32)
    
    def test_poisson2d_inherits_from_base_pde(self, poisson_pde):
        """测试Poisson2D正确继承自BasePDE"""
        assert isinstance(poisson_pde, BasePDE), \
            "Poisson2D应该继承自BasePDE"
    
    def test_poisson2d_initialization(self):
        """测试Poisson2D初始化"""
        pde = Poisson2D(radius=1.0)
        assert pde.radius == 1.0, "radius应该被正确设置"
        
        pde_custom = Poisson2D(radius=2.0)
        assert pde_custom.radius == 2.0, "自定义radius应该被正确设置"
    
    def test_source_term(self, poisson_pde, test_data):
        """测试源项计算"""
        source = poisson_pde.source_term(test_data)
        
        # 检查输出形状
        expected_shape = (test_data.shape[0], 1)
        assert source.shape == expected_shape, \
            f"源项输出形状应该是{expected_shape}, 实际是{source.shape}"
        
        # 检查输出值（对于Poisson2D，源项应该是常数4.0）
        expected_values = torch.full((test_data.shape[0], 1), 4.0)
        assert torch.allclose(source, expected_values), \
            "源项值应该全为4.0"
    
    def test_exact_solution(self, poisson_pde, test_data):
        """测试解析解计算"""
        solution = poisson_pde.exact_solution(test_data)
        
        # 检查输出形状
        expected_shape = (test_data.shape[0], 1)
        assert solution.shape == expected_shape, \
            f"解析解输出形状应该是{expected_shape}, 实际是{solution.shape}"
        
        # 检查特定点的值
        # 对于点 (0,0)，解应该是 1 - (0^2 + 0^2) = 1
        center_solution = solution[0].item()
        assert abs(center_solution - 1.0) < 1e-6, \
            f"中心点(0,0)的解应该是1.0, 实际是{center_solution}"
        
        # 对于点 (0.5,0.5)，解应该是 1 - (0.5^2 + 0.5^2) = 0.5
        mid_solution = solution[1].item()
        assert abs(mid_solution - 0.5) < 1e-6, \
            f"点(0.5,0.5)的解应该是0.5, 实际是{mid_solution}"
        
        # 对于点 (1.0,0.0)，解应该是 1 - (1.0^2 + 0.0^2) = 0
        boundary_solution = solution[2].item()
        assert abs(boundary_solution - 0.0) < 1e-6, \
            f"边界点(1.0,0.0)的解应该是0.0, 实际是{boundary_solution}"
    
    def test_boundary_condition(self, poisson_pde, boundary_data):
        """测试边界条件计算"""
        boundary_values = poisson_pde.boundary_condition(boundary_data)
        
        # 检查输出形状
        expected_shape = (boundary_data.shape[0], 1)
        assert boundary_values.shape == expected_shape, \
            f"边界条件输出形状应该是{expected_shape}, 实际是{boundary_values.shape}"
        
        # 对于边界点，边界条件应该等于解析解
        exact_solutions = poisson_pde.exact_solution(boundary_data)
        assert torch.allclose(boundary_values, exact_solutions), \
            "边界条件应该等于解析解"
        
        # 检查边界值接近0（对于单位圆上的点）
        # 注意：由于数值精度，可能会有微小误差，使用更宽松的容差
        assert torch.all(torch.abs(boundary_values) < 1e-4), \
            "单位圆上的边界值应该接近0（容差1e-4）"
    
    def test_input_validation(self, poisson_pde):
        """测试输入数据验证"""
        # 测试空输入
        empty_data = torch.tensor([], dtype=torch.float32).reshape(0, 2)
        source = poisson_pde.source_term(empty_data)
        assert source.shape == (0, 1), "空输入应该产生空输出"
        
        solution = poisson_pde.exact_solution(empty_data)
        assert solution.shape == (0, 1), "空输入应该产生空输出"
        
        boundary = poisson_pde.boundary_condition(empty_data)
        assert boundary.shape == (0, 1), "空输入应该产生空输出"
    
    def test_dimension_mismatch(self, poisson_pde):
        """测试维度不匹配的处理"""
        # 测试错误维度的输入
        wrong_dim_data = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
        
        # 应该不会崩溃，但行为可能不确定（取决于具体实现）
        # 这里我们只测试不会抛出异常
        try:
            source = poisson_pde.source_term(wrong_dim_data)
            solution = poisson_pde.exact_solution(wrong_dim_data)
            boundary = poisson_pde.boundary_condition(wrong_dim_data)
        except Exception as e:
            pytest.fail(f"方法应该能够处理不同维度的输入，但抛出了异常: {e}")
    
    def test_tensor_dtype_consistency(self, poisson_pde):
        """测试张量数据类型的一致性"""
        # 测试不同输入数据类型
        test_points = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
        
        source = poisson_pde.source_term(test_points)
        solution = poisson_pde.exact_solution(test_points)
        boundary = poisson_pde.boundary_condition(test_points)
        
        # 检查输出类型是否与输入类型一致
        assert source.dtype == test_points.dtype, "源项输出类型应该与输入类型一致"
        assert solution.dtype == test_points.dtype, "解析解输出类型应该与输入类型一致"
        assert boundary.dtype == test_points.dtype, "边界条件输出类型应该与输入类型一致"
    
    def test_numerical_stability(self, poisson_pde):
        """测试数值稳定性"""
        # 测试极小值
        small_values = torch.tensor([[1e-10, 1e-10]], dtype=torch.float32)
        solution = poisson_pde.exact_solution(small_values)
        assert not torch.isnan(solution).any(), "极小值不应该产生NaN"
        assert not torch.isinf(solution).any(), "极小值不应该产生Inf"
        
        # 测试极大值
        large_values = torch.tensor([[1e10, 1e10]], dtype=torch.float32)
        solution = poisson_pde.exact_solution(large_values)
        assert not torch.isnan(solution).any(), "极大值不应该产生NaN"
        # 对于大值，可能会有Inf，这是预期的


@pytest.mark.parametrize("radius", [0.5, 1.0, 2.0, 5.0])
def test_poisson2d_different_radii(radius):
    """测试不同radius参数的Poisson2D"""
    pde = Poisson2D(radius=radius)
    assert pde.radius == radius
    
    # 测试一些基本功能
    test_data = torch.tensor([[0.0, 0.0], [radius/2, radius/2]], dtype=torch.float32)
    
    source = pde.source_term(test_data)
    solution = pde.exact_solution(test_data)
    boundary = pde.boundary_condition(test_data)
    
    # 检查基本属性
    assert source.shape == (2, 1)
    assert solution.shape == (2, 1)
    assert boundary.shape == (2, 1)


if __name__ == "__main__":
    # 可以直接运行测试
    pytest.main([__file__, "-v"])