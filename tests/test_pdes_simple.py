"""
简化的PDE模块pytest测试文件
避免torch依赖，专注于测试框架验证
"""

import pytest
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_import_base_pde():
    """测试是否可以导入BasePDE"""
    try:
        from core.pdes.base_pde import BasePDE
        assert True, "成功导入BasePDE"
    except ImportError as e:
        pytest.fail(f"无法导入BasePDE: {e}")

def test_import_poisson():
    """测试是否可以导入Poisson2D"""
    try:
        from core.pdes.poisson import Poisson2D
        assert True, "成功导入Poisson2D"
    except ImportError as e:
        pytest.fail(f"无法导入Poisson2D: {e}")

def test_base_pde_abstract_class():
    """测试BasePDE是抽象类"""
    from core.pdes.base_pde import BasePDE
    
    # 检查是否有抽象方法
    abstract_methods = getattr(BasePDE, '__abstractmethods__', set())
    expected_methods = {'source_term', 'exact_solution', 'boundary_condition'}
    
    assert abstract_methods == expected_methods, \
        f"期望的抽象方法: {expected_methods}, 实际: {abstract_methods}"

def test_poisson2d_class_structure():
    """测试Poisson2D类结构"""
    from core.pdes.poisson import Poisson2D
    from core.pdes.base_pde import BasePDE
    
    # 检查继承关系
    assert issubclass(Poisson2D, BasePDE), "Poisson2D应该继承自BasePDE"
    
    # 检查必需的方法是否存在
    required_methods = ['source_term', 'exact_solution', 'boundary_condition']
    for method in required_methods:
        assert hasattr(Poisson2D, method), f"Poisson2D应该有{method}方法"

@pytest.mark.parametrize("radius", [0.5, 1.0, 2.0])
def test_poisson2d_initialization(radius):
    """测试Poisson2D的初始化"""
    from core.pdes.poisson import Poisson2D
    
    try:
        pde = Poisson2D(radius=radius)
        assert pde.radius == radius, f"radius应该被设置为{radius}"
    except Exception as e:
        pytest.fail(f"Poisson2D初始化失败: {e}")

def test_poisson2d_initialization_default():
    """测试Poisson2D的默认初始化"""
    from core.pdes.poisson import Poisson2D
    
    try:
        pde = Poisson2D()
        assert hasattr(pde, 'radius'), "Poisson2D实例应该有radius属性"
        assert pde.radius > 0, "radius应该是正数"
    except Exception as e:
        pytest.fail(f"Poisson2D默认初始化失败: {e}")

def test_module_imports():
    """测试模块导入结构"""
    from core.pdes import base_pde, poisson
    
    # 检查模块是否正确导入
    assert hasattr(base_pde, 'BasePDE'), "base_pde模块应该有BasePDE类"
    assert hasattr(poisson, 'Poisson2D'), "poisson模块应该有Poisson2D类"

def test_file_structure():
    """测试文件结构是否存在"""
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 检查核心文件是否存在
    required_files = [
        os.path.join(base_path, 'core', 'pdes', '__init__.py'),
        os.path.join(base_path, 'core', 'pdes', 'base_pde.py'),
        os.path.join(base_path, 'core', 'pdes', 'poisson.py')
    ]
    
    for file_path in required_files:
        assert os.path.exists(file_path), f"文件应该存在: {file_path}"


class TestPDEInterface:
    """测试PDE接口的一致性"""
    
    def test_method_signatures(self):
        """测试方法签名的一致性"""
        from core.pdes.base_pde import BasePDE
        from core.pdes.poisson import Poisson2D
        import inspect
        
        base_methods = {
            'source_term': inspect.signature(BasePDE.source_term),
            'exact_solution': inspect.signature(BasePDE.exact_solution),
            'boundary_condition': inspect.signature(BasePDE.boundary_condition)
        }
        
        poisson_methods = {
            'source_term': inspect.signature(Poisson2D.source_term),
            'exact_solution': inspect.signature(Poisson2D.exact_solution),
            'boundary_condition': inspect.signature(Poisson2D.boundary_condition)
        }
        
        # 检查方法签名是否匹配
        for method_name in base_methods:
            base_sig = base_methods[method_name]
            poisson_sig = poisson_methods[method_name]
            assert str(base_sig) == str(poisson_sig), \
                f"方法{method_name}的签名应该匹配: {base_sig} vs {poisson_sig}"


if __name__ == "__main__":
    # 可以直接运行测试
    pytest.main([__file__, "-v"])