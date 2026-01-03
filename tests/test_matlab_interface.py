"""
Python-MATLAB接口单元测试
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock


class TestDataConverter:
    """DataConverter测试类"""

    def test_numpy_to_matlab_2d(self):
        """测试NumPy 2D数组转换为MATLAB"""
        from core.interface.data_converter import DataConverter

        # 创建测试数据
        numpy_array = np.array([[1.0, 2.0], [3.0, 4.0]])

        # 模拟matlab.double
        with patch('core.interface.data_converter.matlab') as mock_matlab:
            mock_matlab.double = Mock(return_value="matlab_array")

            result = DataConverter.numpy_to_matlab(numpy_array)

            # 验证调用
            mock_matlab.double.assert_called_once()
            assert result == "matlab_array"

    def test_numpy_to_matlab_1d(self):
        """测试NumPy 1D数组转换为MATLAB"""
        from core.interface.data_converter import DataConverter

        numpy_array = np.array([1.0, 2.0, 3.0])

        with patch('core.interface.data_converter.matlab') as mock_matlab:
            mock_matlab.double = Mock(return_value="matlab_array")

            result = DataConverter.numpy_to_matlab(numpy_array)

            mock_matlab.double.assert_called_once()
            assert result == "matlab_array"

    def test_torch_to_matlab_cpu(self):
        """测试PyTorch CPU张量转换为MATLAB"""
        from core.interface.data_converter import DataConverter

        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        with patch('core.interface.data_converter.matlab') as mock_matlab:
            mock_matlab.double = Mock(return_value="matlab_array")

            result = DataConverter.torch_to_matlab(tensor)

            mock_matlab.double.assert_called_once()
            assert result == "matlab_array"

    def test_torch_to_matlab_invalid_input(self):
        """测试无效输入"""
        from core.interface.data_converter import DataConverter

        with pytest.raises(ValueError, match="输入必须是torch.Tensor"):
            DataConverter.torch_to_matlab([1, 2, 3])

    def test_matlab_to_numpy(self):
        """测试MATLAB数组转换为NumPy"""
        from core.interface.data_converter import DataConverter

        # 模拟MATLAB double数组
        with patch('core.interface.data_converter.matlab') as mock_matlab:
            mock_matlab_array = MagicMock()
            mock_matlab_array.__class__ = type('double', (), {})
            mock_matlab.double = type('double', (), {})

            # 模拟np.array行为
            with patch('numpy.array', return_value=np.array([[1.0, 2.0], [3.0, 4.0]])):
                result = DataConverter.matlab_to_numpy(mock_matlab_array)

                assert isinstance(result, np.ndarray)
                assert result.shape == (2, 2)

    def test_matlab_to_torch(self):
        """测试MATLAB数组转换为PyTorch"""
        from core.interface.data_converter import DataConverter

        # 模拟MATLAB数组
        with patch.object(DataConverter, 'matlab_to_numpy', return_value=np.array([[1.0, 2.0]])):
            result = DataConverter.matlab_to_torch("mock_matlab_array", device='cpu')

            assert isinstance(result, torch.Tensor)
            assert result.device.type == 'cpu'

    def test_validate_shape(self):
        """测试形状验证"""
        from core.interface.data_converter import DataConverter

        array = np.array([[1, 2], [3, 4]])

        # 正确的形状
        assert DataConverter.validate_shape(array, (2, 2)) is True

        # 错误的形状
        assert DataConverter.validate_shape(array, (3, 2)) is False

        # 使用None表示任意维度
        assert DataConverter.validate_shape(array, (None, 2)) is True

    def test_ensure_2d_numpy(self):
        """测试确保NumPy数组是2D"""
        from core.interface.data_converter import DataConverter

        # 1D数组
        array_1d = np.array([1, 2, 3])
        result = DataConverter.ensure_2d(array_1d)
        assert result.ndim == 2
        assert result.shape == (3, 1)

        # 2D数组
        array_2d = np.array([[1, 2], [3, 4]])
        result = DataConverter.ensure_2d(array_2d)
        assert result.ndim == 2
        assert result.shape == (2, 2)

    def test_ensure_2d_torch(self):
        """测试确保PyTorch张量是2D"""
        from core.interface.data_converter import DataConverter

        # 1D张量
        tensor_1d = torch.tensor([1, 2, 3])
        result = DataConverter.ensure_2d(tensor_1d)
        assert result.ndim == 2
        assert result.shape == (3, 1)

        # 2D张量
        tensor_2d = torch.tensor([[1, 2], [3, 4]])
        result = DataConverter.ensure_2d(tensor_2d)
        assert result.ndim == 2
        assert result.shape == (2, 2)


class TestMatlabEngineManager:
    """MatlabEngineManager测试类"""

    def test_init(self):
        """测试初始化"""
        from core.interface.matlab_engine_manager import MatlabEngineManager

        manager = MatlabEngineManager(matlab_path="/path/to/matlab", startup_timeout=60)

        assert manager.matlab_path == "/path/to/matlab"
        assert manager.startup_timeout == 60
        assert manager.engine is None
        assert manager.is_running() is False

    def test_start_engine_success(self):
        """测试成功启动引擎"""
        from core.interface.matlab_engine_manager import MatlabEngineManager

        with patch('core.interface.matlab_engine_manager.matlab') as mock_matlab_module:
            mock_engine = Mock()
            mock_matlab_module.engine.start_matlab = Mock(return_value=mock_engine)

            manager = MatlabEngineManager()
            manager.start_engine()

            assert manager.is_running() is True
            assert manager.engine is mock_engine
            mock_matlab_module.engine.start_matlab.assert_called_once()

    def test_start_engine_already_running(self):
        """测试引擎已运行时再次启动"""
        from core.interface.matlab_engine_manager import MatlabEngineManager

        with patch('core.interface.matlab_engine_manager.matlab') as mock_matlab_module:
            mock_engine = Mock()
            mock_matlab_module.engine.start_matlab = Mock(return_value=mock_engine)

            manager = MatlabEngineManager()
            manager.start_engine()
            manager.start_engine()  # 第二次调用

            # 应该只调用一次
            assert mock_matlab_module.engine.start_matlab.call_count == 1

    def test_start_engine_import_error(self):
        """测试MATLAB Engine未安装"""
        from core.interface.matlab_engine_manager import MatlabEngineManager

        with patch('core.interface.matlab_engine_manager.matlab', side_effect=ImportError()):
            manager = MatlabEngineManager()

            with pytest.raises(ImportError, match="无法导入matlab.engine"):
                manager.start_engine()

    def test_stop_engine(self):
        """测试停止引擎"""
        from core.interface.matlab_engine_manager import MatlabEngineManager

        with patch('core.interface.matlab_engine_manager.matlab') as mock_matlab_module:
            mock_engine = Mock()
            mock_matlab_module.engine.start_matlab = Mock(return_value=mock_engine)

            manager = MatlabEngineManager()
            manager.start_engine()
            manager.stop_engine()

            assert manager.is_running() is False
            assert manager.engine is None
            mock_engine.quit.assert_called_once()

    def test_call_function_not_running(self):
        """测试引擎未运行时调用函数"""
        from core.interface.matlab_engine_manager import MatlabEngineManager

        manager = MatlabEngineManager()

        with pytest.raises(RuntimeError, match="MATLAB引擎未运行"):
            manager.call_function('test_func')

    def test_call_function_success(self):
        """测试成功调用函数"""
        from core.interface.matlab_engine_manager import MatlabEngineManager

        with patch('core.interface.matlab_engine_manager.matlab') as mock_matlab_module:
            mock_engine = Mock()
            mock_engine.test_func = Mock(return_value=42)
            mock_matlab_module.engine.start_matlab = Mock(return_value=mock_engine)

            manager = MatlabEngineManager()
            manager.start_engine()

            result = manager.call_function('test_func', 1, 2, 3)

            assert result == 42
            mock_engine.test_func.assert_called_once_with(1, 2, 3)

    def test_context_manager(self):
        """测试上下文管理器"""
        from core.interface.matlab_engine_manager import MatlabEngineManager

        with patch('core.interface.matlab_engine_manager.matlab') as mock_matlab_module:
            mock_engine = Mock()
            mock_matlab_module.engine.start_matlab = Mock(return_value=mock_engine)

            with MatlabEngineManager() as manager:
                assert manager.is_running() is True

            # 退出上下文后应该停止
            assert manager.is_running() is False
            mock_engine.quit.assert_called_once()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
