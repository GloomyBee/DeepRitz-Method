"""
基础集成测试 - 验证Python-MATLAB基本通信
"""

import pytest
import torch
import numpy as np


class TestBasicIntegration:
    """基础集成测试"""

    @pytest.mark.skipif(True, reason="需要MATLAB安装")
    def test_matlab_engine_startup(self):
        """测试MATLAB引擎启动"""
        from core.interface.matlab_engine_manager import MatlabEngineManager

        manager = MatlabEngineManager()
        manager.start_engine()
        assert manager.is_running()
        manager.stop_engine()
        assert not manager.is_running()

    @pytest.mark.skipif(True, reason="需要MATLAB安装")
    def test_matlab_pde_call(self):
        """测试调用MATLAB PDE函数"""
        from core.interface.matlab_engine_manager import MatlabEngineManager
        from core.interface.data_converter import DataConverter

        manager = MatlabEngineManager()
        manager.start_engine()

        # 创建测试点
        points = np.array([[0.0, 0.0], [0.5, 0.5]])
        points_m = DataConverter.numpy_to_matlab(points)

        # 调用MATLAB函数
        result = manager.call_function('Poisson2D(1.0).source_term', points_m)

        # 验证结果
        result_np = DataConverter.matlab_to_numpy(result)
        assert result_np.shape[0] == 2
        assert np.allclose(result_np, 4.0)

        manager.stop_engine()

    @pytest.mark.skipif(True, reason="需要MATLAB安装")
    def test_matlab_loss_wrapper(self):
        """测试MATLAB损失函数包装器"""
        from core.interface.matlab_engine_manager import MatlabEngineManager
        from core.loss.matlab_loss_wrapper import MatlabLossWrapper

        manager = MatlabEngineManager()
        manager.start_engine()

        wrapper = MatlabLossWrapper(manager)

        # 创建测试数据
        output = torch.randn(10, 1)
        grad_output = torch.randn(10, 2)
        source_term = torch.ones(10, 1) * 4.0

        # 计算损失
        loss = wrapper.compute_energy_loss(output, grad_output, source_term, 1.0)

        assert isinstance(loss, torch.Tensor)
        assert loss.numel() == 1

        manager.stop_engine()


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
