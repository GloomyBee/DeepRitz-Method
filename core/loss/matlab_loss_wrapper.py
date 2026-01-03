"""
MATLAB Loss Wrapper - 封装MATLAB损失函数计算
"""

import torch
import logging
from typing import Optional
from .base_loss import BaseLoss
from ..interface.matlab_engine_manager import MatlabEngineManager
from ..interface.data_converter import DataConverter

logger = logging.getLogger(__name__)


class MatlabLossWrapper(BaseLoss):
    """MATLAB损失函数包装器,提供与Python损失函数相同的接口"""

    def __init__(self, engine_manager: MatlabEngineManager, use_quadrature: bool = False):
        """
        初始化MATLAB损失函数包装器

        Args:
            engine_manager: MATLAB引擎管理器
            use_quadrature: 是否使用高斯求积(False=蒙特卡洛积分)
        """
        self.engine = engine_manager
        self.converter = DataConverter
        self.use_quadrature = use_quadrature

        if not self.engine.is_running():
            raise RuntimeError("MATLAB引擎未运行,请先启动引擎")

    def compute_energy_loss(self, output: torch.Tensor, grad_output: torch.Tensor,
                           source_term: torch.Tensor, radius: float,
                           weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算能量泛函损失

        Args:
            output: 模型输出 [N, 1]
            grad_output: 梯度 [N, 2]
            source_term: 源项 [N, 1]
            radius: 域半径
            weights: 求积权重(仅用于高斯求积) [N, 1]

        Returns:
            能量损失(标量张量)
        """
        try:
            # 转换数据到MATLAB
            output_m = self.converter.torch_to_matlab(output)
            grad_m = self.converter.torch_to_matlab(grad_output)
            source_m = self.converter.torch_to_matlab(source_term)

            # 调用MATLAB函数
            if self.use_quadrature and weights is not None:
                weights_m = self.converter.torch_to_matlab(weights)
                loss_m = self.engine.call_function(
                    'LossCalculator.compute_energy_loss_quad',
                    output_m, grad_m, source_m, weights_m,
                    nargout=1
                )
            else:
                loss_m = self.engine.call_function(
                    'LossCalculator.compute_energy_loss_mc',
                    output_m, grad_m, source_m, radius,
                    nargout=1
                )

            # 转换回PyTorch
            loss_tensor = self.converter.matlab_to_torch(loss_m, device=output.device)

            # 确保返回标量
            if loss_tensor.numel() == 1:
                return loss_tensor.squeeze()
            else:
                return loss_tensor.mean()

        except Exception as e:
            logger.error(f"MATLAB能量损失计算失败: {str(e)}")
            raise RuntimeError(f"MATLAB能量损失计算失败: {str(e)}") from e

    def compute_boundary_loss(self, output: torch.Tensor, target: torch.Tensor,
                             penalty: float, radius: float) -> torch.Tensor:
        """
        计算边界条件损失

        Args:
            output: 边界上的模型输出 [M, 1]
            target: 边界上的目标值 [M, 1]
            penalty: 惩罚系数
            radius: 域半径

        Returns:
            边界损失(标量张量)
        """
        try:
            # 转换数据到MATLAB
            output_m = self.converter.torch_to_matlab(output)
            target_m = self.converter.torch_to_matlab(target)

            # 调用MATLAB函数
            loss_m = self.engine.call_function(
                'LossCalculator.compute_boundary_loss',
                output_m, target_m, penalty, radius,
                nargout=1
            )

            # 转换回PyTorch
            loss_tensor = self.converter.matlab_to_torch(loss_m, device=output.device)

            # 确保返回标量
            if loss_tensor.numel() == 1:
                return loss_tensor.squeeze()
            else:
                return loss_tensor.mean()

        except Exception as e:
            logger.error(f"MATLAB边界损失计算失败: {str(e)}")
            raise RuntimeError(f"MATLAB边界损失计算失败: {str(e)}") from e

    def compute_total_loss(self, energy_loss: torch.Tensor,
                        boundary_loss: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        计算总损失

        Args:
            energy_loss: 能量损失
            boundary_loss: 边界损失
            **kwargs: 其他参数(未使用)

        Returns:
            总损失
        """
        try:
            # 转换数据到MATLAB
            energy_m = self.converter.torch_to_matlab(energy_loss.unsqueeze(0))
            boundary_m = self.converter.torch_to_matlab(boundary_loss.unsqueeze(0))

            # 调用MATLAB函数
            loss_m = self.engine.call_function(
                'LossCalculator.compute_total_loss',
                energy_m, boundary_m,
                nargout=1
            )

            # 转换回PyTorch
            loss_tensor = self.converter.matlab_to_torch(loss_m, device=energy_loss.device)

            return loss_tensor.squeeze()

        except Exception as e:
            logger.error(f"MATLAB总损失计算失败: {str(e)}")
            # 回退到简单相加
            logger.warning("回退到Python实现")
            return energy_loss + boundary_loss
