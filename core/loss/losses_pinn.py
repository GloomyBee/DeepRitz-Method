"""
losses_pinn.py: 基于物理信息神经网络（PINN）的损失函数定义模块
"""

import torch
from typing import Tuple
from .base_loss import BaseLoss, EnergyLossMixin, BoundaryLossMixin


class PINNLoss(BaseLoss, BoundaryLossMixin):
    """PINN损失计算类，基于PDE残差"""

    def compute_energy_loss(self, model: torch.nn.Module, data_body: torch.Tensor,
                           source_func) -> torch.Tensor:
        """
        计算PDE残差损失

        Args:
            model: 神经网络模型
            data_body: 内部采样点
            source_func: 源项函数

        Returns:
            PDE残差损失
        """
        self.validate_inputs(data_body)

        # 计算模型输出和二阶导数
        output = model(data_body)

        # 计算一阶导数
        grad_u = torch.autograd.grad(
            outputs=output,
            inputs=data_body,
            grad_outputs=torch.ones_like(output),
            create_graph=True,
            retain_graph=True
        )[0]

        # 计算二阶导数（Laplace算子）
        grad_u_x = grad_u[:, 0:1]
        grad_u_y = grad_u[:, 1:2]

        grad_u_xx = torch.autograd.grad(
            outputs=grad_u_x,
            inputs=data_body,
            grad_outputs=torch.ones_like(grad_u_x),
            create_graph=True,
            retain_graph=True
        )[0][:, 0:1]

        grad_u_yy = torch.autograd.grad(
            outputs=grad_u_y,
            inputs=data_body,
            grad_outputs=torch.ones_like(grad_u_y),
            create_graph=True,
            retain_graph=True
        )[0][:, 1:2]

        # Laplace算子：∇²u
        laplace_u = grad_u_xx + grad_u_yy

        # 源项
        source_term = source_func(data_body)

        # PDE残差：∇²u - f
        pde_residual = laplace_u + source_term

        # 返回残差的均方误差
        return torch.mean(pde_residual ** 2)

    def compute_boundary_loss(self, output: torch.Tensor, target: torch.Tensor,
                             **kwargs) -> torch.Tensor:
        """
        计算边界条件损失（L2范数）

        Args:
            output: 模型边界输出
            target: 目标边界值
            **kwargs: 其他参数

        Returns:
            边界损失
        """
        self.validate_inputs(output, target)
        return self.compute_dirichlet_penalty(output, target, method='l2')

    def compute_total_loss(self, pde_loss: torch.Tensor, boundary_loss: torch.Tensor,
                        penalty: float) -> torch.Tensor:
        """
        计算PINN总损失（带权重）

        Args:
            pde_loss: PDE残差损失
            boundary_loss: 边界损失
            penalty: 边界条件权重

        Returns:
            加权总损失（标量）
        """
        total_loss = pde_loss + penalty * boundary_loss
        return torch.mean(total_loss)


# 为了向后兼容，保留原有的函数接口
_pinn_loss = PINNLoss()


def compute_pde_loss(model: torch.nn.Module, data_body: torch.Tensor, source_func) -> torch.Tensor:
    """向后兼容函数"""
    return _pinn_loss.compute_energy_loss(model, data_body, source_func)


def compute_bc_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """向后兼容函数"""
    return _pinn_loss.compute_boundary_loss(output, target)


def compute_total_pinn_loss(pde_loss: torch.Tensor, bc_loss: torch.Tensor, penalty: float) -> torch.Tensor:
    """向后兼容函数"""
    return _pinn_loss.compute_total_loss(pde_loss, bc_loss, penalty)