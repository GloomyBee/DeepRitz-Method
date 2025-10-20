"""
losses_pinn.py: 基于物理信息神经网络（PINN）/配点法的损失函数定义模块
"""

import torch
from typing import Tuple


def compute_pde_loss(model: torch.nn.Module, data_body: torch.Tensor, source_func) -> torch.Tensor:
    """
    计算偏微分方程（PDE）残差损失。
    对于泊松方程 -∇²u = f，残差定义为 R = ∇²u + f。
    本函数计算残差的均方误差 E[R²]。

    Args:
        model (torch.nn.Module): 神经网络模型。
        data_body (torch.Tensor): 区域内部的采样点（配点），需要设置 requires_grad=True。
                                 形状为 [batch_size, 2]。
        source_func (callable): 源项函数 f(data)，返回一个与 data_body 形状匹配的张量。

    Returns:
        torch.Tensor: PDE残差损失（一个标量）。
    """
    # 确保 data_body 可以进行梯度计算
    if not data_body.requires_grad:
        raise ValueError("`data_body` must have `requires_grad=True` for second-order derivatives.")

    # 1. 前向传播，获取网络输出 u(x, y)
    output_body = model(data_body)

    # 2. 自动微分计算拉普拉斯算子 ∇²u
    # 2.1 计算一阶导数 ∇u = (du/dx, du/dy)
    # create_graph=True 是计算高阶导数的关键
    grad_output = torch.autograd.grad(
        outputs=output_body,
        inputs=data_body,
        grad_outputs=torch.ones_like(output_body),
        create_graph=True
    )[0]
    du_dx = grad_output[:, 0:1]
    du_dy = grad_output[:, 1:2]

    # 2.2 计算二阶导数 d²u/dx² 和 d²u/dy²
    d2u_dx2 = torch.autograd.grad(
        outputs=du_dx,
        inputs=data_body,
        grad_outputs=torch.ones_like(du_dx),
        create_graph=True
    )[0][:, 0:1]

    d2u_dy2 = torch.autograd.grad(
        outputs=du_dy,
        inputs=data_body,
        grad_outputs=torch.ones_like(du_dy),
        create_graph=True  # 对于某些复杂模型或损失函数，保留图可能有用
    )[0][:, 1:2]

    # 2.3 得到拉普拉斯算子
    laplacian_u = d2u_dx2 + d2u_dy2

    # 3. 计算源项 f(x, y)
    source_term = source_func(data_body).to(data_body.device)

    # 4. 计算 PDE 残差 R = ∇²u + f
    pde_residual = laplacian_u + source_term

    # 5. 计算残差的均方误差作为损失
    loss_pde = torch.mean(pde_residual ** 2)

    return loss_pde


def compute_bc_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    计算边界条件（Boundary Condition）损失。
    采用网络输出与目标值的均方误差。

    Args:
        output (torch.Tensor): 模型在边界上的输出 [batch_size, 1]。
        target (torch.Tensor): 边界上的目标值（真实解） [batch_size, 1]。

    Returns:
        torch.Tensor: 边界损失（一个标量）。
    """
    loss_bc = torch.mean((output - target) ** 2)
    return loss_bc


def compute_total_pinn_loss(pde_loss: torch.Tensor, bc_loss: torch.Tensor, penalty: float) -> torch.Tensor:
    """
    计算加权的总损失，用于PINN方法。

    Args:
        pde_loss (torch.Tensor): PDE残差损失。
        bc_loss (torch.Tensor): 边界条件损失。
        penalty (float): 边界损失的权重（惩罚系数）。

    Returns:
        torch.Tensor: 加权后的总损失。
    """
    return pde_loss + penalty * bc_loss

