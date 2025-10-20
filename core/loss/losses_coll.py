# In core/loss/losses.py

import torch


# --- 原来的 (推测的) Monte Carlo 版本 ---
def compute_energy_loss(output_body, grad_output, source_term, radius):
    """
    通过蒙特卡洛方法计算能量损失。
    0.5 * integral(|grad(u)|^2) - integral(f*u)
    """
    # 被积函数: 0.5 * |grad(u)|^2 - f*u
    integrand = 0.5 * torch.sum(grad_output ** 2, dim=1) - source_term * output_body.squeeze(1)

    # Monte Carlo 积分: Area * mean(integrand)
    domain_area = torch.pi * radius ** 2
    loss = domain_area * torch.mean(integrand)
    return loss


# --- 新的 Quadrature (配点) 版本 ---
def compute_energy_loss_quadrature(output_body, grad_output, source_term, weights):
    """
    通过数值积分 (配点法) 计算能量损失。
    0.5 * integral(|grad(u)|^2) - integral(f*u)
    """
    # 被积函数: 0.5 * |grad(u)|^2 - f*u
    # torch.sum(grad_output**2, dim=1) 计算梯度的L2范数的平方
    integrand = 0.5 * torch.sum(grad_output ** 2, dim=1, keepdim=True) - source_term * output_body

    # 数值积分: sum(integrand * weights)
    loss = torch.sum(integrand * weights)
    return loss


# 你还需要修改 compute_boundary_loss 和 compute_total_loss (如果需要的话)
# 但根据现有代码，它们似乎不需要修改。
def compute_boundary_loss(output_boundary, target_boundary, penalty, radius):
    # ... (保持不变)
    return penalty * torch.mean((output_boundary - target_boundary) ** 2) * (2 * torch.pi * radius)


def compute_total_loss(energy_loss, boundary_loss):
    # ... (保持不变)
    return energy_loss + boundary_loss
