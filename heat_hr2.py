import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time


# ============================================================================
# 1. 物理问题定义 [基于 HR 变分原理]
# ============================================================================
class HeatProblem:
    def __init__(self):
        self.k = 20.0  # 导热系数
        self.s = 50.0  # 内热源
        self.T_left = 20.0  # Dirichlet 边界温度
        self.q_top = -100.0  # Neumann 边界热流 (上边界)
        self.area_domain = 1.5  # 梯形面积 (1*1 + 0.5*1*1)
        self.len_left = 1.0
        self.len_top = 1.0


# ============================================================================
# 2. 神经网络架构 (混合变量模型) [cite: 74]
# ============================================================================
class ResBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh()
        )

    def forward(self, x): return x + self.net(x)


class MixedNet(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.in_layer = nn.Linear(2, 64)
        self.blocks = nn.Sequential(*[ResBlock(64) for _ in range(3)])
        self.out_layer = nn.Linear(64, out_dim)

    def forward(self, x):
        return self.out_layer(self.blocks(torch.tanh(self.in_layer(x))))


# ============================================================================
# 3. 训练逻辑: 对抗式 Min-Max 优化
# ============================================================================
def get_grad(f, x):
    """计算一阶梯度，降低对导数阶数的要求 [cite: 79]"""
    return torch.autograd.grad(f, x, torch.ones_like(f), create_graph=True)[0]


def train_hr():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    p = HeatProblem()

    # 初始化双网络
    model_T = MixedNet(1).to(device)  # Actor: 预测温度
    model_q = MixedNet(2).to(device)  # Critic: 预测热流向量 [cite: 830]

    # 不同的学习率策略: q 网络需要更敏锐地捕捉 T 的误差
    opt_T = torch.optim.Adam(model_T.parameters(), lr=0.0001)
    opt_q = torch.optim.Adam(model_q.parameters(), lr=0.0005)

    # 采样器
    from heat_hr import TrapezoidSampler as Sampler  # 复用原采样逻辑

    for step in range(15001):
        # 准备采样点
        x_dom = torch.tensor(Sampler.sample_domain(4000), dtype=torch.float32, device=device).requires_grad_(True)
        x_left = torch.tensor(Sampler.sample_left(1000), dtype=torch.float32, device=device)
        x_top = torch.tensor(Sampler.sample_top(1000), dtype=torch.float32, device=device)

        # ---------------------------------------------------
        # Step A: 更新 q (寻找泛函驻值点 - 极大化)
        # ---------------------------------------------------
        # 固定 T, 优化 q 使得泛函 Pi_HR 最大
        T_val = model_T(x_dom)
        grad_T = get_grad(T_val, x_dom).detach()  # 核心: 梯度耦合 [cite: 20]
        q_pred = model_q(x_dom)

        # 域内 Pi_HR = ∫[1/(2k)|q|² + q·∇T + s·T] dΩ（修正符号和添加源项）
        term_energy = -0.5 / p.k * torch.sum(q_pred ** 2, dim=1, keepdim=True)
        term_coupling = -torch.sum(q_pred * grad_T, dim=1, keepdim=True)
        term_source = -p.s * T_val
        J_dom = torch.mean(term_energy + term_coupling + term_source) * p.area_domain

        # Dirichlet 边界 (左边界 n=[-1, 0]) [cite: 18]
        q_left = model_q(x_left)
        T_left_err = (model_T(x_left).detach() - p.T_left)
        J_bc_dirichlet = torch.mean(-q_left[:, 0:1] * T_left_err) * p.len_left

        # 极大化过程 (取负号转为最小化) + 微量 L2 正则防止爆炸
        loss_q = -(J_dom + J_bc_dirichlet) + 1e-5 * torch.mean(q_pred ** 2)

        opt_q.zero_grad()
        loss_q.backward()
        torch.nn.utils.clip_grad_norm_(model_q.parameters(), 1.0)
        opt_q.step()

        # ---------------------------------------------------
        # Step B: 更新 T (寻找平衡态 - 最小化)
        # ---------------------------------------------------
        T_pred = model_T(x_dom)
        grad_T_new = get_grad(T_pred, x_dom)
        q_fixed = model_q(x_dom).detach()

        # 域内只保留含 T 的项（修正符号）
        # δΠ/δT = q·∇T + s
        J_dom_T = torch.mean(-torch.sum(q_fixed * grad_T_new, dim=1, keepdim=True) - p.s * T_pred) * p.area_domain

        # Dirichlet 边界项
        q_left_fixed = model_q(x_left).detach()
        J_bc_dirichlet_T = torch.mean(-q_left_fixed[:, 0:1] * (model_T(x_left) - p.T_left)) * p.len_left

        # Neumann 边界 (上边界)
        J_bc_neumann_T = torch.mean(model_T(x_top) * p.q_top) * p.len_top

        loss_T = J_dom_T + J_bc_dirichlet_T + J_bc_neumann_T

        opt_T.zero_grad()
        loss_T.backward()
        torch.nn.utils.clip_grad_norm_(model_T.parameters(), 1.0)
        opt_T.step()

        if step % 500 == 0:
            print(f"Step {step}: J_HR={-loss_q.item():.4f}, BC_Err={torch.mean(T_left_err ** 2).item():.6f}")

    return model_T, model_q


if __name__ == "__main__":
    final_T, final_q = train_hr()
    # 后续可视化逻辑同前...
