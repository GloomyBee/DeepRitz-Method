import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time


# ============================================================================
# 1. 物理问题定义
# ============================================================================
class HeatProblem:
    def __init__(self):
        self.k = 20.0  # 导热系数
        self.s = 50.0  # 热源
        self.T_left = 20.0  # Dirichlet BC (目标温度)
        self.q_top = -100.0  # Neumann BC (给定热流)

        self.area_domain = 1.5
        self.len_left = 1.0
        self.len_top = 1.0


# ============================================================================
# 2. 采样器 (不变)
# ============================================================================
class TrapezoidSampler:
    @staticmethod
    def sample_domain(n_samples):
        # 拒绝采样法生成梯形内部点
        x = np.random.uniform(0, 2, n_samples * 3)
        y = np.random.uniform(0, 1, n_samples * 3)
        mask = (x + y <= 2.0)
        return np.column_stack([x[mask], y[mask]])[:n_samples]

    @staticmethod
    def sample_left(n_samples):
        # x=0, 左边界
        x = np.zeros(n_samples)
        y = np.random.uniform(0, 1, n_samples)
        return np.column_stack([x, y])

    @staticmethod
    def sample_top(n_samples):
        # y=1, 上边界
        x = np.random.uniform(0, 1, n_samples)
        y = np.ones(n_samples)
        return np.column_stack([x, y])


# ============================================================================
# 3. 神经网络 (支持不同输出维度)
# ============================================================================
class ResBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.l1 = nn.Linear(width, width)
        self.l2 = nn.Linear(width, width)
        self.act = nn.Tanh()

    def forward(self, x):
        return x + self.l2(self.act(self.l1(x)))


class HeatNet(nn.Module):
    def __init__(self, out_dim=1):
        super().__init__()
        self.in_layer = nn.Linear(2, 50)
        self.blocks = nn.Sequential(*[ResBlock(50) for _ in range(3)])
        self.out_layer = nn.Linear(50, out_dim)  # out_dim=1 for T, 2 for q
        self.act = nn.Tanh()

    def forward(self, x):
        x = self.act(self.in_layer(x))
        x = self.blocks(x)
        return self.out_layer(x)


# ============================================================================
# 4. 对抗训练逻辑 (Min-Max)
# ============================================================================
def divergence(q, x):
    """计算散度 div(q)"""
    u = q[:, 0:1]
    v = q[:, 1:2]
    # create_graph=True 是对抗训练的关键
    du_dx = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0][:, 0:1]
    dv_dy = torch.autograd.grad(v, x, torch.ones_like(v), create_graph=True)[0][:, 1:2]
    return du_dx + dv_dy


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    p = HeatProblem()

    model_T = HeatNet(out_dim=1).to(device)
    model_q = HeatNet(out_dim=2).to(device)

    # [改进1] 降低学习率
    # T 网络负责"修补"误差，需要细致，所以 LR 给小一点
    # q 网络负责"寻找"漏洞，需要敏锐，LR 可以稍大一点
    opt_T = torch.optim.Adam(model_T.parameters(), lr=0.0002)
    opt_q = torch.optim.Adam(model_q.parameters(), lr=0.0005)

    n_steps = 10000  # 增加总步数
    batch = 5000
    batch_bc = 1000

    loss_history = []

    # =======================================================
    # Phase 0: 预训练 (Pre-training) - 锚定边界
    # =======================================================
    print("\n>>> Phase 0: 预训练边界条件 (Pre-training T)...")
    # 先训练 1000 步，让左边界 T 强制收敛到 20
    for i in range(1001):
        x_left = torch.tensor(TrapezoidSampler.sample_left(batch_bc), dtype=torch.float32, device=device)
        T_pred = model_T(x_left)
        loss_pre = torch.mean((T_pred - p.T_left) ** 2)

        opt_T.zero_grad()
        loss_pre.backward()
        opt_T.step()

        if i % 200 == 0:
            print(f"Pretrain Step {i}: BC Error = {loss_pre.item():.6f}")

    print("预训练完成，T 网络已锚定，开始对抗训练。\n")

    # =======================================================
    # Phase 1: 对抗训练 (Adversarial Training)
    # =======================================================
    start_time = time.time()

    for step in range(n_steps):
        # 1. 采样
        x_dom = torch.tensor(TrapezoidSampler.sample_domain(batch), dtype=torch.float32, device=device).requires_grad_(
            True)
        x_left = torch.tensor(TrapezoidSampler.sample_left(batch_bc), dtype=torch.float32, device=device)
        x_top = torch.tensor(TrapezoidSampler.sample_top(batch_bc), dtype=torch.float32, device=device)

        # ---------------------------------------------------
        # Step A: 更新 q (Critic) - 最大化泛函
        # ---------------------------------------------------
        # 目标: 找到让泛函能量最大的流场 q (寻找 T 的漏洞)
        T_fixed = model_T(x_dom).detach()
        q_pred = model_q(x_dom)

        div_q = divergence(q_pred, x_dom)

        # 泛函 J（修正符号）
        # HR 泛函：Π = ∫[1/(2k)|q|² - T·div(q) + s·T] dΩ
        term_energy = 1.0 / (2 * p.k) * torch.sum(q_pred ** 2, dim=1, keepdim=True)
        term_equil = -T_fixed * div_q
        term_source = p.s * T_fixed
        J_dom = torch.mean(term_energy + term_equil + term_source) * p.area_domain

        # 边界项
        q_left = model_q(x_left)
        T_left_fixed = model_T(x_left).detach()
        q_dot_n = -q_left[:, 0:1]
        J_bc_dirichlet = torch.mean(q_dot_n * (T_left_fixed - p.T_left)) * p.len_left

        loss_q = -(J_dom + J_bc_dirichlet)  # 取负号以最大化

        opt_q.zero_grad()
        loss_q.backward()
        # [改进2] 梯度裁剪 (防止 q 网络突然输出无穷大)
        torch.nn.utils.clip_grad_norm_(model_q.parameters(), max_norm=1.0)
        opt_q.step()

        # ---------------------------------------------------
        # Step B: 更新 T (Actor) - 最小化泛函
        # ---------------------------------------------------
        # 目标: 调整 T 以消除 q 找到的漏洞 (满足平衡和边界)

        T_pred = model_T(x_dom)
        q_temp = model_q(x_dom)
        div_q = divergence(q_temp, x_dom)
        div_q_fixed = div_q.detach()  # 这里的 q 视为常数环境

        # 域内（修正源项处理）
        term_equil = -T_pred * div_q_fixed
        term_source = p.s * T_pred
        J_dom = torch.mean(term_equil + term_source) * p.area_domain

        # 左边界
        q_left_fixed = model_q(x_left).detach()
        T_left = model_T(x_left)
        q_dot_n = -q_left_fixed[:, 0:1]
        J_bc_dirichlet = torch.mean(q_dot_n * (T_left - p.T_left)) * p.len_left

        # 上边界 (Neumann)
        T_top = model_T(x_top)
        J_bc_neumann = -torch.mean(T_top * p.q_top) * p.len_top

        loss_T = J_dom + J_bc_dirichlet + J_bc_neumann

        opt_T.zero_grad()
        loss_T.backward()
        # [改进2] 梯度裁剪 (防止 T 网络震荡)
        torch.nn.utils.clip_grad_norm_(model_T.parameters(), max_norm=1.0)
        opt_T.step()

        loss_history.append(loss_T.item())

        # [改进3] 修改为每 100 步打印一次
        if step % 100 == 0:
            print(f"Step {step}: Loss_q(Max)={-loss_q.item():.2f}, Loss_T(Min)={loss_T.item():.2f}")

    print(f"\n训练完成！耗时: {time.time() - start_time:.1f}s")
    return model_T, model_q, loss_history


# ============================================================================
# 5. 可视化 (适配双网络)
# ============================================================================
def plot_results(model_T, model_q, loss_history):
    model_T.eval()
    model_q.eval()
    device = next(model_T.parameters()).device

    fig = plt.figure(figsize=(12, 5))

    # 1. 温度云图
    ax1 = plt.subplot(1, 2, 1)
    x = np.linspace(0, 2, 200)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)
    pts = torch.tensor(np.column_stack([X.ravel(), Y.ravel()]), dtype=torch.float32).to(device)

    with torch.no_grad():
        T_pred = model_T(pts).cpu().numpy().reshape(X.shape)

    mask = (X + Y > 2.0)
    T_pred[mask] = np.nan

    c = ax1.contourf(X, Y, T_pred, levels=100, cmap='inferno')
    plt.colorbar(c, ax=ax1, label='T (°C)')
    ax1.set_title("Temperature (HR Method)")
    ax1.plot([0, 2, 1, 0, 0], [0, 0, 1, 1, 0], 'k-')

    # 2. 验证 Dirichlet 边界是否满足 (左边界 T 应为 20)
    ax2 = plt.subplot(1, 2, 2)
    y_line = np.linspace(0, 1, 100)
    x_line = np.zeros_like(y_line)
    pts_left = torch.tensor(np.column_stack([x_line, y_line]), dtype=torch.float32).to(device)

    with torch.no_grad():
        T_left_pred = model_T(pts_left).cpu().numpy().flatten()
        q_left_pred = model_q(pts_left).cpu().numpy()
        qx_left = q_left_pred[:, 0]

    ax2.plot(y_line, T_left_pred, 'b-', label='Predicted T at x=0')
    ax2.axhline(20.0, color='r', linestyle='--', label='Target T=20')
    ax2.set_ylim(18, 22)  # 聚焦观察误差
    ax2.set_title("Left Boundary Check (x=0)")
    ax2.set_xlabel("y")
    ax2.set_ylabel("Temperature")
    ax2.legend()

    plt.tight_layout()
    plt.savefig('heat_hr_results.png', dpi=150)
    print("结果已保存至 heat_hr_results.png")
    plt.show()


if __name__ == "__main__":
    m_T, m_q, history = train()
    plot_results(m_T, m_q, history)