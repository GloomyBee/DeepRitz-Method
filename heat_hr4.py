import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time


# ============================================================================
# 1. 物理问题定义 (严格对应 Fig 7.16 & heat.py)
# ============================================================================
class HeatProblem:
    def __init__(self):
        self.k = 20.0  # 导热系数
        self.s = 50.0  # 热源
        self.T_left = 20.0  # Dirichlet
        self.q_top = -100.0  # Neumann (流入)

        self.area_domain = 1.5
        self.len_left = 1.0
        self.len_top = 1.0


# ============================================================================
# 2. 采样器
# ============================================================================
class TrapezoidSampler:
    @staticmethod
    def sample_domain(n_samples):
        # 期望接受率约 0.75；多采样避免偶发 batch 不足导致训练噪声/shape 不稳定
        x = np.random.uniform(0, 2, n_samples * 3)
        y = np.random.uniform(0, 1, n_samples * 3)
        mask = (x + y <= 2.0)
        return np.column_stack([x[mask], y[mask]])[:n_samples]

    @staticmethod
    def sample_left(n_samples):
        x = np.zeros(n_samples)
        y = np.random.uniform(0, 1, n_samples)
        return np.column_stack([x, y])

    @staticmethod
    def sample_top(n_samples):
        x = np.random.uniform(0, 1, n_samples)
        y = np.ones(n_samples)
        return np.column_stack([x, y])


# ============================================================================
# 3. 神经网络
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


def get_grad(f, x, create_graph=True):
    return torch.autograd.grad(f, x, torch.ones_like(f), create_graph=create_graph)[0]


# ============================================================================
# 4. 训练逻辑: The Golden Standard HR-DRM
# ============================================================================
def train_hr_golden():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    p = HeatProblem()

    # 初始化
    model_T = MixedNet(1).to(device)
    model_q = MixedNet(2).to(device)

    # 学习率: q 稍大，T 稍小
    opt_T = torch.optim.Adam(model_T.parameters(), lr=1e-3)
    opt_q = torch.optim.Adam(model_q.parameters(), lr=2e-3)  # Critic 跑快点

    scheduler_T = torch.optim.lr_scheduler.StepLR(opt_T, step_size=3000, gamma=0.9)
    scheduler_q = torch.optim.lr_scheduler.StepLR(opt_q, step_size=3000, gamma=0.9)

    # -------------------------------------------------------
    # Phase 0: 预热 (Anchor Boundary)
    # -------------------------------------------------------
    print(">>> Phase 0: Anchoring Dirichlet Boundary...")
    for i in range(1001):
        x_left = torch.tensor(TrapezoidSampler.sample_left(500), dtype=torch.float32, device=device)
        loss_pre = torch.mean((model_T(x_left) - p.T_left) ** 2)
        opt_T.zero_grad()
        loss_pre.backward()
        opt_T.step()
    print(">>> Done.\n")

    # -------------------------------------------------------
    # Phase 1: 对抗训练 (Min-Max)
    # -------------------------------------------------------
    start_time = time.time()
    loss_history = []

    n_steps = 2000  # 你之前的设置
    batch_dom = 4000
    batch_bc = 1000
    q_steps = 5  # Critic:Actor = 5:1

    for step in range(n_steps):
        # 1. 采样
        x_dom = torch.tensor(TrapezoidSampler.sample_domain(batch_dom), dtype=torch.float32,
                             device=device).requires_grad_(True)
        x_left = torch.tensor(TrapezoidSampler.sample_left(batch_bc), dtype=torch.float32, device=device)
        x_top = torch.tensor(TrapezoidSampler.sample_top(batch_bc), dtype=torch.float32, device=device)

        # ===================================================
        # Step A: 更新 q (Critic) -> Maximize Π
        # ===================================================
        # 初始计算 (为了第一次 backward)
        T_full = model_T(x_dom)
        grad_T_full = get_grad(T_full, x_dom, create_graph=False)
        T_val = T_full.detach()
        grad_T = grad_T_full.detach()

        # 循环多次更新 Critic
        for _ in range(q_steps):
            q_pred = model_q(x_dom)

            term_energy = - (0.5 / p.k) * torch.sum(q_pred ** 2, dim=1, keepdim=True)
            term_coupling = - torch.sum(q_pred * grad_T, dim=1, keepdim=True)
            term_source = - p.s * T_val

            J_dom_q = torch.mean(term_energy + term_coupling + term_source) * p.area_domain

            q_left = model_q(x_left)
            T_left_val = model_T(x_left).detach()
            q_dot_n = -q_left[:, 0:1]
            J_bc_q = torch.mean(q_dot_n * (T_left_val - p.T_left)) * p.len_left

            # 这里的 loss_q 就是你在最后想打印的
            loss_q = -(J_dom_q + J_bc_q)

            opt_q.zero_grad()
            loss_q.backward()
            torch.nn.utils.clip_grad_norm_(model_q.parameters(), 1.0)
            opt_q.step()

        # ===================================================
        # Step B: 更新 T (Actor) -> Minimize Π
        # ===================================================

        q_fixed = model_q(x_dom).detach()
        T_pred = model_T(x_dom)
        grad_T_new = get_grad(T_pred, x_dom, create_graph=True)

        term_coupling_T = - torch.sum(q_fixed * grad_T_new, dim=1, keepdim=True)
        term_source_T = - p.s * T_pred

        J_dom_T = torch.mean(term_coupling_T + term_source_T) * p.area_domain

        q_left_fixed = model_q(x_left).detach()
        q_dot_n_fixed = -q_left_fixed[:, 0:1]
        T_bc_left = model_T(x_left)
        J_bc_dirichlet_T = torch.mean(q_dot_n_fixed * (T_bc_left - p.T_left)) * p.len_left

        T_top = model_T(x_top)
        J_bc_neumann_T = torch.mean(p.q_top * T_top) * p.len_top

        loss_T = J_dom_T + J_bc_dirichlet_T + J_bc_neumann_T

        opt_T.zero_grad()
        loss_T.backward()
        torch.nn.utils.clip_grad_norm_(model_T.parameters(), 1.0)
        opt_T.step()

        scheduler_T.step()
        scheduler_q.step()
        loss_history.append(loss_T.item())

        # [修改] 每 100 步打印，且包含 Loss_q
        if step % 100 == 0:
            with torch.no_grad():
                bc_err = torch.mean((model_T(x_left) - p.T_left) ** 2)
                phys_err = torch.mean((q_fixed + p.k * grad_T_new.detach()) ** 2)
            print(
                f"Step {step}: Loss_T={loss_T.item():.2f}, Loss_q={loss_q.item():.2f}, BC_Err={bc_err:.6f}, Phys_Err={phys_err:.4f}")

    return model_T, model_q


# ============================================================================
# 5. 可视化 (验证最终真理)
# ============================================================================
def plot_golden_results(model_T, model_q):
    model_T.eval()
    model_q.eval()
    device = next(model_T.parameters()).device

    fig = plt.figure(figsize=(15, 5))
    x = np.linspace(0, 2, 200)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)
    pts = torch.tensor(np.column_stack([X.ravel(), Y.ravel()]), dtype=torch.float32).to(device)
    mask = (X + Y > 2.0)

    # 1. 温度场
    ax1 = plt.subplot(1, 3, 1)
    with torch.no_grad():
        T_pred = model_T(pts).cpu().numpy().reshape(X.shape)
    T_pred[mask] = np.nan
    c1 = ax1.contourf(X, Y, T_pred, levels=50, cmap='inferno')
    plt.colorbar(c1, ax=ax1, label='T')
    ax1.set_title("Temperature (Correct Physics)")
    ax1.plot([0, 2, 1, 0, 0], [0, 0, 1, 1, 0], 'k')

    # 2. 物理一致性 |q_net + k∇T|
    ax2 = plt.subplot(1, 3, 2)
    pts.requires_grad_(True)
    T_val = model_T(pts)
    grad_T = get_grad(T_val, pts, create_graph=False)
    p = HeatProblem()
    q_physics = -p.k * grad_T
    with torch.no_grad():
        q_nn = model_q(pts)
        err = torch.sqrt(torch.sum((q_nn - q_physics) ** 2, dim=1)).cpu().numpy().reshape(X.shape)
    err[mask] = np.nan
    c2 = ax2.contourf(X, Y, err, levels=50, cmap='viridis')
    plt.colorbar(c2, ax=ax2, label='Error')
    ax2.set_title("Constitutive Error (Should be small)")
    ax2.plot([0, 2, 1, 0, 0], [0, 0, 1, 1, 0], 'k')

    # 3. 沿 y=0.5 的温度剖面对比
    ax3 = plt.subplot(1, 3, 3)
    x_line = np.linspace(0, 1.5, 100)
    y_line = np.ones_like(x_line) * 0.5
    pts_line = torch.tensor(np.column_stack([x_line, y_line]), dtype=torch.float32).to(device)
    with torch.no_grad():
        T_line = model_T(pts_line).cpu().numpy()
    ax3.plot(x_line, T_line, 'r-', linewidth=2, label='HR Solution')
    ax3.set_title("Profile at y=0.5")
    ax3.set_xlabel("x")
    ax3.set_ylabel("T")
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig('heat_hr_golden.png')
    plt.show()


if __name__ == "__main__":
    m_T, m_q = train_hr_golden()
    plot_golden_results(m_T, m_q)
