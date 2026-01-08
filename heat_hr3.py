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
        self.T_left = 20.0
        self.q_top = -100.0
        self.area_domain = 1.5
        self.len_left = 1.0
        self.len_top = 1.0


# ============================================================================
# 2. 采样器 (保持一致)
# ============================================================================
class TrapezoidSampler:
    @staticmethod
    def sample_domain(n_samples):
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
# 3. 神经网络 (MixedNet)
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
        # 强制将输出映射到合理范围，防止初期梯度爆炸
        return self.out_layer(self.blocks(torch.tanh(self.in_layer(x))))


# ============================================================================
# 4. 辅助函数
# ============================================================================
def get_grad(f, x):
    """计算梯度 ∇f"""
    grads = torch.autograd.grad(f, x, torch.ones_like(f), create_graph=True)[0]
    return grads


# ============================================================================
# 5. 训练逻辑: 修正后的 HR-DRM
# ============================================================================
def train_hr_final():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    p = HeatProblem()

    # 初始化
    model_T = MixedNet(1).to(device)  # Actor
    model_q = MixedNet(2).to(device)  # Critic

    opt_T = torch.optim.Adam(model_T.parameters(), lr=1e-3)
    opt_q = torch.optim.Adam(model_q.parameters(), lr=1e-3)

    scheduler_T = torch.optim.lr_scheduler.StepLR(opt_T, step_size=2000, gamma=0.9)
    scheduler_q = torch.optim.lr_scheduler.StepLR(opt_q, step_size=2000, gamma=0.9)

    # -------------------------------------------------------
    # Phase 0: 预热 (保持不变)
    # -------------------------------------------------------
    print(">>> Phase 0: Pre-training Boundary Conditions...")
    for i in range(1001):
        x_left = torch.tensor(TrapezoidSampler.sample_left(500), dtype=torch.float32, device=device)
        loss_pre = torch.mean((model_T(x_left) - p.T_left) ** 2)
        opt_T.zero_grad()
        loss_pre.backward()
        opt_T.step()
    print(">>> Pre-training Done.\n")

    # -------------------------------------------------------
    # Phase 1: 对抗训练
    # -------------------------------------------------------
    start_time = time.time()
    loss_history = []

    n_steps = 10000
    batch = 4000

    for step in range(n_steps):
        # 1. 采样
        x_dom = torch.tensor(TrapezoidSampler.sample_domain(batch), dtype=torch.float32, device=device).requires_grad_(
            True)
        x_left = torch.tensor(TrapezoidSampler.sample_left(1000), dtype=torch.float32, device=device)
        x_top = torch.tensor(TrapezoidSampler.sample_top(1000), dtype=torch.float32, device=device)

        # ===================================================
        # Step A: 更新 q (Critic) -> Maximize Π_HR
        # ===================================================
        # [修正点 1]: 这里不能先 detach，否则无法对 x 求导
        # 先计算完整的 T (带梯度)
        T_full = model_T(x_dom)

        # 计算 ∇T (利用完整的计算图)
        grad_T_full = get_grad(T_full, x_dom)

        # [修正点 2]: 计算完梯度后，再 detach 成为常数，传给 q 网络使用
        T_val = T_full.detach()  # T 作为系数
        grad_T = grad_T_full.detach()  # ∇T 作为系数

        q_pred = model_q(x_dom)  # q 需要优化

        # --- 构造泛函 Π (针对 q) ---
        # Π = ∫ [-1/(2k) q·q - q·∇T + sT] dΩ
        # 注意: term_coupling 中的 grad_T 已经是 detach 过的，所以不会更新 T 网络
        term_energy = - (0.5 / p.k) * torch.sum(q_pred ** 2, dim=1, keepdim=True)
        term_coupling = - torch.sum(q_pred * grad_T, dim=1, keepdim=True)
        term_source = - p.s * T_val

        J_dom = torch.mean(term_energy + term_coupling + term_source) * p.area_domain

        # 边界项
        q_left = model_q(x_left)
        T_left_val = model_T(x_left).detach()  # T 在边界上也是常数
        q_dot_n = -q_left[:, 0:1]
        J_bc_dirichlet = torch.mean(q_dot_n * (T_left_val - p.T_left)) * p.len_left

        loss_q = -(J_dom + J_bc_dirichlet)  # Maximize -> Minimize negative

        opt_q.zero_grad()
        loss_q.backward()
        torch.nn.utils.clip_grad_norm_(model_q.parameters(), 1.0)
        opt_q.step()

        # ===================================================
        # Step B: 更新 T (Actor) -> Minimize Π_HR
        # ===================================================
        q_fixed = model_q(x_dom).detach()

        T_pred = model_T(x_dom)
        grad_T_new = get_grad(T_pred, x_dom)

        # 域内耦合项 (正确)
        term_coupling = - torch.sum(q_fixed * grad_T_new, dim=1, keepdim=True)
        term_source = - p.s * T_pred
        J_dom_T = torch.mean(term_coupling + term_source) * p.area_domain

        # --- 修正后的边界项 ---
        q_left_fixed = model_q(x_left).detach()
        q_dot_n_fixed = -q_left_fixed[:, 0:1]

        # 核心修正：使用 model_T(x_left) 而不是 T_pred
        T_bc_left = model_T(x_left)
        J_bc_dirichlet_T = torch.mean(q_dot_n_fixed * (T_bc_left - p.T_left)) * p.len_left
        # Neumann 边界
        T_top = model_T(x_top)
        # 与 heat.py 保持一致: +∫ q_top * T ds (q_top 为物理外法向热流，负值表示热流入)
        J_bc_neumann_T = torch.mean(T_top * p.q_top) * p.len_top

        loss_T = J_dom_T + J_bc_dirichlet_T + J_bc_neumann_T

        opt_T.zero_grad()
        loss_T.backward()
        torch.nn.utils.clip_grad_norm_(model_T.parameters(), 1.0)
        opt_T.step()

        # Schedulers
        scheduler_T.step()
        scheduler_q.step()

        loss_history.append(loss_T.item())

        if step % 500 == 0:
            q_mag = torch.mean(torch.sqrt(torch.sum(q_fixed ** 2, dim=1)))
            bc_err = torch.mean((model_T(x_left).detach() - p.T_left) ** 2)
            print(f"Step {step}: Loss_T={loss_T.item():.2f}, |q|={q_mag:.2f}, BC_Err={bc_err:.6f}")

    print(f"\nTraining Finished. Time: {time.time() - start_time:.1f}s")
    return model_T, model_q, loss_history

# ============================================================================
# 6. 可视化 (复用原逻辑，增强对比)
# ============================================================================
def plot_comparison(model_T, model_q):
    model_T.eval()
    model_q.eval()
    device = next(model_T.parameters()).device

    fig = plt.figure(figsize=(15, 6))

    # 1. 温度场 T
    ax1 = plt.subplot(1, 3, 1)
    x = np.linspace(0, 2, 200)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)
    pts = torch.tensor(np.column_stack([X.ravel(), Y.ravel()]), dtype=torch.float32).to(device)

    with torch.no_grad():
        T_pred = model_T(pts).cpu().numpy().reshape(X.shape)

    mask = (X + Y > 2.0)
    T_pred[mask] = np.nan
    c = ax1.contourf(X, Y, T_pred, levels=50, cmap='inferno')
    plt.colorbar(c, ax=ax1, label='T')
    ax1.set_title("Temperature (HR Method)")
    ax1.plot([0, 2, 1, 0, 0], [0, 0, 1, 1, 0], 'k')

    # 2. 热流场 q (Magnitude)
    ax2 = plt.subplot(1, 3, 2)
    with torch.no_grad():
        q_out = model_q(pts).cpu().numpy()
        q_mag = np.sqrt(q_out[:, 0] ** 2 + q_out[:, 1] ** 2).reshape(X.shape)
    q_mag[mask] = np.nan
    c2 = ax2.contourf(X, Y, q_mag, levels=50, cmap='viridis')
    plt.colorbar(c2, ax=ax2, label='|q|')
    ax2.set_title("Heat Flux Magnitude")
    ax2.plot([0, 2, 1, 0, 0], [0, 0, 1, 1, 0], 'k')

    # 3. 验证 q = -k * grad T (物理一致性检查)
    ax3 = plt.subplot(1, 3, 3)
    pts.requires_grad_(True)
    T_val = model_T(pts)
    grad_T = get_grad(T_val, pts)
    q_physics = -20.0 * grad_T  # k=20

    with torch.no_grad():
        q_nn = model_q(pts)
        err_q = torch.norm(q_nn - q_physics, dim=1).detach().cpu().numpy().reshape(X.shape)

    err_q[mask] = np.nan
    c3 = ax3.contourf(X, Y, err_q, levels=50, cmap='plasma')
    plt.colorbar(c3, ax=ax3, label='Error')
    ax3.set_title("|q_NN + k*grad(T_NN)| Error")
    ax3.plot([0, 2, 1, 0, 0], [0, 0, 1, 1, 0], 'k')

    plt.tight_layout()
    plt.savefig('heat_hr_final.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    m_T, m_q, _ = train_hr_final()
    plot_comparison(m_T, m_q)
