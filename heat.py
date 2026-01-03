import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time


# ============================================================================
# 1. 物理问题定义
# ============================================================================
class HeatProblem:
    """图 7.16 热传导问题"""

    def __init__(self):
        self.k = 20.0  # 导热系数
        self.s = 50.0  # 热源
        self.T_left = 20.0  # Dirichlet BC
        self.q_top = -100.0  # Neumann BC (Given in figure)

        # 几何面积计算
        # 梯形: (上底1 + 下底2) * 高1 / 2 = 1.5
        self.area_domain = 1.5
        self.len_left = 1.0
        self.len_top = 1.0


# ============================================================================
# 2. 梯形采样器 (DataSampler)
# ============================================================================
class TrapezoidSampler:
    @staticmethod
    def sample_domain(n_samples):
        # 拒绝采样法生成梯形内部点
        # 包围盒 [0,2] x [0,1]
        x = np.random.uniform(0, 2, n_samples * 3)  # 多生成一点以备筛选
        y = np.random.uniform(0, 1, n_samples * 3)

        # 几何条件: x + y <= 2
        mask = (x + y <= 2.0)
        valid_points = np.column_stack([x[mask], y[mask]])

        return valid_points[:n_samples]

    @staticmethod
    def sample_left(n_samples):
        # x = 0, y in [0, 1]
        x = np.zeros(n_samples)
        y = np.random.uniform(0, 1, n_samples)
        return np.column_stack([x, y])

    @staticmethod
    def sample_top(n_samples):
        # y = 1, x in [0, 1] (注意上底长度只有1)
        x = np.random.uniform(0, 1, n_samples)
        y = np.ones(n_samples)
        return np.column_stack([x, y])


# ============================================================================
# 3. 神经网络 (ResNet) - 保持不变
# ============================================================================
class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.l1 = nn.Linear(width, width)
        self.l2 = nn.Linear(width, width)
        self.act = nn.Tanh()

    def forward(self, x):
        return x + self.l2(self.act(self.l1(x)))


class HeatNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_layer = nn.Linear(2, 50)
        self.blocks = nn.Sequential(*[ResidualBlock(50) for _ in range(3)])
        self.out_layer = nn.Linear(50, 1)
        self.act = nn.Tanh()

    def forward(self, x):
        x = self.act(self.in_layer(x))
        x = self.blocks(x)
        return self.out_layer(x)


# ============================================================================
# 4. 训练逻辑
# ============================================================================
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    problem = HeatProblem()
    model = HeatNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练参数
    n_steps = 5000
    batch_domain = 5000
    batch_bound = 1000
    beta = 500.0  # Dirichlet惩罚系数

    loss_history = []

    print("\n" + "=" * 80)
    print("开始训练热传导问题")
    print("=" * 80)
    start_time = time.time()

    for step in range(n_steps):
        # 1. 动态采样
        x_dom = torch.tensor(TrapezoidSampler.sample_domain(batch_domain), dtype=torch.float32,
                             device=device).requires_grad_(True)
        x_left = torch.tensor(TrapezoidSampler.sample_left(batch_bound), dtype=torch.float32, device=device)
        x_top = torch.tensor(TrapezoidSampler.sample_top(batch_bound), dtype=torch.float32, device=device)

        # 2. 计算域内能量 Loss
        u_dom = model(x_dom)
        grads = torch.autograd.grad(u_dom, x_dom, torch.ones_like(u_dom), create_graph=True)[0]
        grad_sq = torch.sum(grads ** 2, dim=1, keepdim=True)

        # J = ∫(0.5*k*|∇T|² - s*T) dA
        energy_loss = torch.mean(0.5 * problem.k * grad_sq - problem.s * u_dom) * problem.area_domain

        # 3. 计算 Dirichlet 边界 Loss (左边 T=20) - 罚函数法
        u_left = model(x_left)
        bc_left_loss = torch.mean((u_left - problem.T_left) ** 2) * problem.len_left * beta

        # 4. 计算 Neumann 边界 Loss (上边 q=-100) - 变分积分项
        # Term = + ∫ q_n * T ds
        u_top = model(x_top)
        bc_top_loss = torch.mean(problem.q_top * u_top) * problem.len_top

        # 5. 总 Loss (注意：底边和斜边是绝热 q=0，自然满足，不需要写代码)
        loss = energy_loss + bc_left_loss + bc_top_loss

        # 更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if step % 500 == 0:
            elapsed = time.time() - start_time
            print(f"Step {step}/{n_steps}: Loss = {loss.item():.4f} (耗时: {elapsed:.1f}s)")

    total_time = time.time() - start_time
    print(f"\n训练完成！总耗时: {total_time:.1f}秒")
    print(f"最终损失: {loss_history[-1]:.4f}")

    return model, loss_history


# ============================================================================
# 5. 可视化结果
# ============================================================================
def plot_results(model, loss_history):
    """
    绘制完整的训练结果 (优化版: 修复文字重叠，提升美观度)

    Args:
        model: 训练好的模型
        loss_history: 损失历史
    """
    model.eval()
    device = next(model.parameters()).device

    # 创建2x2子图布局，增加高度以容纳外部标签
    fig = plt.figure(figsize=(14, 11))

    # ========== 子图1: 温度分布云图 ==========
    ax1 = plt.subplot(2, 2, 1)

    # 创建网格用于绘图
    x = np.linspace(0, 2, 200)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)

    # 转换为Tensor
    points = np.column_stack([X.ravel(), Y.ravel()])
    pts_tensor = torch.tensor(points, dtype=torch.float32).to(device)

    with torch.no_grad():
        T_pred = model(pts_tensor).cpu().numpy().reshape(X.shape)

    # 将几何区域外的部分设为 NaN (用于绘图遮罩)
    mask = (X + Y > 2.0)
    T_pred[mask] = np.nan

    # [优化] 增加 levels 到 100，让颜色过渡更平滑
    contour = ax1.contourf(X, Y, T_pred, levels=100, cmap='inferno')
    plt.colorbar(contour, ax=ax1, label='Temperature (°C)')

    # [优化] pad=20 抬高标题，防止与下方标注重叠
    ax1.set_title("Temperature Distribution", fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel("x (m)", fontsize=12)
    ax1.set_ylabel("y (m)", fontsize=12)
    ax1.set_aspect('equal')

    # 画出边界框 [优化] 线条稍微细一点更精致
    ax1.plot([0, 2, 1, 0, 0], [0, 0, 1, 1, 0], 'k-', linewidth=1.5)

    # [优化] 使用 annotate 替代 text，用箭头指向，避免文字挤在一起
    # 1. 左侧边界 (Dirichlet)
    ax1.annotate('T=20°C\n(Dirichlet)', xy=(0, 0.5), xytext=(-0.5, 0.5),
                 arrowprops=dict(arrowstyle="->", color='blue'),
                 fontsize=10, ha='center', va='center', color='blue',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='blue'))

    # 2. 上方边界 (Neumann)
    ax1.annotate('q=-100 W/m²\n(Neumann)', xy=(0.5, 1.0), xytext=(0.5, 1.25),
                 arrowprops=dict(arrowstyle="->", color='red'),
                 fontsize=10, ha='center', va='center', color='red',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='red'))

    # 3. 绝热边界 (示意)
    ax1.annotate('Adiabatic (q=0)', xy=(1.0, 0.0), xytext=(1.0, -0.25),
                 arrowprops=dict(arrowstyle="->", color='black'),
                 fontsize=10, ha='center', va='center',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # ========== 子图2: 损失曲线 ==========
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(loss_history, linewidth=1.5, color='blue')
    ax2.set_xlabel('Training Steps', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Training Loss Curve', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    # 添加最终损失值标注
    final_loss = loss_history[-1]
    ax2.text(0.95, 0.95, f'Final Loss: {final_loss:.4f}',
             transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # ========== 子图3: 温度等值线图 ==========
    ax3 = plt.subplot(2, 2, 3)

    # 绘制等值线
    contour_lines = ax3.contour(X, Y, T_pred, levels=15, colors='black', linewidths=0.5)
    ax3.clabel(contour_lines, inline=True, fontsize=8, fmt='%.1f°C')

    # [优化] 背景云图也增加 levels
    contourf = ax3.contourf(X, Y, T_pred, levels=100, cmap='coolwarm', alpha=0.6)
    plt.colorbar(contourf, ax=ax3, label='Temperature (°C)')

    ax3.set_title("Temperature Contour Lines", fontsize=14, fontweight='bold')
    ax3.set_xlabel("x (m)", fontsize=12)
    ax3.set_ylabel("y (m)", fontsize=12)
    ax3.set_aspect('equal')
    ax3.plot([0, 2, 1, 0, 0], [0, 0, 1, 1, 0], 'k-', linewidth=1.5)

    # ========== 子图4: 温度分布统计 ==========
    ax4 = plt.subplot(2, 2, 4)

    # 提取有效温度值（去除NaN）
    valid_temps = T_pred[~np.isnan(T_pred)]

    # 绘制直方图
    ax4.hist(valid_temps, bins=50, edgecolor='black', alpha=0.7, color='orange')
    ax4.set_xlabel('Temperature (°C)', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.set_title('Temperature Distribution Histogram', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    # 添加统计信息
    stats_text = f'Min: {valid_temps.min():.2f}°C\n'
    stats_text += f'Max: {valid_temps.max():.2f}°C\n'
    stats_text += f'Mean: {valid_temps.mean():.2f}°C\n'
    stats_text += f'Std: {valid_temps.std():.2f}°C'

    ax4.text(0.95, 0.95, stats_text,
             transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    plt.tight_layout()
    # [关键] 这一步保存图片，之后你可以在文件夹里找到它
    plt.savefig('heat_results.png', dpi=300, bbox_inches='tight')
    print("\n结果图已保存到: heat_results.png")

    # 尝试弹窗显示（如果在支持的环境下）
    try:
        plt.show()
    except:
        pass


def plot_temperature_profile(model):
    """
    绘制沿特定路径的温度剖面

    Args:
        model: 训练好的模型
    """
    model.eval()
    device = next(model.parameters()).device

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ========== 剖面1: 沿x轴 (y=0.5) ==========
    x_profile = np.linspace(0, 1.5, 100)  # x+y<=2, 所以x最大到1.5
    y_profile = np.ones_like(x_profile) * 0.5

    points = np.column_stack([x_profile, y_profile])
    pts_tensor = torch.tensor(points, dtype=torch.float32).to(device)

    with torch.no_grad():
        T_x = model(pts_tensor).cpu().numpy().flatten()

    axes[0].plot(x_profile, T_x, 'b-', linewidth=2, label='y=0.5')
    axes[0].set_xlabel('x (m)', fontsize=12)
    axes[0].set_ylabel('Temperature (°C)', fontsize=12)
    axes[0].set_title('Temperature Profile along x-axis', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10)

    # ========== 剖面2: 沿y轴 (x=0.5) ==========
    x_profile2 = np.ones(100) * 0.5
    y_profile2 = np.linspace(0, 1, 100)

    points2 = np.column_stack([x_profile2, y_profile2])
    pts_tensor2 = torch.tensor(points2, dtype=torch.float32).to(device)

    with torch.no_grad():
        T_y = model(pts_tensor2).cpu().numpy().flatten()

    axes[1].plot(y_profile2, T_y, 'r-', linewidth=2, label='x=0.5')
    axes[1].set_xlabel('y (m)', fontsize=12)
    axes[1].set_ylabel('Temperature (°C)', fontsize=12)
    axes[1].set_title('Temperature Profile along y-axis', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    plt.savefig('heat_profiles.png', dpi=300, bbox_inches='tight')
    print("温度剖面图已保存到: heat_profiles.png")
    plt.show()


if __name__ == "__main__":
    # 训练模型
    trained_model, loss_history = train()

    # 绘制完整结果
    print("\n生成结果图...")
    plot_results(trained_model, loss_history)

    # 绘制温度剖面
    print("\n生成温度剖面图...")
    plot_temperature_profile(trained_model)

    print("\n" + "=" * 80)
    print("所有图表已生成完毕！")
    print("=" * 80)