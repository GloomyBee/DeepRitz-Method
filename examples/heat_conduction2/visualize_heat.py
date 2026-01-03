"""
热传导结果可视化脚本
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from core.models.mlp import EnhancedRitzNet
from heat_pde import HeatPDE


def load_model(model_path: str, device: str):
    """加载训练好的模型"""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['config']

    # 重建PDE
    pde = HeatPDE(
        k=config['physics']['k_thermal'],
        s=config['physics']['s_source'],
        T_left=config['physics']['T_left']
    )

    # 准备模型参数
    model_params = {
        'd': config['d'],
        'dd': config['dd'],
        'width': config['width'],
        'depth': config['depth'],
        'device': device
    }

    # 重建核函数节点
    kernel_config = config.get("kernel", {})
    nodes_per_dim = kernel_config.get("nodes_count", 10)
    scale = kernel_config.get("scale", 0.5)

    x_nodes = torch.linspace(0, 2, nodes_per_dim)
    y_nodes = torch.linspace(0, 1, nodes_per_dim)
    nodes = torch.stack(torch.meshgrid(x_nodes, y_nodes, indexing='xy'), dim=-1).reshape(-1, 2)
    mask = (nodes[:, 0] + nodes[:, 1] <= 2.0)
    nodes = nodes[mask]
    s = torch.tensor(scale, dtype=torch.float32)

    # 重建模型
    model = EnhancedRitzNet(model_params, nodes, s).to(device)
    model.pde = pde

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 尝试加载损失历史
    loss_history = checkpoint.get('loss_history', [])
    if not loss_history:
        # 如果checkpoint中没有，尝试从文件加载
        loss_file = os.path.join(os.path.dirname(model_path), 'heat_loss_history.txt')
        if os.path.exists(loss_file):
            with open(loss_file, 'r') as f:
                lines = f.readlines()[1:]  # 跳过标题行
                loss_history = [float(line.split()[1]) for line in lines if line.strip()]

    return model, pde, loss_history


def plot_results(model, pde, loss_history, save_dir: str):
    """绘制完整结果"""
    device = next(model.parameters()).device

    # 创建图形
    fig = plt.figure(figsize=(16, 10))

    # ========== 子图1: 温度分布云图 ==========
    ax1 = plt.subplot(2, 3, 1)

    # 创建网格
    x = np.linspace(0, 2, 200)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)

    # 计算温度
    points = np.column_stack([X.ravel(), Y.ravel()])
    pts_tensor = torch.tensor(points, dtype=torch.float32).to(device)

    with torch.no_grad():
        T_pred = model(pts_tensor).cpu().numpy().reshape(X.shape)

    # 遮罩梯形外的区域
    mask = (X + Y > 2.0)
    T_pred[mask] = np.nan

    # 绘制云图
    contour = ax1.contourf(X, Y, T_pred, levels=100, cmap='inferno')
    plt.colorbar(contour, ax=ax1, label='Temperature (°C)')
    ax1.set_title("Temperature Distribution", fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel("x (m)", fontsize=12)
    ax1.set_ylabel("y (m)", fontsize=12)
    ax1.set_aspect('equal')

    # 绘制边界
    vertices = np.array([[0, 0], [2, 0], [1, 1], [0, 1], [0, 0]])
    ax1.plot(vertices[:, 0], vertices[:, 1], 'k-', linewidth=2)

    # 标注边界条件
    ax1.annotate('T=20°C\\n(Dirichlet)', xy=(0, 0.5), xytext=(-0.4, 0.5),
                 arrowprops=dict(arrowstyle="->", color='blue'),
                 fontsize=9, ha='center', color='blue',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax1.annotate('q=-100 W/m²\\n(Neumann)', xy=(0.5, 1.0), xytext=(0.5, 1.2),
                 arrowprops=dict(arrowstyle="->", color='red'),
                 fontsize=9, ha='center', color='red',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # ========== 子图2: 损失曲线 ==========
    ax2 = plt.subplot(2, 3, 2)
    if loss_history:
        ax2.plot(loss_history, linewidth=1.5, color='blue')
        final_loss = loss_history[-1]
        ax2.text(0.95, 0.95, f'Final Loss: {final_loss:.4f}',
                 transform=ax2.transAxes, fontsize=10,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    else:
        ax2.text(0.5, 0.5, 'Loss history not available',
                 transform=ax2.transAxes, fontsize=12,
                 ha='center', va='center')
    ax2.set_xlabel('Training Steps', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Training Loss Curve', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    if loss_history:
        ax2.set_yscale('log')

    # ========== 子图3: 温度等值线 ==========
    ax3 = plt.subplot(2, 3, 3)
    contour_lines = ax3.contour(X, Y, T_pred, levels=15, colors='black', linewidths=0.5)
    ax3.clabel(contour_lines, inline=True, fontsize=8, fmt='%.1f°C')
    contourf = ax3.contourf(X, Y, T_pred, levels=100, cmap='coolwarm', alpha=0.6)
    plt.colorbar(contourf, ax=ax3, label='Temperature (°C)')
    ax3.set_title("Temperature Contours", fontsize=14, fontweight='bold')
    ax3.set_xlabel("x (m)", fontsize=12)
    ax3.set_ylabel("y (m)", fontsize=12)
    ax3.set_aspect('equal')
    ax3.plot(vertices[:, 0], vertices[:, 1], 'k-', linewidth=2)

    # ========== 子图4: 温度统计直方图 ==========
    ax4 = plt.subplot(2, 3, 4)
    valid_temps = T_pred[~np.isnan(T_pred)]
    ax4.hist(valid_temps, bins=50, edgecolor='black', alpha=0.7, color='orange')
    ax4.set_xlabel('Temperature (°C)', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.set_title('Temperature Distribution', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    stats_text = f'Min: {valid_temps.min():.2f}°C\\n'
    stats_text += f'Max: {valid_temps.max():.2f}°C\\n'
    stats_text += f'Mean: {valid_temps.mean():.2f}°C\\n'
    stats_text += f'Std: {valid_temps.std():.2f}°C'

    ax4.text(0.95, 0.95, stats_text,
             transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    # ========== 子图5: 沿x轴温度剖面 (y=0.5) ==========
    ax5 = plt.subplot(2, 3, 5)
    x_profile = np.linspace(0, 1.5, 100)
    y_profile = np.ones_like(x_profile) * 0.5

    points_profile = np.column_stack([x_profile, y_profile])
    pts_tensor_profile = torch.tensor(points_profile, dtype=torch.float32).to(device)

    with torch.no_grad():
        T_x = model(pts_tensor_profile).cpu().numpy().flatten()

    ax5.plot(x_profile, T_x, 'b-', linewidth=2, label='y=0.5')
    ax5.set_xlabel('x (m)', fontsize=12)
    ax5.set_ylabel('Temperature (°C)', fontsize=12)
    ax5.set_title('Temperature Profile (x-axis)', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend(fontsize=10)

    # ========== 子图6: 沿y轴温度剖面 (x=0.5) ==========
    ax6 = plt.subplot(2, 3, 6)
    x_profile2 = np.ones(100) * 0.5
    y_profile2 = np.linspace(0, 1, 100)

    points_profile2 = np.column_stack([x_profile2, y_profile2])
    pts_tensor_profile2 = torch.tensor(points_profile2, dtype=torch.float32).to(device)

    with torch.no_grad():
        T_y = model(pts_tensor_profile2).cpu().numpy().flatten()

    ax6.plot(y_profile2, T_y, 'r-', linewidth=2, label='x=0.5')
    ax6.set_xlabel('y (m)', fontsize=12)
    ax6.set_ylabel('Temperature (°C)', fontsize=12)
    ax6.set_title('Temperature Profile (y-axis)', fontsize=14, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.legend(fontsize=10)

    plt.tight_layout()

    # 保存图片
    save_path = os.path.join(save_dir, 'heat_results.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"结果图已保存到: {save_path}")

    plt.show()


def main():
    print("=" * 80)
    print("热传导结果可视化")
    print("=" * 80)

    # 设置路径
    script_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(script_dir, '../..'))
    output_dir = os.path.join(project_root, 'output')
    model_path = os.path.join(output_dir, 'heat_model.pt')

    # 检查模型文件
    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 {model_path}")
        print("请先运行 train_heat_hr.py 训练模型")
        return

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载模型
    print(f"\n加载模型: {model_path}")
    model, pde, loss_history = load_model(model_path, device)
    print(f"模型加载成功")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")

    if loss_history:
        print(f"  训练步数: {len(loss_history)}")
        print(f"  最终损失: {loss_history[-1]:.6f}")
    else:
        print(f"  警告: 未找到损失历史记录")

    # 绘制结果
    print("\n生成可视化...")
    plot_results(model, pde, loss_history, output_dir)

    print("\n" + "=" * 80)
    print("可视化完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
