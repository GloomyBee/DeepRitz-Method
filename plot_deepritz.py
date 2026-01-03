"""
绘图工具脚本 - 为DeepRitz算例生成完整的可视化图表

使用方法：
    python plot_deepritz.py

功能：
    1. 训练历史曲线（损失、L2误差、H1误差）
    2. 解的云图对比（网络解、精确解、误差分布）
    3. 误差分析图表
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import sys

# 导入主文件中的类
from deepritz_standalone import (
    RitzNet, Poisson2D, DeepRitzTrainer,
    DataSampler, Visualizer
)


def plot_comprehensive_results(model, pde, device, config,
                               loss_history, l2_errors, h1_errors):
    """
    生成综合结果图表

    Args:
        model: 训练好的模型
        pde: PDE问题
        device: 计算设备
        config: 配置参数
        loss_history: 损失历史
        l2_errors: L2误差历史
        h1_errors: H1误差历史
    """
    model.eval()
    radius = config['radius']

    # 创建3x2子图布局
    fig = plt.figure(figsize=(16, 12))

    # ========== 子图1: 训练损失曲线 ==========
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(loss_history, linewidth=1.5, color='blue', alpha=0.7)
    ax1.set_xlabel('训练步数', fontsize=12)
    ax1.set_ylabel('损失值', fontsize=12)
    ax1.set_title('训练损失曲线', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # ========== 子图2: 相对L2误差曲线 ==========
    ax2 = plt.subplot(3, 2, 2)
    eval_steps = [i * config['eval_step'] for i in range(len(l2_errors))]
    ax2.plot(eval_steps, l2_errors, 'o-', linewidth=2, markersize=4, color='red')
    ax2.set_xlabel('训练步数', fontsize=12)
    ax2.set_ylabel('相对L2误差', fontsize=12)
    ax2.set_title('相对L2误差曲线', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    # 添加最终误差标注
    final_l2 = l2_errors[-1]
    ax2.text(0.95, 0.95, f'最终误差: {final_l2:.6f}\n({final_l2*100:.2f}%)',
             transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # ========== 子图3: 相对H1误差曲线 ==========
    ax3 = plt.subplot(3, 2, 3)
    ax3.plot(eval_steps, h1_errors, 's-', linewidth=2, markersize=4, color='green')
    ax3.set_xlabel('训练步数', fontsize=12)
    ax3.set_ylabel('相对H1误差', fontsize=12)
    ax3.set_title('相对H1误差曲线', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')

    # 添加最终误差标注
    final_h1 = h1_errors[-1]
    ax3.text(0.95, 0.95, f'最终误差: {final_h1:.6f}\n({final_h1*100:.2f}%)',
             transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    # ========== 子图4: 网络解云图 ==========
    ax4 = plt.subplot(3, 2, 4)

    # 生成网格点
    x = np.linspace(-radius, radius, 100)
    y = np.linspace(-radius, radius, 100)
    X, Y = np.meshgrid(x, y)

    # 只保留圆盘内的点
    mask = X**2 + Y**2 <= radius**2
    points = np.column_stack([X[mask], Y[mask]])

    # 计算网络输出
    with torch.no_grad():
        points_tensor = torch.from_numpy(points).float().to(device)
        output = model(points_tensor).cpu().numpy()

    # 重构为网格形状
    output_grid = np.full(X.shape, np.nan)
    output_grid[mask] = output.flatten()

    im4 = ax4.contourf(X, Y, output_grid, levels=20, cmap='viridis')
    plt.colorbar(im4, ax=ax4)
    ax4.set_title('网络解 u_θ(x,y)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('x', fontsize=12)
    ax4.set_ylabel('y', fontsize=12)
    ax4.set_aspect('equal')

    # ========== 子图5: 精确解云图 ==========
    ax5 = plt.subplot(3, 2, 5)

    # 计算精确解
    with torch.no_grad():
        target = pde.exact_solution(points_tensor).cpu().numpy()

    target_grid = np.full(X.shape, np.nan)
    target_grid[mask] = target.flatten()

    im5 = ax5.contourf(X, Y, target_grid, levels=20, cmap='viridis')
    plt.colorbar(im5, ax=ax5)
    ax5.set_title('精确解 u_exact(x,y)', fontsize=14, fontweight='bold')
    ax5.set_xlabel('x', fontsize=12)
    ax5.set_ylabel('y', fontsize=12)
    ax5.set_aspect('equal')

    # ========== 子图6: 误差分布云图 ==========
    ax6 = plt.subplot(3, 2, 6)

    error_grid = np.full(X.shape, np.nan)
    error_grid[mask] = np.abs(output.flatten() - target.flatten())

    im6 = ax6.contourf(X, Y, error_grid, levels=20, cmap='hot')
    plt.colorbar(im6, ax=ax6)
    ax6.set_title('绝对误差 |u_θ - u_exact|', fontsize=14, fontweight='bold')
    ax6.set_xlabel('x', fontsize=12)
    ax6.set_ylabel('y', fontsize=12)
    ax6.set_aspect('equal')

    # 添加误差统计
    max_error = np.nanmax(error_grid)
    mean_error = np.nanmean(error_grid)
    ax6.text(0.95, 0.05, f'最大误差: {max_error:.6f}\n平均误差: {mean_error:.6f}',
             transform=ax6.transAxes, fontsize=10,
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig('deepritz_comprehensive.png', dpi=300, bbox_inches='tight')
    print("综合结果图已保存到: deepritz_comprehensive.png")
    plt.show()


def plot_solution_slices(model, pde, device, radius):
    """
    绘制解的切片图（沿x轴和y轴）

    Args:
        model: 训练好的模型
        pde: PDE问题
        device: 计算设备
        radius: 域半径
    """
    model.eval()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ========== 切片1: 沿x轴 (y=0) ==========
    x_slice = np.linspace(-radius, radius, 200)
    y_slice = np.zeros_like(x_slice)

    points = np.column_stack([x_slice, y_slice])
    points_tensor = torch.from_numpy(points).float().to(device)

    with torch.no_grad():
        output = model(points_tensor).cpu().numpy().flatten()
        target = pde.exact_solution(points_tensor).cpu().numpy().flatten()

    axes[0].plot(x_slice, output, 'b-', linewidth=2, label='网络解 u_θ')
    axes[0].plot(x_slice, target, 'r--', linewidth=2, label='精确解 u_exact')
    axes[0].set_xlabel('x', fontsize=12)
    axes[0].set_ylabel('u(x, 0)', fontsize=12)
    axes[0].set_title('沿x轴切片 (y=0)', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10)

    # ========== 切片2: 沿y轴 (x=0) ==========
    x_slice2 = np.zeros(200)
    y_slice2 = np.linspace(-radius, radius, 200)

    points2 = np.column_stack([x_slice2, y_slice2])
    points_tensor2 = torch.from_numpy(points2).float().to(device)

    with torch.no_grad():
        output2 = model(points_tensor2).cpu().numpy().flatten()
        target2 = pde.exact_solution(points_tensor2).cpu().numpy().flatten()

    axes[1].plot(y_slice2, output2, 'b-', linewidth=2, label='网络解 u_θ')
    axes[1].plot(y_slice2, target2, 'r--', linewidth=2, label='精确解 u_exact')
    axes[1].set_xlabel('y', fontsize=12)
    axes[1].set_ylabel('u(0, y)', fontsize=12)
    axes[1].set_title('沿y轴切片 (x=0)', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    plt.savefig('deepritz_slices.png', dpi=300, bbox_inches='tight')
    print("切片图已保存到: deepritz_slices.png")
    plt.show()


def main():
    """主函数 - 训练并生成所有图表"""

    print("\n" + "=" * 80)
    print("DeepRitz方法 - 完整可视化")
    print("=" * 80)

    # 配置参数（与主文件保持一致）
    config = {
        'radius': 1.0,
        'width': 100,
        'depth': 3,
        'train_steps': 5000,
        'lr': 0.001,
        'decay': 0.0001,
        'step_size': 500,
        'gamma': 0.5,
        'body_batch': 4096,
        'boundary_batch': 4096,
        'num_quad': 40000,
        'penalty': 1000,
        'eval_step': 100,
        'print_step': 500,
        'sample_step': 200,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    device = config['device']
    print(f"\n使用设备: {device}")

    # 创建PDE问题
    pde = Poisson2D(radius=config['radius'])
    print(f"\nPDE问题: {pde.name}")

    # 创建神经网络
    model = RitzNet(
        input_dim=2,
        output_dim=1,
        width=config['width'],
        depth=config['depth']
    )

    # 创建训练器
    trainer = DeepRitzTrainer(model, pde, device, config)

    # 训练模型
    loss_history, l2_errors, h1_errors = trainer.train()

    # 生成综合结果图
    print("\n生成综合结果图...")
    plot_comprehensive_results(
        model, pde, device, config,
        loss_history, l2_errors, h1_errors
    )

    # 生成切片图
    print("\n生成切片图...")
    plot_solution_slices(model, pde, device, config['radius'])

    # 使用原有的可视化工具
    print("\n生成解对比图...")
    Visualizer.plot_solution_comparison(
        model, pde, device,
        radius=config['radius'],
        n_samples=100,
        save_path='deepritz_comparison.png'
    )

    print("\n" + "=" * 80)
    print("所有图表已生成完毕！")
    print("=" * 80)
    print("\n生成的图表文件：")
    print("  1. deepritz_comprehensive.png - 综合结果图（6个子图）")
    print("  2. deepritz_slices.png - 切片对比图")
    print("  3. deepritz_comparison.png - 解对比图（4个子图）")


if __name__ == "__main__":
    main()
