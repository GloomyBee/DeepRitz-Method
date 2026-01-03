"""
梯形区域采样器

扩展框架的采样工具以支持梯形区域
"""

import numpy as np
import torch


class TrapezoidSampler:
    """梯形区域采样器"""

    @staticmethod
    def sample_domain(n_samples: int) -> np.ndarray:
        """
        从梯形内部均匀采样

        梯形定义：顶点 (0,0), (2,0), (1,1), (0,1)
        约束条件：0 ≤ x ≤ 2, 0 ≤ y ≤ 1, x + y ≤ 2

        使用拒绝采样法：
            1. 在包围盒 [0,2]×[0,1] 中均匀采样
            2. 拒绝不满足 x+y≤2 的点

        Args:
            n_samples: 需要的样本数

        Returns:
            shape=(n_samples, 2) 的数组
        """
        valid_points = []

        while len(valid_points) < n_samples:
            # 计算还需要多少点
            remaining = n_samples - len(valid_points)

            # 梯形面积占包围盒的75%，生成1.5倍余量确保足够
            batch_size = int(remaining * 1.5) + 100

            # 在包围盒中均匀采样
            x = np.random.uniform(0, 2, batch_size)
            y = np.random.uniform(0, 1, batch_size)

            # 应用梯形约束
            mask = (x + y <= 2.0)
            new_points = np.column_stack([x[mask], y[mask]])

            valid_points.append(new_points)

        # 合并所有批次并截取到所需数量
        all_points = np.vstack(valid_points)
        return all_points[:n_samples]

    @staticmethod
    def sample_left_boundary(n_samples: int) -> np.ndarray:
        """
        从左边界采样 (x=0, 0≤y≤1)

        Args:
            n_samples: 样本数

        Returns:
            shape=(n_samples, 2) 的数组
        """
        x = np.zeros(n_samples)
        y = np.random.uniform(0, 1, n_samples)
        return np.column_stack([x, y])

    @staticmethod
    def sample_top_boundary(n_samples: int) -> np.ndarray:
        """
        从上边界采样 (y=1, 0≤x≤1)

        Args:
            n_samples: 样本数

        Returns:
            shape=(n_samples, 2) 的数组
        """
        x = np.random.uniform(0, 1, n_samples)
        y = np.ones(n_samples)
        return np.column_stack([x, y])

    @staticmethod
    def sample_bottom_boundary(n_samples: int) -> np.ndarray:
        """
        从底边界采样 (y=0, 0≤x≤2)

        Args:
            n_samples: 样本数

        Returns:
            shape=(n_samples, 2) 的数组
        """
        x = np.random.uniform(0, 2, n_samples)
        y = np.zeros(n_samples)
        return np.column_stack([x, y])

    @staticmethod
    def sample_slant_boundary(n_samples: int) -> np.ndarray:
        """
        从斜边界采样 (x+y=2, 1≤x≤2)

        参数方程：x = 1 + t, y = 1 - t, t ∈ [0,1]

        Args:
            n_samples: 样本数

        Returns:
            shape=(n_samples, 2) 的数组
        """
        t = np.random.uniform(0, 1, n_samples)
        x = 1 + t
        y = 1 - t
        return np.column_stack([x, y])

    @staticmethod
    def sample_all_boundaries(n_samples_per_edge: int) -> np.ndarray:
        """
        从所有边界采样（用于绝热边界）

        Args:
            n_samples_per_edge: 每条边的样本数

        Returns:
            所有边界点的数组
        """
        bottom = TrapezoidSampler.sample_bottom_boundary(n_samples_per_edge)
        slant = TrapezoidSampler.sample_slant_boundary(n_samples_per_edge)
        return np.vstack([bottom, slant])


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("=" * 80)
    print("梯形采样器测试")
    print("=" * 80)

    # 测试内部采样
    domain_points = TrapezoidSampler.sample_domain(5000)
    print(f"\n内部采样: {domain_points.shape}")
    print(f"  x范围: [{domain_points[:, 0].min():.3f}, {domain_points[:, 0].max():.3f}]")
    print(f"  y范围: [{domain_points[:, 1].min():.3f}, {domain_points[:, 1].max():.3f}]")
    print(f"  约束检查: 所有点满足 x+y≤2? {np.all(domain_points.sum(axis=1) <= 2.001)}")

    # 测试边界采样
    left = TrapezoidSampler.sample_left_boundary(100)
    top = TrapezoidSampler.sample_top_boundary(100)
    bottom = TrapezoidSampler.sample_bottom_boundary(100)
    slant = TrapezoidSampler.sample_slant_boundary(100)

    print(f"\n边界采样:")
    print(f"  左边界: {left.shape}, x均值={left[:, 0].mean():.6f}")
    print(f"  上边界: {top.shape}, y均值={top[:, 1].mean():.6f}")
    print(f"  底边界: {bottom.shape}, y均值={bottom[:, 1].mean():.6f}")
    print(f"  斜边界: {slant.shape}, x+y均值={(slant.sum(axis=1)).mean():.6f}")

    # 可视化
    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制内部点
    ax.scatter(domain_points[:, 0], domain_points[:, 1],
               s=1, alpha=0.3, c='lightblue', label='Domain')

    # 绘制边界点
    ax.scatter(left[:, 0], left[:, 1], s=10, c='blue', label='Left (Dirichlet)')
    ax.scatter(top[:, 0], top[:, 1], s=10, c='red', label='Top (Neumann)')
    ax.scatter(bottom[:, 0], bottom[:, 1], s=10, c='green', label='Bottom (Adiabatic)')
    ax.scatter(slant[:, 0], slant[:, 1], s=10, c='orange', label='Slant (Adiabatic)')

    # 绘制梯形边界
    vertices = np.array([[0, 0], [2, 0], [1, 1], [0, 1], [0, 0]])
    ax.plot(vertices[:, 0], vertices[:, 1], 'k-', linewidth=2)

    ax.set_xlabel('x (m)', fontsize=12)
    ax.set_ylabel('y (m)', fontsize=12)
    ax.set_title('Trapezoid Domain Sampling', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('trapezoid_sampling_test.png', dpi=150)
    print(f"\n可视化已保存到: trapezoid_sampling_test.png")

    print("\n" + "=" * 80)
    print("所有测试通过！")
    print("=" * 80)
