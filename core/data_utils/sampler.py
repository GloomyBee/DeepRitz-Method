"""
采样工具函数
"""

import numpy as np
import torch
from typing import Tuple # 建议加上类型注解


class Utils:
    """采样和计算工具类"""
    
    @staticmethod
    def gaussian_kernel(r: torch.Tensor, s: torch.Tensor, device: str) -> torch.Tensor:
        """
        高斯核函数
        
        Args:
            r: 距离张量
            s: 核参数
            device: 设备
            
        Returns:
            高斯核值
        """
        return torch.exp(-r**2 / (2 * s**2))

    @staticmethod
    def sample_from_disk(radius: float, num_points: int) -> np.ndarray:
        """
        在圆盘内均匀采样
        
        Args:
            radius: 圆盘半径
            num_points: 采样点数
            
        Returns:
            采样点坐标 [num_points, 2]
        """
        r = np.sqrt(np.random.uniform(0, 1, num_points)) * radius
        theta = np.random.uniform(0, 2 * np.pi, num_points)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.stack((x, y), axis=-1)

    @staticmethod
    def sample_from_surface(radius: float, num_points: int) -> np.ndarray:
        """
        在圆盘边界上均匀采样
        
        Args:
            radius: 圆盘半径
            num_points: 采样点数
            
        Returns:
            边界采样点坐标 [num_points, 2]
        """
        theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        return np.stack((x, y), axis=-1)

    @staticmethod
    def generate_quadrature_grid(radius: float, num_points_per_axis: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        为圆形区域生成一个均匀的配点网格和对应的积分权重 (Numerical Quadrature Grid)。

        该方法使用简单的黎曼和思想，通过在正方形网格中筛选点来为圆形区域创建配点。

        Args:
            radius: 圆的半径.
            num_points_per_axis: 覆盖圆的正方形区域中，每个坐标轴上的点数。
                                 总的候选点数为 num_points_per_axis * num_points_per_axis.
        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - 坐标点数组 (N, 2), N为最终落在圆内的点数.
                - 权重数组 (N,), 每个点对应的面积权重.
        """
        # 1. 在覆盖圆的正方形区域 [-radius, radius] x [-radius, radius] 内生成均匀网格
        linspace = np.linspace(-radius, radius, num_points_per_axis)
        x_grid, y_grid = np.meshgrid(linspace, linspace)
        grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T
        # 2. 筛选出在圆内部和边界上的点 (距离圆心 <= 半径)
        distances_sq = grid_points[:, 0] ** 2 + grid_points[:, 1] ** 2
        inner_points_mask = distances_sq <= radius ** 2
        inner_points = grid_points[inner_points_mask]
        # 3. 计算每个点的权重 (代表的小面积)
        # 步长 delta_x = delta_y
        delta = (2 * radius) / (num_points_per_axis - 1)
        cell_area = delta ** 2

        # 所有点的权重都等于这个小面积
        num_inner_points = inner_points.shape[0]
        weights = np.full(num_inner_points, cell_area)

        # 验证一下总面积是否接近圆面积
        total_area_approx = np.sum(weights)
        true_area = np.pi * radius ** 2
        print(f"Generated {num_inner_points} quadrature points inside the disk.")
        print(f"Approximated Area: {total_area_approx:.4f}, True Area: {true_area:.4f}")

        return inner_points, weights

    @staticmethod
    def compute_error(output: torch.Tensor, target: torch.Tensor) -> float:
        """
        计算相对 L2 误差
        
        Args:
            output: 模型输出
            target: 目标值
            
        Returns:
            相对L2误差
        """
        import math
        error = output - target
        error = math.sqrt(torch.mean(error ** 2))
        ref = math.sqrt(torch.mean(target ** 2))
        return error / ref