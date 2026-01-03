import numpy as np
import torch
from trapezoid_sampler_hr import TrapezoidSampler

class HRSampler(TrapezoidSampler):
    """
    支持法向量输出的采样器 (用于HR变分原理)
    继承自原采样器，增加法向量返回
    """

    @staticmethod
    def sample_left_boundary_with_normal(n_samples: int):
        """返回 (坐标, 法向量)"""
        coords = TrapezoidSampler.sample_left_boundary(n_samples)
        # 左边界法向量 (-1, 0)
        normals = np.zeros_like(coords)
        normals[:, 0] = -1.0
        normals[:, 1] = 0.0
        return coords, normals

    @staticmethod
    def sample_top_boundary_with_normal(n_samples: int):
        """返回 (坐标, 法向量)"""
        coords = TrapezoidSampler.sample_top_boundary(n_samples)
        # 上边界法向量 (0, 1)
        normals = np.zeros_like(coords)
        normals[:, 0] = 0.0
        normals[:, 1] = 1.0
        return coords, normals