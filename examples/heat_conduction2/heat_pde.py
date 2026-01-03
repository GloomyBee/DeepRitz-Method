"""
热传导方程实现 - 梯形区域

问题描述：
    稳态热传导方程：-k∇²T = s in Ω

    几何区域：梯形 Ω = {(x,y): 0≤x≤2, 0≤y≤1, x+y≤2}
    顶点：(0,0), (2,0), (1,1), (0,1)

    边界条件：
        - 左边界 (x=0): Dirichlet BC, T = 20°C
        - 上边界 (y=1): Neumann BC, q = -100 W/m²
        - 底边和斜边: 绝热边界 q = 0

    物理参数：
        - k = 20.0 W/(m·K)  导热系数
        - s = 50.0 W/m³     体积热源
"""

import torch
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from core.pdes.base_pde import BasePDE


class HeatPDE(BasePDE):
    """二维稳态热传导方程"""

    def __init__(self, k: float = 20.0, s: float = 50.0, T_left: float = 20.0):
        """
        初始化热传导问题

        Args:
            k: 导热系数 W/(m·K)
            s: 体积热源 W/m³
            T_left: 左边界温度 °C
        """
        self.k = k
        self.s = s
        self.T_left = T_left
        self.name = "HeatConduction2D"

        # 几何参数
        self.area = 1.5  # 梯形面积 = (1+2)*1/2
        self.len_left = 1.0  # 左边界长度
        self.len_top = 1.0   # 上边界长度

    def source_term(self, data: torch.Tensor) -> torch.Tensor:
        """
        源项 f(x,y) = s/k (归一化后的热源)

        将 -k∇²T = s 改写为 -∇²T = s/k 以匹配Poisson方程形式

        Args:
            data: 输入坐标 [batch_size, 2]

        Returns:
            源项值 [batch_size, 1]
        """
        batch_size = data.shape[0]
        return (self.s / self.k) * torch.ones(
            [batch_size, 1], dtype=torch.float, device=data.device
        )

    def exact_solution(self, data: torch.Tensor) -> torch.Tensor:
        """
        解析解（此问题无解析解，返回None用于标识）

        注意：实际热传导问题通常没有解析解，需要数值求解

        Args:
            data: 输入坐标 [batch_size, 2]

        Returns:
            None（表示无解析解）
        """
        # 返回None表示此问题没有解析解
        # 训练器需要处理这种情况
        return None

    def boundary_condition(self, data: torch.Tensor) -> torch.Tensor:
        """
        Dirichlet边界条件（仅用于左边界）

        Args:
            data: 边界点坐标 [batch_size, 2]

        Returns:
            边界温度值 [batch_size, 1]
        """
        batch_size = data.shape[0]
        return self.T_left * torch.ones(
            [batch_size, 1], dtype=torch.float, device=data.device
        )

    def is_on_left_boundary(self, data: torch.Tensor, tol: float = 1e-6) -> torch.Tensor:
        """
        判断点是否在左边界上 (x ≈ 0)

        Args:
            data: 坐标点 [batch_size, 2]
            tol: 容差

        Returns:
            布尔掩码 [batch_size]
        """
        return torch.abs(data[:, 0]) < tol

    def is_on_top_boundary(self, data: torch.Tensor, tol: float = 1e-6) -> torch.Tensor:
        """
        判断点是否在上边界上 (y ≈ 1, 0 ≤ x ≤ 1)

        Args:
            data: 坐标点 [batch_size, 2]
            tol: 容差

        Returns:
            布尔掩码 [batch_size]
        """
        on_top = torch.abs(data[:, 1] - 1.0) < tol
        in_x_range = (data[:, 0] >= -tol) & (data[:, 0] <= 1.0 + tol)
        return on_top & in_x_range


if __name__ == "__main__":
    print("=" * 80)
    print("HeatPDE 接口测试")
    print("=" * 80)

    # 创建PDE实例
    pde = HeatPDE(k=20.0, s=50.0, T_left=20.0)

    # 测试点
    test_points = torch.tensor([
        [0.0, 0.5],   # 左边界
        [0.5, 1.0],   # 上边界
        [1.0, 0.5],   # 内部点
    ], dtype=torch.float32)

    # 测试源项
    source = pde.source_term(test_points)
    print(f"\n源项测试:")
    print(f"  输入形状: {test_points.shape}")
    print(f"  输出形状: {source.shape}")
    print(f"  源项值: {source.squeeze().tolist()}")
    print(f"  期望值: {[pde.s/pde.k] * 3}")

    # 测试边界条件
    bc = pde.boundary_condition(test_points)
    print(f"\n边界条件测试:")
    print(f"  输出形状: {bc.shape}")
    print(f"  边界值: {bc.squeeze().tolist()}")
    print(f"  期望值: {[pde.T_left] * 3}")

    # 测试边界判断
    left_mask = pde.is_on_left_boundary(test_points)
    top_mask = pde.is_on_top_boundary(test_points)
    print(f"\n边界判断测试:")
    print(f"  左边界掩码: {left_mask.tolist()}")
    print(f"  上边界掩码: {top_mask.tolist()}")

    # 测试解析解
    exact = pde.exact_solution(test_points)
    print(f"\n解析解测试:")
    print(f"  返回值: {exact}")
    print(f"  说明: None表示此问题无解析解")

    print("\n" + "=" * 80)
    print("所有接口测试通过！")
    print("=" * 80)
