"""
================================================================================
Deep Ritz方法求解Poisson方程 - 完整单文件实现
================================================================================

算法原理：
-----------
Deep Ritz方法将PDE问题转化为变分问题，使用神经网络逼近解函数。

问题设定：
    -Δu = f(x)  in Ω = {x ∈ R²: |x| < R}  (Poisson方程)
    u = g(x)    on ∂Ω                      (Dirichlet边界条件)

变分形式：
    最小化能量泛函 E(u) = ∫_Ω [1/2|∇u|² - fu] dx + λ∫_∂Ω |u-g|² ds

    其中：
    - 第一项：梯度能量项 (1/2|∇u|²)
    - 第二项：源项积分 (-fu)
    - 第三项：边界惩罚项 (λ|u-g|²)

数值实现：
    1. 使用神经网络 u_θ(x) 逼近解函数
    2. 蒙特卡洛积分估计能量泛函
    3. 自动微分计算梯度 ∇u_θ
    4. Adam优化器最小化损失函数

网络结构：
    采用全连接残差网络 (ResNet)
    - 输入层：2维坐标 (x, y)
    - 隐藏层：多层全连接 + 残差连接
    - 输出层：1维标量 u(x,y)

训练流程：
    1. 随机采样内部点和边界点
    2. 前向传播计算网络输出
    3. 自动微分计算梯度
    4. 计算能量损失和边界损失
    5. 反向传播更新参数
    6. 周期性评估L2和H1误差

作者：DeepRitz项目组
日期：2026-01-02
================================================================================
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import math
import os
import time

# ============================================================================
# 第一部分：数据采样工具类
# ============================================================================

class DataSampler:
    """数据采样工具类 - 用于生成训练和测试数据"""

    @staticmethod
    def sample_from_disk(radius: float, num_samples: int) -> np.ndarray:
        """
        从圆盘内部均匀采样点

        数学原理：
            使用极坐标变换：x = r*cos(θ), y = r*sin(θ)
            为保证均匀分布，r ~ sqrt(U[0,1])，θ ~ U[0,2π]

        Args:
            radius: 圆盘半径 R
            num_samples: 采样点数 N

        Returns:
            shape=(N, 2) 的数组，每行是一个点 (x, y)
        """
        # 生成随机半径和角度
        # r = R * sqrt(u)，其中u~U[0,1]，保证面积均匀分布
        r = radius * np.sqrt(np.random.rand(num_samples))

        # θ ~ U[0, 2π]
        theta = 2 * np.pi * np.random.rand(num_samples)

        # 极坐标转直角坐标
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        # 返回 (x, y) 坐标对
        return np.column_stack([x, y])

    @staticmethod
    def sample_from_surface(radius: float, num_samples: int) -> np.ndarray:
        """
        从圆周边界均匀采样点

        数学原理：
            圆周参数方程：x = R*cos(θ), y = R*sin(θ)
            θ ~ U[0, 2π] 保证弧长均匀分布

        Args:
            radius: 圆盘半径 R
            num_samples: 采样点数 N

        Returns:
            shape=(N, 2) 的数组，每行是边界上的点 (x, y)
        """
        # θ ~ U[0, 2π]
        theta = 2 * np.pi * np.random.rand(num_samples)

        # 圆周上的点：r = R (固定)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)

        return np.column_stack([x, y])

    @staticmethod
    def compute_l2_error(output: torch.Tensor, target: torch.Tensor,
                        radius: float, relative: bool = True) -> float:
        """
        计算L2误差（绝对误差或相对误差）

        绝对L2误差：
            ||u - u_exact||_L² = sqrt(∫_Ω |u - u_exact|² dx)

        相对L2误差（推荐用于科研汇报）：
            Relative L2 Error = ||u - u_exact||_L² / ||u_exact||_L²

        蒙特卡洛估计：
            ||u||_L² ≈ sqrt(Area(Ω) * mean(|u|²))
                    = sqrt(πR² * mean(|u|²))

        Args:
            output: 网络输出 u(x)
            target: 精确解 u_exact(x)
            radius: 域半径 R
            relative: 是否计算相对误差（默认True）

        Returns:
            L2误差值（相对误差或绝对误差）
        """
        area = math.pi * radius ** 2

        # 计算分子：||u - u_exact||_L²
        squared_error = (output - target) ** 2
        l2_error_numerator = torch.sqrt(torch.mean(squared_error) * area)

        if relative:
            # 计算分母：||u_exact||_L²
            squared_target = target ** 2
            l2_norm_target = torch.sqrt(torch.mean(squared_target) * area)

            # 相对误差 = ||u - u_exact||_L² / ||u_exact||_L²
            return (l2_error_numerator / l2_norm_target).item()
        else:
            # 绝对误差
            return l2_error_numerator.item()


# ============================================================================
# 第二部分：PDE问题定义
# ============================================================================

class Poisson2D:
    """
    二维Poisson方程问题定义

    方程：-Δu = f(x,y) in Ω
    边界：u = g(x,y) on ∂Ω

    测试问题：
        精确解：u(x,y) = sin(πx)sin(πy)
        源项：f(x,y) = 2π²sin(πx)sin(πy)  (由-Δu计算得到)
        边界：g(x,y) = 0  (在圆周上精确解≈0)
    """

    def __init__(self, radius: float = 1.0):
        """
        初始化Poisson问题

        Args:
            radius: 圆形区域半径
        """
        self.radius = radius
        self.name = "Poisson2D"

    def source_term(self, x: torch.Tensor) -> torch.Tensor:
        """
        源项 f(x,y)

        对于精确解 u = sin(πx)sin(πy)：
            ∂²u/∂x² = -π²sin(πx)sin(πy)
            ∂²u/∂y² = -π²sin(πx)sin(πy)
            Δu = ∂²u/∂x² + ∂²u/∂y² = -2π²sin(πx)sin(πy)

        因此 f = -Δu = 2π²sin(πx)sin(πy)

        Args:
            x: 输入坐标 shape=(N, 2)

        Returns:
            源项值 shape=(N, 1)
        """
        return 2 * (math.pi ** 2) * torch.sin(math.pi * x[:, 0:1]) * \
               torch.sin(math.pi * x[:, 1:2])

    def exact_solution(self, x: torch.Tensor) -> torch.Tensor:
        """
        精确解 u_exact(x,y) = sin(πx)sin(πy)

        Args:
            x: 输入坐标 shape=(N, 2)

        Returns:
            精确解值 shape=(N, 1)
        """
        return torch.sin(math.pi * x[:, 0:1]) * torch.sin(math.pi * x[:, 1:2])

    def boundary_condition(self, x: torch.Tensor) -> torch.Tensor:
        """
        边界条件 g(x,y)

        对于圆形区域，边界上精确解的值

        Args:
            x: 边界点坐标 shape=(N, 2)

        Returns:
            边界值 shape=(N, 1)
        """
        return self.exact_solution(x)


print("=" * 80)
print("第一部分加载完成：数据采样和PDE问题定义")
print("=" * 80)


# ============================================================================
# 第三部分：全连接残差神经网络
# ============================================================================

class ResidualBlock(nn.Module):
    """
    残差块 (Residual Block)

    结构：x → Linear → Tanh → Linear → (+) → output
                                        ↑
                                        x (skip connection)

    数学表达：
        output = x + F(x)
        其中 F(x) = W₂·tanh(W₁·x + b₁) + b₂

    残差连接的优点：
        1. 缓解梯度消失问题
        2. 允许训练更深的网络
        3. 提供恒等映射的快捷路径
    """

    def __init__(self, width: int):
        """
        初始化残差块

        Args:
            width: 隐藏层宽度
        """
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(width, width)
        self.linear2 = nn.Linear(width, width)
        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入 shape=(batch, width)

        Returns:
            输出 shape=(batch, width)
        """
        # 残差路径：F(x) = W₂·tanh(W₁·x + b₁) + b₂
        residual = self.linear2(self.activation(self.linear1(x)))

        # 跳跃连接：output = x + F(x)
        return x + residual


class RitzNet(nn.Module):
    """
    Ritz神经网络 - 全连接残差网络

    网络结构：
        Input(2) → Linear(width) → Tanh →
        [ResBlock × (depth-1)] →
        Linear(1) → Output

    参数说明：
        - input_dim: 输入维度 (2D坐标)
        - output_dim: 输出维度 (标量解)
        - width: 隐藏层宽度
        - depth: 网络深度 (包含残差块数量)
    """

    def __init__(self, input_dim: int = 2, output_dim: int = 1,
                 width: int = 100, depth: int = 3):
        """
        初始化Ritz网络

        Args:
            input_dim: 输入维度 (默认2，表示2D坐标)
            output_dim: 输出维度 (默认1，表示标量解)
            width: 隐藏层宽度
            depth: 网络深度 (残差块数量)
        """
        super(RitzNet, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.width = width
        self.depth = depth

        # 输入层：2D → width
        self.input_layer = nn.Linear(input_dim, width)

        # 残差块序列
        self.res_blocks = nn.ModuleList([
            ResidualBlock(width) for _ in range(depth - 1)
        ])

        # 输出层：width → 1
        self.output_layer = nn.Linear(width, output_dim)

        # 激活函数
        self.activation = nn.Tanh()

        print(f"\n网络结构：")
        print(f"  输入维度: {input_dim}")
        print(f"  隐藏层宽度: {width}")
        print(f"  残差块数量: {depth - 1}")
        print(f"  输出维度: {output_dim}")
        print(f"  总参数量: {self.count_parameters():,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        计算流程：
            1. x → Linear → Tanh (输入层)
            2. 通过多个残差块
            3. Linear → output (输出层)

        Args:
            x: 输入坐标 shape=(batch, 2)

        Returns:
            网络输出 shape=(batch, 1)
        """
        # 输入层
        x = self.activation(self.input_layer(x))

        # 残差块序列
        for res_block in self.res_blocks:
            x = res_block(x)

        # 输出层
        output = self.output_layer(x)

        return output

    def count_parameters(self) -> int:
        """计算网络总参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


print("=" * 80)
print("第二部分加载完成：全连接残差神经网络")
print("=" * 80)


# ============================================================================
# 第四部分：损失函数计算（核心算法）
# ============================================================================

class LossComputer:
    """
    损失函数计算器 - Deep Ritz方法的核心

    能量泛函：
        E(u) = ∫_Ω [1/2|∇u|² - fu] dx + λ∫_∂Ω |u-g|² ds

    蒙特卡洛估计：
        ∫_Ω f(x) dx ≈ |Ω| · (1/N) Σᵢ f(xᵢ)
        其中 xᵢ ~ Uniform(Ω)，|Ω| = πR²

        ∫_∂Ω f(x) ds ≈ |∂Ω| · (1/M) Σⱼ f(xⱼ)
        其中 xⱼ ~ Uniform(∂Ω)，|∂Ω| = 2πR
    """

    def __init__(self, pde: Poisson2D, device: str):
        """
        初始化损失计算器

        Args:
            pde: PDE问题定义
            device: 计算设备
        """
        self.pde = pde
        self.device = device
        self.radius = pde.radius

    def compute_energy_loss(self, model: nn.Module,
                           data_body: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        计算能量损失（内部积分项）

        数学公式：
            E_interior = ∫_Ω [1/2|∇u|² - fu] dx

        蒙特卡洛估计：
            ≈ πR² · mean([1/2|∇u(xᵢ)|² - f(xᵢ)u(xᵢ)])

        详细步骤：
            1. 前向传播：u = model(x)
            2. 自动微分：∇u = ∂u/∂x
            3. 能量项：1/2|∇u|² = 1/2(∂u/∂x)² + 1/2(∂u/∂y)²
            4. 源项：-fu
            5. 蒙特卡洛积分

        Args:
            model: 神经网络模型
            data_body: 内部采样点 shape=(N, 2)

        Returns:
            (能量损失, 中间结果字典)
        """
        # 步骤1：前向传播计算 u(x)
        output = model(data_body)  # shape=(N, 1)

        # 步骤2：自动微分计算梯度 ∇u = (∂u/∂x, ∂u/∂y)
        grad_output = torch.autograd.grad(
            outputs=output,
            inputs=data_body,
            grad_outputs=torch.ones_like(output),
            retain_graph=True,
            create_graph=True,
            only_inputs=True
        )[0]  # shape=(N, 2)

        # 步骤3：计算能量项 1/2|∇u|²
        # |∇u|² = (∂u/∂x)² + (∂u/∂y)²
        grad_squared = torch.sum(grad_output ** 2, dim=1, keepdim=True)  # shape=(N, 1)
        energy_term = 0.5 * grad_squared  # shape=(N, 1)

        # 步骤4：计算源项 -fu
        source_term = self.pde.source_term(data_body).to(self.device)  # shape=(N, 1)
        source_integral = -source_term * output  # shape=(N, 1)

        # 步骤5：蒙特卡洛积分
        # ∫_Ω f dx ≈ Area(Ω) · mean(f)
        area = math.pi * self.radius ** 2
        energy_loss = torch.mean(energy_term + source_integral) * area

        # 保存中间结果用于调试
        debug_info = {
            'output_mean': output.mean().item(),
            'output_std': output.std().item(),
            'grad_norm_mean': torch.sqrt(grad_squared).mean().item(),
            'energy_term_mean': energy_term.mean().item(),
            'source_term_mean': source_integral.mean().item(),
        }

        return energy_loss, debug_info

    def compute_boundary_loss(self, model: nn.Module,
                             data_boundary: torch.Tensor,
                             penalty: float) -> Tuple[torch.Tensor, dict]:
        """
        计算边界损失（边界惩罚项）

        数学公式：
            E_boundary = λ ∫_∂Ω |u - g|² ds

        蒙特卡洛估计：
            ≈ λ · 2πR · mean(|u(xⱼ) - g(xⱼ)|²)

        Args:
            model: 神经网络模型
            data_boundary: 边界采样点 shape=(M, 2)
            penalty: 惩罚系数 λ

        Returns:
            (边界损失, 中间结果字典)
        """
        # 步骤1：计算边界上的网络输出 u(x)
        output_boundary = model(data_boundary)  # shape=(M, 1)

        # 步骤2：计算边界条件 g(x)
        target_boundary = self.pde.boundary_condition(data_boundary)  # shape=(M, 1)

        # 步骤3：计算边界误差 |u - g|²
        boundary_error = (output_boundary - target_boundary) ** 2  # shape=(M, 1)

        # 步骤4：蒙特卡洛积分
        # ∫_∂Ω f ds ≈ Length(∂Ω) · mean(f)
        boundary_length = 2 * math.pi * self.radius
        boundary_loss = penalty * torch.mean(boundary_error) * boundary_length

        # 保存中间结果
        debug_info = {
            'boundary_output_mean': output_boundary.mean().item(),
            'boundary_target_mean': target_boundary.mean().item(),
            'boundary_error_mean': torch.sqrt(boundary_error).mean().item(),
        }

        return boundary_loss, debug_info

    def compute_total_loss(self, model: nn.Module,
                          data_body: torch.Tensor,
                          data_boundary: torch.Tensor,
                          penalty: float,
                          verbose: bool = False) -> Tuple[torch.Tensor, dict]:
        """
        计算总损失

        总损失 = 能量损失 + 边界损失

        Args:
            model: 神经网络模型
            data_body: 内部采样点
            data_boundary: 边界采样点
            penalty: 边界惩罚系数
            verbose: 是否打印详细信息

        Returns:
            (总损失, 所有中间结果)
        """
        # 计算能量损失
        energy_loss, energy_info = self.compute_energy_loss(model, data_body)

        # 计算边界损失
        boundary_loss, boundary_info = self.compute_boundary_loss(
            model, data_boundary, penalty
        )

        # 总损失
        total_loss = energy_loss + boundary_loss

        # 合并调试信息
        debug_info = {
            'energy_loss': energy_loss.item(),
            'boundary_loss': boundary_loss.item(),
            'total_loss': total_loss.item(),
            **energy_info,
            **boundary_info
        }

        if verbose:
            print(f"\n损失详情：")
            print(f"  能量损失: {energy_loss.item():.6f}")
            print(f"  边界损失: {boundary_loss.item():.6f}")
            print(f"  总损失: {total_loss.item():.6f}")
            print(f"  梯度范数: {energy_info['grad_norm_mean']:.6f}")
            print(f"  边界误差: {boundary_info['boundary_error_mean']:.6f}")

        return total_loss, debug_info


print("=" * 80)
print("第三部分加载完成：损失函数计算")
print("=" * 80)


# ============================================================================
# 第五部分：训练器
# ============================================================================

class DeepRitzTrainer:
    """Deep Ritz训练器 - 管理完整的训练流程"""

    def __init__(self, model: nn.Module, pde: Poisson2D, device: str, config: dict):
        """
        初始化训练器

        Args:
            model: 神经网络模型
            pde: PDE问题定义
            device: 计算设备
            config: 配置参数
        """
        self.model = model.to(device)
        self.pde = pde
        self.device = device
        self.config = config

        # 创建损失计算器
        self.loss_computer = LossComputer(pde, device)

        # 设置优化器
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config['decay']
        )

        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config['step_size'],
            gamma=config['gamma']
        )

        # 训练历史
        self.loss_history = []
        self.l2_error_history = []
        self.h1_error_history = []

    def train(self) -> Tuple[List[float], List[float], List[float]]:
        """
        训练模型

        训练流程：
            1. 采样训练数据
            2. 前向传播计算损失
            3. 反向传播更新参数
            4. 周期性评估误差
            5. 周期性重新采样

        Returns:
            (损失历史, L2误差历史, H1误差历史)
        """
        print("\n" + "=" * 80)
        print("开始训练")
        print("=" * 80)

        self.model.train()
        start_time = time.time()

        for step in range(self.config['train_steps']):
            # 步骤1：采样训练数据
            if step % self.config['sample_step'] == 0:
                data_body, data_boundary = self._sample_training_data()

            # 步骤2：计算损失
            loss, debug_info = self.loss_computer.compute_total_loss(
                self.model, data_body, data_boundary,
                penalty=self.config['penalty'],
                verbose=(step % self.config['print_step'] == 0)
            )

            # 步骤3：反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # 记录损失
            self.loss_history.append(loss.item())

            # 步骤4：周期性评估
            if step % self.config['eval_step'] == 0:
                l2_error, h1_error = self._evaluate()
                self.l2_error_history.append(l2_error)
                self.h1_error_history.append(h1_error)

                elapsed = time.time() - start_time
                print(f"\nStep {step}/{self.config['train_steps']} "
                      f"(耗时: {elapsed:.1f}s)")
                print(f"  损失: {loss.item():.6f}")
                print(f"  相对L2误差: {l2_error:.6f}")
                print(f"  相对H1误差: {h1_error:.6f}")
                print(f"  学习率: {self.scheduler.get_last_lr()[0]:.6f}")

        total_time = time.time() - start_time
        print(f"\n训练完成！总耗时: {total_time:.1f}秒")

        return self.loss_history, self.l2_error_history, self.h1_error_history

    def _sample_training_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        采样训练数据

        Returns:
            (内部点, 边界点)
        """
        # 采样内部点
        data_body = torch.from_numpy(
            DataSampler.sample_from_disk(
                self.config['radius'],
                self.config['body_batch']
            )
        ).float().to(self.device)
        data_body.requires_grad = True

        # 采样边界点
        data_boundary = torch.from_numpy(
            DataSampler.sample_from_surface(
                self.config['radius'],
                self.config['boundary_batch']
            )
        ).float().to(self.device)

        return data_body, data_boundary

    def _evaluate(self) -> Tuple[float, float]:
        """
        评估模型性能 - 修复版
        """
        self.model.eval()

        # 注意：这里不能使用 with torch.no_grad():
        # 因为计算H1误差需要对输入求导，必须构建计算图

        # 采样测试点
        test_data = torch.from_numpy(
            DataSampler.sample_from_disk(
                self.config['radius'],
                self.config['num_quad']
            )
        ).float().to(self.device)

        # 必须开启梯度追踪
        test_data.requires_grad = True

        # 计算网络输出和精确解
        output = self.model(test_data)
        target = self.pde.exact_solution(test_data)

        # 相对L2误差（默认计算相对误差）
        # 注意：这里计算L2误差不需要梯度，但为了代码简洁，我们在图中一起计算也没关系
        l2_error = DataSampler.compute_l2_error(
            output, target, self.config['radius'], relative=True
        )

        # --- H1 误差计算核心 (导致报错的部分) ---

        # 计算网络输出的梯度 ∇u
        grad_output = torch.autograd.grad(
            output, test_data,
            grad_outputs=torch.ones_like(output),
            create_graph=True
        )[0]

        # 计算精确解的梯度 ∇u_exact
        grad_target = torch.autograd.grad(
            target, test_data,
            grad_outputs=torch.ones_like(target),
            create_graph=True
        )[0]

        # 计算梯度的L2范数 (使用detach()切断后续不需要的梯度，节省内存)
        area = math.pi * self.config['radius'] ** 2

        # ||∇u - ∇u_exact||_L²²
        grad_error_squared = torch.mean((grad_output - grad_target) ** 2) * area

        # ||∇u_exact||_L²²
        grad_target_squared = torch.mean(grad_target ** 2) * area

        # ||u - u_exact||_L²² (为了H1计算需要重新算绝对值)
        l2_error_abs_squared = torch.mean((output - target) ** 2) * area

        # ||u_exact||_L²²
        l2_target_squared = torch.mean(target ** 2) * area

        # H1范数的平方：||u||_H¹² = ||u||_L²² + ||∇u||_L²²
        # 使用 .item() 获取数值，避免显存累积
        h1_error_numerator = math.sqrt(l2_error_abs_squared.item() + grad_error_squared.item())
        h1_norm_target = math.sqrt(l2_target_squared.item() + grad_target_squared.item())

        # 相对H1误差
        h1_error = h1_error_numerator / h1_norm_target

        self.model.train()
        return l2_error, h1_error


print("=" * 80)
print("第四部分加载完成：训练器")
print("=" * 80)


# ============================================================================
# 第六部分：可视化工具
# ============================================================================

class Visualizer:
    """可视化工具类"""

    @staticmethod
    def plot_training_history(loss_history: List[float],
                              l2_errors: List[float],
                              h1_errors: List[float],
                              eval_step: int,
                              save_path: str = None):
        """
        绘制训练历史

        Args:
            loss_history: 损失历史
            l2_errors: L2误差历史
            h1_errors: H1误差历史
            eval_step: 评估步数间隔
            save_path: 保存路径
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # 损失曲线
        axes[0].plot(loss_history, linewidth=1.5)
        axes[0].set_xlabel('训练步数', fontsize=12)
        axes[0].set_ylabel('损失值', fontsize=12)
        axes[0].set_title('训练损失曲线', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_yscale('log')

        # L2误差曲线
        eval_steps = [i * eval_step for i in range(len(l2_errors))]
        axes[1].plot(eval_steps, l2_errors, 'o-', linewidth=2, markersize=4)
        axes[1].set_xlabel('训练步数', fontsize=12)
        axes[1].set_ylabel('相对L2误差', fontsize=12)
        axes[1].set_title('相对L2误差曲线', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_yscale('log')

        # H1误差曲线
        axes[2].plot(eval_steps, h1_errors, 's-', linewidth=2, markersize=4, color='green')
        axes[2].set_xlabel('训练步数', fontsize=12)
        axes[2].set_ylabel('相对H1误差', fontsize=12)
        axes[2].set_title('相对H1误差曲线', fontsize=14, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_yscale('log')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"训练历史图已保存到: {save_path}")

        plt.show()

    @staticmethod
    def plot_solution_comparison(model: nn.Module, pde: Poisson2D,
                                 device: str, radius: float,
                                 n_samples: int = 100,
                                 save_path: str = None):
        """
        绘制解的对比图

        Args:
            model: 训练好的模型
            pde: PDE问题
            device: 计算设备
            radius: 域半径
            n_samples: 采样点数
            save_path: 保存路径
        """
        model.eval()

        # 生成网格点
        x = np.linspace(-radius, radius, n_samples)
        y = np.linspace(-radius, radius, n_samples)
        X, Y = np.meshgrid(x, y)

        # 只保留圆盘内的点
        mask = X**2 + Y**2 <= radius**2
        points = np.column_stack([X[mask], Y[mask]])

        # 计算网络输出和精确解
        with torch.no_grad():
            points_tensor = torch.from_numpy(points).float().to(device)
            output = model(points_tensor).cpu().numpy()
            target = pde.exact_solution(points_tensor).cpu().numpy()

        # 重构为网格形状
        output_grid = np.full(X.shape, np.nan)
        target_grid = np.full(X.shape, np.nan)
        error_grid = np.full(X.shape, np.nan)

        output_grid[mask] = output.flatten()
        target_grid[mask] = target.flatten()
        error_grid[mask] = np.abs(output.flatten() - target.flatten())

        # 绘图
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 网络解
        im0 = axes[0, 0].contourf(X, Y, output_grid, levels=20, cmap='viridis')
        axes[0, 0].set_title('网络解 u_θ(x,y)', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('x')
        axes[0, 0].set_ylabel('y')
        axes[0, 0].set_aspect('equal')
        plt.colorbar(im0, ax=axes[0, 0])

        # 精确解
        im1 = axes[0, 1].contourf(X, Y, target_grid, levels=20, cmap='viridis')
        axes[0, 1].set_title('精确解 u_exact(x,y)', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('x')
        axes[0, 1].set_ylabel('y')
        axes[0, 1].set_aspect('equal')
        plt.colorbar(im1, ax=axes[0, 1])

        # 误差分布
        im2 = axes[1, 0].contourf(X, Y, error_grid, levels=20, cmap='hot')
        axes[1, 0].set_title('绝对误差 |u_θ - u_exact|', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('x')
        axes[1, 0].set_ylabel('y')
        axes[1, 0].set_aspect('equal')
        plt.colorbar(im2, ax=axes[1, 0])

        # 误差统计
        axes[1, 1].hist(error_grid[mask], bins=50, edgecolor='black', alpha=0.7)
        axes[1, 1].set_xlabel('误差值', fontsize=12)
        axes[1, 1].set_ylabel('频数', fontsize=12)
        axes[1, 1].set_title('误差分布直方图', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)

        # 添加统计信息
        max_error = np.nanmax(error_grid)
        mean_error = np.nanmean(error_grid)
        axes[1, 1].text(0.95, 0.95, f'最大误差: {max_error:.6f}\n平均误差: {mean_error:.6f}',
                       transform=axes[1, 1].transAxes,
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                       fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"解对比图已保存到: {save_path}")

        plt.show()


print("=" * 80)
print("第五部分加载完成：可视化工具")
print("=" * 80)


# ============================================================================
# 第七部分：主函数
# ============================================================================

def main():
    """主函数 - 完整的训练和评估流程"""

    print("\n" + "=" * 80)
    print("Deep Ritz方法求解Poisson方程")
    print("=" * 80)

    # 配置参数
    config = {
        # 问题参数
        'radius': 1.0,

        # 网络参数
        'width': 100,
        'depth': 3,

        # 训练参数
        'train_steps': 5000,
        'lr': 0.001,
        'decay': 0.0001,
        'step_size': 500,
        'gamma': 0.5,

        # 采样参数
        'body_batch': 4096,
        'boundary_batch': 4096,
        'num_quad': 40000,

        # 边界惩罚
        'penalty': 1000,

        # 评估和打印
        'eval_step': 100,
        'print_step': 500,
        'sample_step': 200,

        # 设备
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    print(f"\n配置参数：")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # 设置设备
    device = config['device']
    print(f"\n使用设备: {device}")

    # 创建PDE问题
    pde = Poisson2D(radius=config['radius'])
    print(f"\nPDE问题: {pde.name}")
    print(f"  域: 圆盘 Ω = {{x ∈ R²: |x| < {config['radius']}}}")
    print(f"  方程: -Δu = f(x)")
    print(f"  边界: u = g(x) on ∂Ω")

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

    # 打印最终结果
    print("\n" + "=" * 80)
    print("训练结果总结")
    print("=" * 80)
    print(f"最终损失: {loss_history[-1]:.6f}")
    print(f"最终相对L2误差: {l2_errors[-1]:.6f} ({l2_errors[-1]*100:.2f}%)")
    print(f"最终相对H1误差: {h1_errors[-1]:.6f} ({h1_errors[-1]*100:.2f}%)")

    # 可视化训练历史
    print("\n生成训练历史图...")
    Visualizer.plot_training_history(
        loss_history, l2_errors, h1_errors,
        eval_step=config['eval_step'],
        save_path='training_history.png'
    )

    # 可视化解的对比
    print("\n生成解对比图...")
    Visualizer.plot_solution_comparison(
        model, pde, device,
        radius=config['radius'],
        n_samples=100,
        save_path='solution_comparison.png'
    )

    print("\n" + "=" * 80)
    print("程序执行完毕！")
    print("=" * 80)


if __name__ == "__main__":
    main()

