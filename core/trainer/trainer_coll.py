"""
deep ritz核心训练逻辑 - 配点法训练器
"""
import torch
import os
import math
from typing import List, Tuple
from .base_trainer import BaseTrainer
from ..data_utils.sampler import Utils
from ..loss.losses_coll import compute_energy_loss_quadrature, compute_boundary_loss_quadrature, compute_total_loss_quadrature


class CollocationTrainer(BaseTrainer):
    """配点法训练器类"""

    def __init__(self, model, device: str, params: dict):
        """
        初始化配点法训练器

        Args:
            model: 神经网络模型（包含pde属性）
            device: 计算设备
            params: 配置参数
        """
        super().__init__(model, device, params)
        # 准备训练数据
        self._prepare_training_data()

    def _prepare_training_data(self) -> None:
        """准备配点法训练数据"""
        # 预先生成固定的配点，供整个训练过程使用

        # 从参数中获取每个轴的点数
        num_points_per_axis = int(math.sqrt(self.params.get("bodyQuadBatch", 2500)))

        # 1. 生成内部配点和权重
        points_body_np, weights_body_np = Utils.generate_quadrature_grid(
            self.params["radius"], num_points_per_axis
        )
        # 转换为张量并移动到设备
        self.data_body = torch.from_numpy(points_body_np).float().to(self.device)
        self.weights_body = torch.from_numpy(weights_body_np).float().to(self.device).view(-1, 1)
        # 设置 requires_grad=True 以便计算梯度
        self.data_body.requires_grad = True

        # 2. 生成边界点（可随机采样）
        self.data_boundary = torch.from_numpy(Utils.sample_from_surface(
            self.params["radius"], self.params["bdryBatch"]
        )).float().to(self.device)


    def test(self) -> Tuple[float, float]:
        num_quad = self.params["numQuad"]
        data = torch.from_numpy(Utils.sample_from_disk(self.params["radius"], num_quad)).float().to(self.device)
        data.requires_grad_(True)

        # ← 关键修复：为 EnhancedRitzNet 添加核函数特征
        nodes = self.model.nodes  # 从模型获取
        s = self.model.s  # 从模型获取

        # 计算径向距离特征 (N,1)
        distances = torch.sqrt(torch.sum((data.unsqueeze(1) - nodes.unsqueeze(0)) ** 2, dim=2))
        kernel_features = torch.exp(-distances ** 2 / (2 * s ** 2))  # (N, M)

        # 拼接：(N, 2+21=23)
        enhanced_data = torch.cat([data, kernel_features], dim=1)

        output = self.model(enhanced_data)  # 现在维度正确！
        target = self.model.pde.exact_solution(data)  # exact_solution 只用 (x,y)

        l2_error = Utils.compute_error(output, target)

        # 梯度计算用增强数据
        grad_output = torch.autograd.grad(
            outputs=output.sum(),
            inputs=enhanced_data,
            grad_outputs=torch.ones_like(output.sum()),
            create_graph=True,
            retain_graph=True,
            allow_unused=True
        )[0]

        grad_target = torch.autograd.grad(
            outputs=target.sum(),
            inputs=data,
            grad_outputs=torch.ones_like(target.sum()),
            create_graph=True,
            retain_graph=True,
            allow_unused=True
        )[0]

        l2_grad_error = torch.sqrt(torch.mean((grad_output[:, :2] - grad_target) ** 2)) * math.pi * self.params[
            "radius"] ** 2
        h1_error = torch.sqrt(l2_error ** 2 + l2_grad_error ** 2)

        return l2_error, h1_error

    def _compute_loss(self, data_body, data_boundary, weights_body=None) -> torch.Tensor:
        """
        计算配点法损失

        Args:
            data_body: 内部配点数据
            data_boundary: 边界点数据
            weights_body: 配点权重（可选）

        Returns:
            总损失
        """
        # 使用传入的权重或默认权重
        if weights_body is None:
            weights_body = self.weights_body

        output_body = self.model(data_body)
        grad_output = torch.autograd.grad(
            output_body.sum(), data_body,
            retain_graph=True, create_graph=True
        )[0]

        source_term = self.model.pde.source_term(data_body).to(self.device)

        # 配点法能量损失
        energy_loss = compute_energy_loss_quadrature(output_body, grad_output, source_term, weights_body)

        # 边界损失
        target_boundary = self.model.pde.boundary_condition(data_boundary)
        output_boundary = self.model(data_boundary)
        boundary_loss = compute_boundary_loss_quadrature(output_boundary, target_boundary, self.params["penalty"], self.params["radius"])

        # 总损失
        return compute_total_loss_quadrature(energy_loss, boundary_loss)

    def train(self) -> Tuple[List[int], List[float], List[float]]:
        """
        训练模型 - 配点法（使用固定配点进行数值积分）

        Returns:
            (List[int], List[float], List[float]): (训练步数列表, L2误差列表, H1误差列表)
        """
        self.model.train()
        steps, l2_errors, h1_errors = [], [], []
        loss_window = []

        loss_history_file = "rkdr_coll_loss_history.txt"
        error_history_file = "rkdr_coll_error_history.txt"
        loss_path = os.path.join(self.data_dir, loss_history_file)

        with open(loss_path, "w") as f:
            f.write("Step Loss\n")
            for step in range(self.params["trainStep"]):
                # 使用固定配点计算损失
                loss = self._compute_loss(self.data_body, self.data_boundary)

                # 检查收敛
                if self._check_convergence(loss_window, loss.item(), step):
                    break

                # 记录损失
                f.write(f"{step} {loss.item()}\n")

                # 定期评估和记录进度
                if step % self.params["writeStep"] == 0:
                    self.model.eval()
                    l2_error, h1_error = self.test()
                    self.model.train()
                    steps.append(step)
                    l2_errors.append(l2_error)
                    h1_errors.append(h1_error.item())
                    self._log_progress(step, loss.item(), l2_error, h1_error)

                # 周期性更新边界点
                if step % self.params.get("sampleStep", 100) == 0:
                    self.data_boundary = torch.from_numpy(Utils.sample_from_surface(
                        self.params["radius"], self.params["bdryBatch"]
                    )).float().to(self.device)

                # 优化步骤
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

        # 保存训练历史
        self._save_history(steps, l2_errors, h1_errors, loss_history_file, error_history_file)

        return steps, l2_errors, h1_errors

