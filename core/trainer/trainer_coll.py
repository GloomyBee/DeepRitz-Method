"""
trainer_coll.py: 配点法训练器 - 专用于数值积分方法
"""
import torch
import os
import math
from typing import List, Tuple
from .base_trainer import BaseTrainer
from ..loss.losses_coll import compute_energy_loss_quadrature, compute_boundary_loss_quadrature, compute_total_loss_quadrature
from ..data_utils.sampler import Utils

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

        # 创建损失计算器
        from ..loss.losses_coll import QuadratureEnergyLoss
        self.loss_calculator = QuadratureEnergyLoss()

        # 准备训练数据
        self._prepare_training_data()

    def _prepare_training_data(self) -> None:
        """准备配点法训练数据"""
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
            self.params["radius"], self.params["bdryBatch"])
        ).float().to(self.device)

    def _compute_loss(self, data_body, data_boundary, weights_body=None) -> torch.Tensor:
        """
        计算配点法损失

        Args:
            data_body: 内部配点数据
            data_boundary: 边界点数据
            weights_body: 积分权重（可选）

        Returns:
            总损失
        """
        # 使用提供的权重或默认权重
        if weights_body is None:
            weights_body = self.weights_body

        # 计算模型输出和梯度
        output_body = self.model(data_body)
        grad_output = torch.autograd.grad(
            output_body, data_body,
            grad_outputs=torch.ones_like(output_body),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        return self.loss_calculator.compute_total_loss(
            energy_loss=self.loss_calculator.compute_energy_loss(
                data_body,
                grad_output=grad_output,
                source_term=self.model.pde.source_term(data_body),
                weights=weights_body
            ),
            boundary_loss=self.loss_calculator.compute_boundary_loss(
                target_boundary=self.model.pde.boundary_condition(data_boundary),
                output_boundary=self.model(data_boundary),
                penalty=self.params["penalty"],
                radius=self.params["radius"]
            )
        )

    def train(self) -> Tuple[List[int], List[float], List[float]]:
        """
        训练模型 - 配点法（使用固定配点进行数值积分）

        Returns:
            (训练步数列表, L2误差列表, H1误差列表)
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
                        self.params["radius"], self.params["bdryBatch"])
                    ).float().to(self.device)

                # 优化步骤
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

        # 保存训练历史
        self._save_history(steps, l2_errors, h1_errors, loss_history_file, error_history_file)

        return steps, l2_errors, h1_errors