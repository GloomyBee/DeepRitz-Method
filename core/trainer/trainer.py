"""
deep ritz核心训练逻辑 - DeepRitz方法（蒙特卡洛积分）
"""

import torch
import os
from typing import List, Tuple
from .base_trainer import BaseTrainer
from ..loss.losses import compute_energy_loss, compute_boundary_loss, compute_total_loss
from ..data_utils.sampler import Utils


class Trainer(BaseTrainer):
    """DeepRitz训练器类（蒙特卡洛积分方法）"""

    def __init__(self, model, device: str, params: dict):
        """
        初始化DeepRitz训练器

        Args:
            model: 神经网络模型（包含pde属性）
            device: 计算设备
            params: 配置参数
        """
        super().__init__(model, device, params)
        # 准备训练数据
        self._prepare_training_data()

    def _prepare_training_data(self) -> None:
        """准备训练数据"""
        self.data_body = torch.from_numpy(
            Utils.sample_from_disk(self.params["radius"], self.params["bodyBatch"])
        ).float().to(self.device)
        self.data_body.requires_grad = True

        self.data_boundary = torch.from_numpy(
            Utils.sample_from_surface(self.params["radius"], self.params["bdryBatch"])
        ).float().to(self.device)

    def _compute_loss(self, data_body, data_boundary) -> torch.Tensor:
        """
        计算DeepRitz损失

        Args:
            data_body: 内部点数据
            data_boundary: 边界点数据

        Returns:
            总损失
        """
        # 计算内部点输出和梯度
        output_body = self.model(data_body)
        grad_output = torch.autograd.grad(
            output_body, data_body,
            grad_outputs=torch.ones_like(output_body),
            retain_graph=True, create_graph=True, only_inputs=True
        )[0]

        # 能量损失
        source_term = self.model.pde.source_term(data_body).to(self.device)
        energy_loss = compute_energy_loss(output_body, grad_output, source_term, self.params["radius"])

        # 边界损失
        target_boundary = self.model.pde.boundary_condition(data_boundary)
        output_boundary = self.model(data_boundary)
        boundary_loss = compute_boundary_loss(output_boundary, target_boundary, self.params["penalty"], self.params["radius"])

        # 总损失
        return compute_total_loss(energy_loss, boundary_loss)

    def train(self) -> Tuple[List[int], List[float], List[float]]:
        """
        训练模型 - DeepRitz方法（蒙特卡洛积分）

        Returns:
            (训练step列表, L2误差列表, H1误差列表)
        """
        self.model.train()
        steps, l2_errors, h1_errors = [], [], []
        loss_window = []

        loss_history_file = "rkdr_mc_loss_history.txt"
        error_history_file = "rkdr_mc_error_history.txt"
        loss_path = os.path.join(self.data_dir, loss_history_file)

        with open(loss_path, "w") as f:
            f.write("Step Loss\n")
            for step in range(self.params["trainStep"]):
                # 使用当前数据计算损失
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

                # 周期性重新采样数据
                if step % self.params["sampleStep"] == 0:
                    self._prepare_training_data()

                # 优化步骤
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

        # 保存训练历史
        self._save_history(steps, l2_errors, h1_errors, loss_history_file, error_history_file)

        return steps, l2_errors, h1_errors