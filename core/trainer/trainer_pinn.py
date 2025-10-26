"""
trainer_pinn.py: 基于物理信息神经网络（PINN）的训练逻辑
"""

import torch
import os
from typing import List, Tuple
from .base_trainer import BaseTrainer
from ..loss.losses_pinn import compute_pde_loss, compute_bc_loss, compute_total_pinn_loss
from ..data_utils.sampler import Utils


class PINNTrainer(BaseTrainer):
    """PINN训练器类，用于优化基于PDE残差的神经网络"""

    def __init__(self, model, device: str, params: dict):
        """
        初始化PINN训练器

        Args:
            model: 神经网络模型（包含pde属性）
            device: 计算设备 ('cuda' 或 'cpu')
            params: 配置参数字典
        """
        super().__init__(model, device, params)
        # 准备训练数据
        self._prepare_training_data()

    def _prepare_training_data(self) -> None:
        """准备PINN训练数据"""
        self.data_body = torch.from_numpy(
            Utils.sample_from_disk(self.params["radius"], self.params["bodyBatch"])
        ).float().to(self.device)
        self.data_body.requires_grad = True

        self.data_boundary = torch.from_numpy(
            Utils.sample_from_surface(self.params["radius"], self.params["bdryBatch"])
        ).float().to(self.device)

    def _compute_loss(self, data_body, data_boundary) -> torch.Tensor:
        """
        计算PINN损失

        Args:
            data_body: 内部点数据
            data_boundary: 边界点数据

        Returns:
            总损失
        """
        # 计算PDE残差损失
        pde_loss = compute_pde_loss(self.model, data_body, self.model.pde.source_term)

        # 计算边界条件损失
        output_boundary = self.model(data_boundary)
        target_boundary = self.model.pde.boundary_condition(data_boundary)
        bc_loss = compute_bc_loss(output_boundary, target_boundary)

        # 总损失
        return compute_total_pinn_loss(pde_loss, bc_loss, self.params["penalty"])

    def train(self) -> Tuple[List[int], List[float], List[float]]:
        """
        训练模型，优化PINN损失

        Returns:
            (List[int], List[float], List[float]): (训练步数列表, L2误差列表, H1误差列表)
        """
        self.model.train()
        steps, l2_errors, h1_errors = [], [], []
        loss_window = []

        loss_history_file = "pinn_loss_history.txt"
        error_history_file = "pinn_error_history.txt"
        loss_path = os.path.join(self.data_dir, loss_history_file)

        with open(loss_path, "w") as f:
            f.write("Step Loss\n")
            for step in range(self.params["trainStep"]):
                # 计算PINN损失
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