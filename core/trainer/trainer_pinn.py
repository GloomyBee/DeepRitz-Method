"""
trainer_pinn.py: 基于物理信息神经网络（PINN）的训练逻辑
"""
import os

import torch
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
            device: 计算设备
            params: 配置参数
        """
        super().__init__(model, device, params)

        # 创建损失计算器
        from ..loss.losses_pinn import PINNLoss
        self.loss_calculator = PINNLoss()

        # 准备训练数据
        self._prepare_training_data()

    def _prepare_training_data(self) -> None:
        """准备PINN训练数据"""
        from ..data_utils.sampler import Utils
        self.data_body = torch.from_numpy(
            Utils.sample_from_disk(self.params["radius"], self.params["bodyBatch"])
        ).float().to(self.device)
        self.data_body.requires_grad = True

        # 生成边界数据
        self.data_boundary = torch.from_numpy(
            Utils.sample_from_surface(self.params["radius"], self.params["bdryBatch"])
        ).float().to(self.device)

    def compute_energy_loss(self, model: torch.nn.Module, data_body: torch.Tensor,
                           source_func) -> torch.Tensor:
        """
        计算PDE残差损失

        Args:
            model: 神经网络模型
            data_body: 内部采样点
            source_func: 源项函数

        Returns:
            PDE残差损失
        """
        return self.loss_calculator.compute_energy_loss(model, data_body, source_func)

    def compute_boundary_loss(self, output: torch.Tensor, target: torch.Tensor,
                             **kwargs) -> torch.Tensor:
        """
        计算边界条件损失（L2范数）

        Args:
            output: 模型边界输出
            target: 目标边界值
            **kwargs: 其他参数

        Returns:
            边界损失
        """
        return self.loss_calculator.compute_boundary_loss(output, target, penalty=self.params["penalty"], radius=self.params["radius"], **kwargs)

    def compute_total_loss(self, pde_loss: torch.Tensor, boundary_loss: torch.Tensor,
                        penalty: float) -> torch.Tensor:
        """
        计算PINN总损失（带权重）

        Args:
            pde_loss: PDE残差损失
            boundary_loss: 边界损失
            penalty: 边界条件权重

        Returns:
            加权总损失
        """
        return self.loss_calculator.compute_total_loss(pde_loss, boundary_loss, penalty)

    def _compute_loss(self, data_body, data_boundary) -> torch.Tensor:
        """
        计算PINN损失（实现BaseTrainer的抽象方法）

        Args:
            data_body: 内部采样点
            data_boundary: 边界点

        Returns:
            总损失
        """
        # 计算PDE残差损失
        pde_loss = self.compute_energy_loss(self.model, data_body, self.model.pde.source_term)

        # 计算边界条件损失
        output_boundary = self.model(data_boundary)
        target_boundary = self.model.pde.boundary_condition(data_boundary)
        boundary_loss = self.compute_boundary_loss(output_boundary, target_boundary)

        # 总损失
        return self.compute_total_loss(pde_loss, boundary_loss, self.params["penalty"])

    def train(self) -> Tuple[List[int], List[float], List[float]]:
        """
        训练模型，优化PINN损失

        Returns:
            (训练步数列表, L2误差列表, H1误差列表)
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
                # 使用_compute_loss方法计算损失
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