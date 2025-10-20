"""
trainer_pinn.py: 基于物理信息神经网络（PINN）的训练逻辑
"""

import torch
from torch.optim.lr_scheduler import StepLR
import os
import math
from typing import List, Tuple
from ..data_utils.sampler import Utils
from ..loss.losses_pinn import compute_pde_loss, compute_bc_loss, compute_total_pinn_loss


class PINNTrainer:
    """PINN训练器类，用于优化基于PDE残差的神经网络"""

    def __init__(self, model, device: str, params: dict):
        """
        初始化训练器

        Args:
            model: 神经网络模型（包含pde属性）
            device: 计算设备 ('cuda' 或 'cpu')
            params: 配置参数字典，包含 lr, decay, step_size, gamma 等
        """
        self.model = model
        self.device = device
        self.params = params
        self.optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["decay"])
        self.scheduler = StepLR(self.optimizer, step_size=params["step_size"], gamma=params["gamma"])
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
        os.makedirs(self.data_dir, exist_ok=True)

    def test(self) -> Tuple[float, float]:
        """
        测试模型性能，计算L2和H1误差

        Returns:
            (float, float): (L2误差, H1误差)
        """
        num_quad = self.params["numQuad"]
        data = torch.from_numpy(Utils.sample_from_disk(self.params["radius"], num_quad)).float().to(self.device)
        data.requires_grad = True

        output = self.model(data)
        target = self.model.pde.exact_solution(data)

        l2_error = Utils.compute_error(output, target)

        grad_output = torch.autograd.grad(output, data, grad_outputs=torch.ones_like(output), create_graph=True)[0]
        grad_target = torch.autograd.grad(target, data, grad_outputs=torch.ones_like(target), create_graph=True)[0]

        l2_grad_error = torch.sqrt(torch.mean((grad_output - grad_target) ** 2)) * math.pi * self.params["radius"] ** 2
        h1_error = torch.sqrt(l2_error ** 2 + l2_grad_error ** 2)

        return l2_error, h1_error

    def train(self) -> Tuple[List[int], List[float], List[float]]:
        """
        训练模型，优化PINN损失

        Returns:
            (List[int], List[float], List[float]): (训练步数列表, L2误差列表, H1误差列表)
        """
        self.model.train()
        data_body = torch.from_numpy(
            Utils.sample_from_disk(self.params["radius"], self.params["bodyBatch"])).float().to(self.device)
        data_body.requires_grad = True
        data_boundary = torch.from_numpy(
            Utils.sample_from_surface(self.params["radius"], self.params["bdryBatch"])).float().to(self.device)

        steps, l2_errors, h1_errors = [], [], []
        loss_window = []
        window_size = self.params.get("window_size", 100)
        tolerance = self.params.get("tolerance", 1e-5)

        with open(os.path.join(self.data_dir, "pinn_loss_history.txt"), "w") as f:
            f.write("Step Loss\n")
            for step in range(self.params["trainStep"]):
                # 计算PDE残差损失
                output_body = self.model(data_body)
                pde_loss = compute_pde_loss(self.model, data_body, self.model.pde.source_term)

                # 计算边界条件损失
                output_boundary = self.model(data_boundary)
                target_boundary = self.model.pde.boundary_condition(data_boundary)
                bc_loss = compute_bc_loss(output_boundary, target_boundary)

                # 总损失
                loss = compute_total_pinn_loss(pde_loss, bc_loss, self.params["penalty"])

                # 早停检查
                loss_window.append(loss.item())
                if len(loss_window) > window_size:
                    loss_window.pop(0)
                if len(loss_window) == window_size:
                    loss_diff = max(loss_window) - min(loss_window)
                    if loss_diff < float(tolerance):
                        print(
                            f"Training stopped at step {step}: Loss converged (difference {loss_diff:.8f} < {tolerance})")
                        break

                # 日志与误差记录
                if step % self.params["writeStep"] == 0:
                    self.model.eval()
                    l2_error, h1_error = self.test()
                    self.model.train()
                    steps.append(step)
                    l2_errors.append(l2_error)
                    h1_errors.append(h1_error.item())
                    print(
                        f"Step {step}: Loss = {loss.item():.6f}, L2 Error = {l2_error:.6f}, H1 Error = {h1_error:.6f}")
                    f.write(f"{step} {loss.item()}\n")

                # 周期性重新采样
                if step % self.params["sampleStep"] == 0:
                    data_body = torch.from_numpy(
                        Utils.sample_from_disk(self.params["radius"], self.params["bodyBatch"])).float().to(self.device)
                    data_body.requires_grad = True
                    data_boundary = torch.from_numpy(
                        Utils.sample_from_surface(self.params["radius"], self.params["bdryBatch"])).float().to(
                        self.device)

                # 优化步骤
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

        # 保存误差历史
        with open(os.path.join(self.data_dir, "pinn_error_history.txt"), "w") as f:
            f.write("Step L2_Error H1_Error\n")
            for step, l2_err, h1_err in zip(steps, l2_errors, h1_errors):
                f.write(f"{step} {l2_err} {h1_err}\n")

        return steps, l2_errors, h1_errors