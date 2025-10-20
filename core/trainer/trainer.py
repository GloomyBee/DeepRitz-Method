"""
deep ritz核心训练逻辑
"""

import torch
from torch.optim.lr_scheduler import StepLR
import os
import math
from typing import List, Tuple
from ..data_utils.sampler import Utils
from ..loss.losses import compute_energy_loss, compute_boundary_loss, compute_total_loss
from ..loss.losses_pinn import compute_pde_loss, compute_bc_loss, compute_total_pinn_loss

class Trainer:
    """训练器类"""
    
    def __init__(self, model, device: str, params: dict):
        self.model = model
        self.device = device
        self.params = params
        self.optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["decay"])
        self.scheduler = StepLR(self.optimizer, step_size=params["step_size"], gamma=params["gamma"])
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
        os.makedirs(self.data_dir, exist_ok=True)

    def test(self) -> Tuple[float, float]:
        """
        测试模型性能
        
        Returns:
            (L2误差, H1误差)
        """
        num_quad = self.params["numQuad"]
        data = torch.from_numpy(Utils.sample_from_disk(self.params["radius"], num_quad)).float().to(self.device)
        data.requires_grad = True

        output = self.model(data)
        target = self.model.pde.exact_solution(data)

        l2_error = Utils.compute_error(output, target)

        grad_output = torch.autograd.grad(output, data, grad_outputs=torch.ones_like(output), create_graph=True)[0]
        grad_target = torch.autograd.grad(target, data, grad_outputs=torch.ones_like(target), create_graph=True)[0]

        l2_grad_error = torch.sqrt(torch.mean((grad_output - grad_target) ** 2)) * math.pi * self.params["radius"]**2
        h1_error = torch.sqrt(l2_error ** 2 + l2_grad_error ** 2)

        return l2_error, h1_error

    def train(self) -> Tuple[List[int], List[float], List[float]]:
        """
        训练模型
        
        Returns:
            (训练步数列表, L2误差列表, H1误差列表)
        """
        self.model.train()
        data_body = torch.from_numpy(Utils.sample_from_disk(self.params["radius"], self.params["bodyBatch"])).float().to(self.device)
        data_body.requires_grad = True
        data_boundary = torch.from_numpy(Utils.sample_from_surface(self.params["radius"], self.params["bdryBatch"])).float().to(self.device)

        steps, l2_errors, h1_errors = [], [], []
        loss_window = []
        window_size = self.params.get("window_size", 100)
        tolerance = self.params.get("tolerance", 1e-5)

        with open(os.path.join(self.data_dir, "rkdr_loss_history.txt"), "w") as f:
            f.write("Step Loss\n")
            for step in range(self.params["trainStep"]):
                output_body = self.model(data_body)
                grad_output = torch.autograd.grad(output_body, data_body, grad_outputs=torch.ones_like(output_body),
                                                  retain_graph=True, create_graph=True, only_inputs=True)[0]
                
                source_term = self.model.pde.source_term(data_body).to(self.device)
                energy_loss = compute_energy_loss(output_body, grad_output, source_term, self.params["radius"])
                
                target_boundary = self.model.pde.boundary_condition(data_boundary)
                output_boundary = self.model(data_boundary)
                boundary_loss = compute_boundary_loss(output_boundary, target_boundary, self.params["penalty"], self.params["radius"])
                
                loss = compute_total_loss(energy_loss, boundary_loss)

                loss_window.append(loss.item())
                if len(loss_window) > window_size:
                    loss_window.pop(0)
                if len(loss_window) == window_size:
                    loss_diff = max(loss_window) - min(loss_window)
                    if loss_diff < float(tolerance):
                        print(f"Training stopped at step {step}: Loss converged (difference {loss_diff:.8f} < {tolerance})")
                        break

                if step % self.params["writeStep"] == 0:
                    self.model.eval()
                    l2_error, h1_error = self.test()
                    self.model.train()
                    steps.append(step)
                    l2_errors.append(l2_error)
                    h1_errors.append(h1_error.item())
                    print(f"Step {step}: Loss = {loss.item():.6f}, L2 Error = {l2_error:.6f}, H1 Error = {h1_error:.6f}")
                    f.write(f"{step} {loss.item()}\n")

                if step % self.params["sampleStep"] == 0:
                    data_body = torch.from_numpy(Utils.sample_from_disk(self.params["radius"], self.params["bodyBatch"])).float().to(self.device)
                    data_body.requires_grad = True
                    data_boundary = torch.from_numpy(Utils.sample_from_surface(self.params["radius"], self.params["bdryBatch"])).float().to(self.device)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

        with open(os.path.join(self.data_dir, "rkdr_error_history.txt"), "w") as f:
            f.write("Step L2_Error H1_Error\n")
            for step, l2_err, h1_err in zip(steps, l2_errors, h1_errors):
                f.write(f"{step} {l2_err} {h1_err}\n")

        return steps, l2_errors, h1_errors