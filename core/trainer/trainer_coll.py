"""
deep ritz核心训练逻辑
"""
import torch
from torch.optim.lr_scheduler import StepLR
import os
import math
from typing import List, Tuple
from ..data_utils.sampler import Utils
# 确保导入了正确的损失函数
from ..loss.losses import compute_energy_loss, compute_boundary_loss, compute_total_loss
from ..loss.losses_coll import compute_energy_loss_quadrature


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

        # ====================  关键修复: 初始化配点和权重 ====================
        # 这个初始化块对 train_coll 方法至关重要。
        # 我们在这里预先生成固定的配点，供 train_coll 在整个训练过程中使用。

        # 从参数中获取每个轴的点数。
        # 'bodyQuadBatch' 应该是一个能开方的数，如 2500 (50*50), 10000 (100*100)
        # 我们使用 .get() 来提供一个默认值，以防参数文件中没有定义。
        num_points_per_axis = int(math.sqrt(self.params.get("bodyQuadBatch", 2500)))

        # 1. 生成内部配点和权重
        points_body_np, weights_body_np = Utils.generate_quadrature_grid(
            self.params["radius"], num_points_per_axis
        )
        # 将它们转换为张量并移动到指定设备 (GPU/CPU)
        self.data_body = torch.from_numpy(points_body_np).float().to(self.device)
        self.weights_body = torch.from_numpy(weights_body_np).float().to(self.device).view(-1, 1) # 调整形状以便广播
        # 设置 requires_grad=True 以便计算关于输入的梯度
        self.data_body.requires_grad = True

        # 2. 生成边界点 (边界点仍可随机采样, 或者也可以固定)
        self.data_boundary = torch.from_numpy(Utils.sample_from_surface(
            self.params["radius"], self.params["bdryBatch"]
        )).float().to(self.device)
        # ======================= (初始化修复结束) =======================

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

    def train(self) -> Tuple[List[int], List[float], List[float]]:
        """
        训练模型 - 原始的蒙特卡洛随机采样版本 (保持不变)
        """
        # ... 这里的代码保持和你原来的一样 ...
        self.model.train()
        data_body = torch.from_numpy(Utils.sample_from_disk(self.params["radius"], self.params["bodyBatch"])).float().to(self.device)
        data_body.requires_grad = True
        data_boundary = torch.from_numpy(Utils.sample_from_surface(self.params["radius"], self.params["bdryBatch"])).float().to(self.device)

        steps, l2_errors, h1_errors = [], [], []
        loss_window = []
        window_size = self.params.get("window_size", 100)
        tolerance = self.params.get("tolerance", 1e-5)

        with open(os.path.join(self.data_dir, "rkdr_mc_loss_history.txt"), "w") as f:
            f.write("Step Loss\n")
            for step in range(self.params["trainStep"]):
                output_body = self.model(data_body)
                grad_output = torch.autograd.grad(output_body.sum(), data_body,
                                                  retain_graph=True, create_graph=True)[0]

                source_term = self.model.pde.source_term(data_body).to(self.device)
                energy_loss = compute_energy_loss(output_body, grad_output, source_term, self.params["radius"])

                target_boundary = self.model.pde.boundary_condition(data_boundary)
                output_boundary = self.model(data_boundary)
                boundary_loss = compute_boundary_loss(output_boundary, target_boundary, self.params["penalty"], self.params["radius"])

                loss = compute_total_loss(energy_loss, boundary_loss)

                loss_window.append(loss.item())
                if len(loss_window) > window_size and step > window_size:
                    loss_diff = max(loss_window) - min(loss_window)
                    if loss_diff < float(tolerance):
                        print(f"Training stopped at step {step}: Loss converged (difference {loss_diff:.8f} < {tolerance})")
                        break
                    loss_window.pop(0)

                if step % self.params["writeStep"] == 0:
                    l2_error, h1_error = self.test()
                    steps.append(step)
                    l2_errors.append(l2_error)
                    h1_errors.append(h1_error)
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

        with open(os.path.join(self.data_dir, "rkdr_mc_error_history.txt"), "w") as f:
            f.write("Step L2_Error H1_Error\n")
            for step_val, l2_err, h1_err in zip(steps, l2_errors, h1_errors):
                f.write(f"{step_val} {l2_err} {h1_err}\n")
        return steps, l2_errors, h1_errors


    def train_coll(self) -> Tuple[List[int], List[float], List[float]]:
        """
        训练模型 - 使用固定的配点 (Collocation/Quadrature) 进行积分
        """
        self.model.train()
        steps, l2_errors, h1_errors = [], [], []
        loss_window = []
        window_size = self.params.get("window_size", 100)
        tolerance = self.params.get("tolerance", 1e-5)

        # 使用不同的文件名保存配点法的历史记录
        loss_history_path = os.path.join(self.data_dir, "rkdr_coll_loss_history.txt")
        error_history_path = os.path.join(self.data_dir, "rkdr_coll_error_history.txt")

        with open(loss_history_path, "w") as f:
            f.write("Step Loss\n")
            for step in range(self.params["trainStep"]):
                # 使用在 __init__ 中定义的固定的 self.data_body 和 self.weights_body
                output_body = self.model(self.data_body)

                # 计算梯度。注意: .sum() 是一个好习惯，可以确保 grad_outputs 的形状正确
                grad_output = torch.autograd.grad(output_body.sum(), self.data_body,
                                                  retain_graph=True, create_graph=True)[0]

                source_term = self.model.pde.source_term(self.data_body).to(self.device)

                # 调用配点版本的能量损失函数
                energy_loss = compute_energy_loss_quadrature(output_body, grad_output, source_term, self.weights_body)

                # 边界损失处理不变 (仍然可以使用随机采样的边界点)
                target_boundary = self.model.pde.boundary_condition(self.data_boundary)
                output_boundary = self.model(self.data_boundary)
                boundary_loss = compute_boundary_loss(output_boundary, target_boundary, self.params["penalty"], self.params["radius"])

                loss = compute_total_loss(energy_loss, boundary_loss)

                # --- 关键补充: 从原始 train 方法复制收敛判断和评估逻辑 ---
                loss_window.append(loss.item())
                if len(loss_window) > window_size and step > window_size:
                    loss_diff = max(loss_window) - min(loss_window)
                    if loss_diff < float(tolerance):
                        print(f"Training stopped at step {step}: Loss converged (difference {loss_diff:.8f} < {tolerance})")
                        break
                    loss_window.pop(0)

                if step % self.params["writeStep"] == 0:
                    l2_error, h1_error = self.test() # 调用 test 方法进行评估
                    steps.append(step)
                    l2_errors.append(l2_error)
                    h1_errors.append(h1_error) # .item() 已在 test 方法内部处理
                    print(f"Step {step}: Loss = {loss.item():.6f}, L2 Error = {l2_error:.6f}, H1 Error = {h1_error:.6f}")
                    f.write(f"{step} {loss.item()}\n")

                # 如果希望边界点在训练中也更新，可以加入以下代码
                if step % self.params.get("sampleStep", 100) == 0:
                    self.data_boundary = torch.from_numpy(Utils.sample_from_surface(
                        self.params["radius"], self.params["bdryBatch"]
                    )).float().to(self.device)
                # --- (逻辑补充结束) ---

                # 优化器步骤
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

        with open(error_history_path, "w") as f:
            f.write("Step L2_Error H1_Error\n")
            for step_val, l2_err, h1_err in zip(steps, l2_errors, h1_errors):
                f.write(f"{step_val} {l2_err} {h1_err}\n")

        return steps, l2_errors, h1_errors
