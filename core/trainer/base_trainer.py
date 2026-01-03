"""
训练器基类定义
"""

from abc import ABC, abstractmethod
import torch
from torch.optim.lr_scheduler import StepLR
import os
import math
from typing import List, Tuple, Dict, Any
from ..data_utils.sampler import Utils


class BaseTrainer(ABC):
    """训练器基类，定义通用接口和实现"""

    def __init__(self, model, device: str, params: Dict[str, Any]):
        """
        初始化训练器

        Args:
            model: 神经网络模型（包含pde属性）
            device: 计算设备 ('cuda' 或 'cpu')
            params: 配置参数字典
        """
        self.model = model
        self.device = device
        self.params = params

        # 设置优化器和调度器
        self._setup_optimizer_scheduler()

        # 设置输出目录
        self._setup_output_directory()

    def _setup_optimizer_scheduler(self) -> None:
        """设置优化器和学习率调度器"""
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.params["lr"],
            weight_decay=self.params["decay"]
        )
        self.scheduler = StepLR(
            self.optimizer,
            step_size=self.params["step_size"],
            gamma=self.params["gamma"]
        )

    def _setup_output_directory(self) -> None:
        """设置输出目录"""
        # 获取项目根目录：从core/trainer/base_trainer.py向上3级
        current_file = os.path.abspath(__file__)
        trainer_dir = os.path.dirname(current_file)  # core/trainer
        core_dir = os.path.dirname(trainer_dir)  # core
        project_root = os.path.dirname(core_dir)  # 项目根目录
        self.data_dir = os.path.join(project_root, "output")
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

        grad_output = torch.autograd.grad(
            output, data,
            grad_outputs=torch.ones_like(output),
            create_graph=True
        )[0]
        grad_target = torch.autograd.grad(
            target, data,
            grad_outputs=torch.ones_like(target),
            create_graph=True
        )[0]

        l2_grad_error = torch.sqrt(torch.mean((grad_output - grad_target) ** 2)) * math.pi * self.params["radius"] ** 2
        h1_error = torch.sqrt(l2_error ** 2 + l2_grad_error ** 2)

        return l2_error, h1_error

    def _save_history(self, steps: List[int], l2_errors: List[float],
                     h1_errors: List[float], loss_history_file: str,
                     error_history_file: str) -> None:
        """
        保存训练历史记录

        Args:
            steps: 训练步数列表
            l2_errors: L2误差列表
            h1_errors: H1误差列表
            loss_history_file: 损失历史文件名
            error_history_file: 误差历史文件名
        """
        # 保存误差历史
        error_path = os.path.join(self.data_dir, error_history_file)
        with open(error_path, "w") as f:
            f.write("Step L2_Error H1_Error\n")
            for step, l2_err, h1_err in zip(steps, l2_errors, h1_errors):
                f.write(f"{step} {l2_err} {h1_err}\n")

    def _check_convergence(self, loss_window: List[float], loss_value: float, step: int) -> bool:
        """
        检查训练是否收敛

        Args:
            loss_window: 损失窗口
            loss_value: 当前损失值
            step: 当前步数

        Returns:
            bool: 是否收敛
        """
        window_size = self.params.get("window_size", 100)
        tolerance = self.params.get("tolerance", 1e-5)

        loss_window.append(loss_value)
        if len(loss_window) > window_size and step > window_size:
            loss_diff = max(loss_window) - min(loss_window)
            if loss_diff < float(tolerance):
                print(f"Training stopped at step {step}: Loss converged (difference {loss_diff:.8f} < {tolerance})")
                return True
            loss_window.pop(0)

        return False

    def _log_progress(self, step: int, loss: float, l2_error: float, h1_error: float) -> None:
        """
        记录训练进度

        Args:
            step: 当前步数
            loss: 当前损失
            l2_error: L2误差
            h1_error: H1误差
        """
        print(f"Step {step}: Loss = {loss:.6f}, L2 Error = {l2_error:.6f}, H1 Error = {h1_error:.6f}")

    @abstractmethod
    def _compute_loss(self, *args, **kwargs) -> torch.Tensor:
        """
        计算损失（子类必须实现）

        Returns:
            计算得到的损失张量
        """
        pass

    @abstractmethod
    def _prepare_training_data(self) -> None:
        """
        准备训练数据（子类必须实现）
        """
        pass

    @abstractmethod
    def train(self) -> Tuple[List[int], List[float], List[float]]:
        """
        训练模型（子类必须实现）

        Returns:
            (List[int], List[float], List[float]): (训练步数列表, L2误差列表, H1误差列表)
        """
        pass