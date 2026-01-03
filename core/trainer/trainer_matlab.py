"""
MATLAB Trainer - 支持MATLAB后端的训练器
"""

import torch
import os
import logging
from typing import List, Tuple, Optional
from .base_trainer import BaseTrainer
from ..loss.matlab_loss_wrapper import MatlabLossWrapper
from ..loss.losses import MonteCarloEnergyLoss
from ..interface.matlab_engine_manager import MatlabEngineManager

logger = logging.getLogger(__name__)


class MatlabTrainer(BaseTrainer):
    """支持MATLAB后端的DeepRitz训练器"""

    def __init__(self, model, device: str, params: dict, use_matlab: bool = True):
        """
        初始化MATLAB训练器

        Args:
            model: 神经网络模型
            device: 计算设备
            params: 配置参数
            use_matlab: 是否使用MATLAB后端
        """
        super().__init__(model, device, params)

        self.use_matlab = use_matlab
        self.matlab_engine = None
        self.fallback_to_python = params.get('fallback_to_python', True)

        # 设置损失计算器
        self._setup_loss_calculator()

        # 准备训练数据
        self._prepare_training_data()

    def _setup_loss_calculator(self) -> None:
        """设置损失计算器(MATLAB或Python)"""
        if self.use_matlab:
            try:
                # 启动MATLAB引擎
                logger.info("启动MATLAB引擎...")
                self.matlab_engine = MatlabEngineManager(
                    matlab_path=self.params.get('matlab_path'),
                    startup_timeout=self.params.get('matlab_startup_timeout', 30)
                )
                self.matlab_engine.start_engine()

                # 创建MATLAB损失计算器
                self.loss_calculator = MatlabLossWrapper(
                    self.matlab_engine,
                    use_quadrature=False  # 使用蒙特卡洛积分
                )
                logger.info("MATLAB后端已启用")

            except Exception as e:
                logger.error(f"MATLAB引擎启动失败: {str(e)}")
                if self.fallback_to_python:
                    logger.warning("回退到Python实现")
                    self.use_matlab = False
                    self.loss_calculator = MonteCarloEnergyLoss()
                else:
                    raise RuntimeError(f"MATLAB引擎启动失败且未启用回退: {str(e)}") from e
        else:
            # 使用Python实现
            self.loss_calculator = MonteCarloEnergyLoss()
            logger.info("Python后端已启用")

    def _prepare_training_data(self) -> None:
        """准备训练数据"""
        from ..data_utils.sampler import Utils
        self.data_body = torch.from_numpy(
            Utils.sample_from_disk(self.params["radius"], self.params["bodyBatch"])
        ).float().to(self.device)
        self.data_body.requires_grad = True

        self.data_boundary = torch.from_numpy(
            Utils.sample_from_surface(self.params["radius"], self.params["bdryBatch"])
        ).float().to(self.device)

    def _compute_loss(self, data_body, data_boundary) -> torch.Tensor:
        """
        计算损失

        Args:
            data_body: 内部点数据
            data_boundary: 边界点数据

        Returns:
            总损失(标量)
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
        energy_loss = self.loss_calculator.compute_energy_loss(
            output_body, grad_output, source_term, self.params["radius"]
        )

        # 边界损失
        output_boundary = self.model(data_boundary)
        target_boundary = self.model.pde.boundary_condition(data_boundary)
        boundary_loss = self.loss_calculator.compute_boundary_loss(
            output_boundary, target_boundary,
            penalty=self.params["penalty"], radius=self.params["radius"]
        )

        # 总损失
        total_loss = self.loss_calculator.compute_total_loss(energy_loss, boundary_loss)
        return torch.mean(total_loss) if total_loss.numel() > 1 else total_loss

    def train(self) -> Tuple[List[int], List[float], List[float]]:
        """
        训练模型

        Returns:
            (训练step列表, L2误差列表, H1误差列表)
        """
        self.model.train()
        steps, l2_errors, h1_errors = [], [], []
        loss_window = []

        # 确定输出文件名
        backend_prefix = "matlab" if self.use_matlab else "python"
        loss_history_file = f"{backend_prefix}_mc_loss_history.txt"
        error_history_file = f"{backend_prefix}_mc_error_history.txt"
        loss_path = os.path.join(self.data_dir, loss_history_file)

        logger.info(f"开始训练(后端: {'MATLAB' if self.use_matlab else 'Python'})")

        with open(loss_path, "w", buffering=1) as f:
            f.write("Step Loss\\n")
            for step in range(self.params["trainStep"]):
                try:
                    # 使用当前数据计算损失
                    loss = self._compute_loss(self.data_body, self.data_boundary)

                    # 反向传播和优化
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # 记录损失
                    loss_value = loss.item()
                    loss_window.append(loss_value)
                    if len(loss_window) > 100:
                        loss_window.pop(0)

                    # 定期评估和记录
                    if (step + 1) % self.params["recordStep"] == 0:
                        l2_error, h1_error = self._evaluate_errors()
                        steps.append(step + 1)
                        l2_errors.append(l2_error)
                        h1_errors.append(h1_error)

                        avg_loss = sum(loss_window) / len(loss_window)
                        logger.info(
                            f"Step {step+1}/{self.params['trainStep']}: "
                            f"Loss={avg_loss:.6e}, L2={l2_error:.6e}, H1={h1_error:.6e}"
                        )
                        f.write(f"{step+1} {avg_loss:.6e}\\n")

                    # 重新采样
                    if (step + 1) % self.params["resampleStep"] == 0:
                        self._prepare_training_data()

                except Exception as e:
                    logger.error(f"训练步骤{step+1}失败: {str(e)}")
                    if self.use_matlab and self.fallback_to_python:
                        logger.warning("尝试回退到Python实现")
                        self.use_matlab = False
                        self._setup_loss_calculator()
                        continue
                    else:
                        raise

        # 保存误差历史
        self._save_error_history(steps, l2_errors, h1_errors, error_history_file)

        logger.info("训练完成")
        return steps, l2_errors, h1_errors

    def __del__(self):
        """析构函数,确保MATLAB引擎被关闭"""
        if self.matlab_engine is not None and self.matlab_engine.is_running():
            logger.info("关闭MATLAB引擎")
            self.matlab_engine.stop_engine()
