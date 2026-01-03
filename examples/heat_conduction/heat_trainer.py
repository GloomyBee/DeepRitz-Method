"""
热传导问题训练器 - 遵循框架标准

参考 core/trainer/trainer_coll.py 的实现模式
"""

import torch
import os
from typing import List, Tuple
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from core.trainer.base_trainer import BaseTrainer
from trapezoid_sampler import TrapezoidSampler


class HeatTrainer(BaseTrainer):
    """热传导问题训练器"""

    def __init__(self, model, device: str, params: dict):
        """
        初始化训练器

        Args:
            model: 神经网络模型（包含pde属性）
            device: 计算设备
            params: 配置参数字典
        """
        super().__init__(model, device, params)

        # 准备训练数据
        self._prepare_training_data()

    def _prepare_training_data(self) -> None:
        """准备训练数据（固定内部点，动态边界点）"""
        # 内部点（固定）
        data_body_np = TrapezoidSampler.sample_domain(self.params["bodyBatch"])
        self.data_body = torch.from_numpy(data_body_np).float().to(self.device)
        self.data_body.requires_grad = True

        # 左边界点（Dirichlet BC）
        data_left_np = TrapezoidSampler.sample_left_boundary(self.params["bdryBatch"])
        self.data_left = torch.from_numpy(data_left_np).float().to(self.device)

        # 上边界点（Neumann BC）
        data_top_np = TrapezoidSampler.sample_top_boundary(self.params["bdryBatch"])
        self.data_top = torch.from_numpy(data_top_np).float().to(self.device)

    def _compute_loss(self) -> torch.Tensor:
        """
        计算总损失

        Returns:
            总损失张量
        """
        # 1. 计算能量损失（内部积分）
        output_body = self.model(self.data_body)

        # 计算梯度
        grad_output = torch.autograd.grad(
            outputs=output_body,
            inputs=self.data_body,
            grad_outputs=torch.ones_like(output_body),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        # 梯度能量项：0.5*k*|∇T|²
        grad_squared = torch.sum(grad_output ** 2, dim=1, keepdim=True)
        k_thermal = self.params["physics"]["k_thermal"]
        energy_term = 0.5 * k_thermal * grad_squared

        # 源项：-s*T
        s_source = self.params["physics"]["s_source"]
        source_term = -s_source * output_body

        # 蒙特卡洛积分
        area = self.params["trapezoid"]["area"]
        energy_loss = torch.mean(energy_term + source_term) * area

        # 2. Dirichlet边界损失（左边界）
        output_left = self.model(self.data_left)
        T_left = self.params["physics"]["T_left"]
        bc_left_error = (output_left - T_left) ** 2
        len_left = self.params["trapezoid"]["len_left"]
        bc_left_loss = self.params["penalty"] * torch.mean(bc_left_error) * len_left

        # 3. Neumann边界损失（上边界）
        output_top = self.model(self.data_top)
        q_top = self.params["physics"]["q_top"]
        len_top = self.params["trapezoid"]["len_top"]
        bc_top_loss = q_top * torch.mean(output_top) * len_top

        # 总损失
        total_loss = energy_loss + bc_left_loss + bc_top_loss

        return total_loss

    def train(self) -> Tuple[List[int], List[float], List[float]]:
        """
        训练模型

        Returns:
            (训练步数列表, L2误差列表, H1误差列表)
        """
        print("\\n" + "=" * 80)
        print("开始训练热传导问题")
        print("=" * 80)

        self.model.train()
        steps, l2_errors, h1_errors = [], [], []
        loss_window = []

        # 准备输出文件
        loss_history_file = "heat_loss_history.txt"
        error_history_file = "heat_error_history.txt"
        loss_path = os.path.join(self.data_dir, loss_history_file)

        with open(loss_path, "w") as f:
            f.write("Step Loss\\n")

            for step in range(self.params["trainStep"]):
                # 计算损失
                loss = self._compute_loss()

                # 检查收敛
                if self._check_convergence(loss_window, loss.item(), step):
                    break

                # 记录损失
                f.write(f"{step} {loss.item()}\\n")

                # 定期评估和记录进度
                if step % self.params["writeStep"] == 0:
                    # 注意：热传导问题没有解析解，跳过误差评估
                    # 只记录损失和残差
                    print(f"Step {step}/{self.params['trainStep']}: Loss = {loss.item():.6f}")
                    steps.append(step)
                    l2_errors.append(0.0)  # 占位
                    h1_errors.append(0.0)  # 占位

                # 周期性重新采样边界点
                if step % self.params.get("sampleStep", 200) == 0 and step > 0:
                    data_left_np = TrapezoidSampler.sample_left_boundary(self.params["bdryBatch"])
                    self.data_left = torch.from_numpy(data_left_np).float().to(self.device)

                    data_top_np = TrapezoidSampler.sample_top_boundary(self.params["bdryBatch"])
                    self.data_top = torch.from_numpy(data_top_np).float().to(self.device)

                # 优化步骤
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

        print(f"\\n训练完成！最终损失: {loss.item():.6f}")

        # 保存训练历史
        self._save_history(steps, l2_errors, h1_errors, loss_history_file, error_history_file)

        return steps, l2_errors, h1_errors

    def evaluate_residual(self, n_test: int = 10000) -> float:
        """
        评估PDE残差

        Args:
            n_test: 测试点数

        Returns:
            平均残差
        """
        self.model.eval()

        # 采样测试点
        test_data = torch.from_numpy(
            TrapezoidSampler.sample_domain(n_test)
        ).float().to(self.device)
        test_data.requires_grad = True

        # 前向传播
        u = self.model(test_data)

        # 计算一阶导数
        grad_u = torch.autograd.grad(
            u, test_data,
            grad_outputs=torch.ones_like(u),
            create_graph=True
        )[0]

        # 计算二阶导数（拉普拉斯算子）
        laplacian = 0
        for i in range(2):
            grad_i = grad_u[:, i:i+1]
            grad2_i = torch.autograd.grad(
                grad_i, test_data,
                grad_outputs=torch.ones_like(grad_i),
                create_graph=True
            )[0][:, i:i+1]
            laplacian += grad2_i

        # PDE残差：-k∇²T - s
        k_thermal = self.params["physics"]["k_thermal"]
        s_source = self.params["physics"]["s_source"]
        residual = -k_thermal * laplacian - s_source
        mean_residual = torch.mean(torch.abs(residual)).item()

        self.model.train()
        return mean_residual


if __name__ == "__main__":
    print("HeatTrainer 类定义完成")
    print("请使用 train_heat_hr.py 脚本进行训练")
