"""
可视化脚本 - 对结果进行可视化
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pdes.poisson import Poisson2D
from core.models.mlp import EnhancedRitzNet
from core.data_utils.sampler import Utils
from core.utils import setup_matplotlib
from config.config_loader import load_config


class Visualizer:
    """可视化类"""

    def __init__(self, params: dict, method_prefix: str = "rkdr"):
        self.params = params
        self.method_prefix = method_prefix
        # 获取项目根目录
        current_file = os.path.abspath(__file__)
        scripts_dir = os.path.dirname(current_file)
        project_root = os.path.dirname(scripts_dir)
        self.data_dir = os.path.join(project_root, "output")
        self.figures_dir = os.path.join(project_root, "figures")
        os.makedirs(self.figures_dir, exist_ok=True)

    def save_plot(self, plt_obj, filename: str):
        """
        保存图表

        Args:
            plt_obj: matplotlib plt对象
            filename: 文件名
        """
        suffix = filename.split('-')[-1]
        base_name = filename.split('-')[0]
        full_filename = os.path.join(self.figures_dir, f"{base_name}-{self.params['k']}-{suffix}.png")
        plt_obj.savefig(full_filename, dpi=300, bbox_inches='tight')
        print(f"Image saved to: {full_filename}")

    def plot_result(self, model, device, train_time: float, test_time: float):
        """
        绘制结果

        Args:
            model: 训练好的模型
            device: 设备
            train_time: 训练时间
            test_time: 测试时间
        """
        n_sample = self.params["nSample"]
        r_list = np.linspace(0, self.params["radius"], n_sample)
        theta_list = np.linspace(0, 2 * np.pi, n_sample)
        xx, yy = np.meshgrid(r_list, theta_list)
        x = xx * np.cos(yy)
        y = xx * np.sin(yy)
        coords = np.stack((x.flatten(), y.flatten()), axis=-1)
        coords_tensor = torch.from_numpy(coords).float().to(device)

        pred = model(coords_tensor).detach().cpu().numpy().reshape(n_sample, n_sample)

        # 使用PDE实例计算解析解
        pde = Poisson2D(radius=self.params["radius"])
        exact_sol = pde.exact_solution(coords_tensor).detach().cpu().numpy().reshape(n_sample, n_sample)

        # 保存解数据文件（使用方法前缀）
        solution_data_file = f"{self.method_prefix}_solution_data.txt"
        with open(os.path.join(self.data_dir, solution_data_file), "w") as f:
            f.write("x y Predicted Exact\n")
            for i in range(n_sample):
                for j in range(n_sample):
                    f.write(f"{x[i,j]} {y[i,j]} {pred[i,j]} {exact_sol[i,j]}\n")

        plt.figure(figsize=(10, 6))
        plt.contourf(x, y, pred, levels=50, cmap='viridis')
        plt.colorbar(label="Predicted u(x, y)")
        plt.xlabel("x")
        plt.ylabel("y")
        method_name = self.method_prefix.upper()
        plt.title(f"{method_name} Predicted Solution\n(width={self.params['width']}, depth={self.params['depth']}, penalty={self.params['penalty']}, step_size={self.params['step_size']})\n"
            f"Train Time: {train_time:.2f}s, Test Time: {test_time:.2f}s"
        )
        self.save_plot(plt, "5-1-1")
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.contourf(x, y, exact_sol, levels=50, cmap='viridis')
        plt.colorbar(label="Exact u(x, y)")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Exact Solution")
        self.save_plot(plt, "5-1-8")
        plt.show()

    def plot_loss_curve(self):
        """绘制损失曲线"""
        steps, losses = [], []
        loss_history_file = f"{self.method_prefix}_loss_history.txt"
        with open(os.path.join(self.data_dir, loss_history_file), "r") as f:
            lines = f.readlines()[1:]
            for line in lines:
                step, loss = map(float, line.strip().split())
                steps.append(step)
                losses.append(loss)

        if not steps or not losses:
            print(f"No loss data found in {loss_history_file}")
            return

        initial_steps = [s for s in steps if s <= 100]
        initial_losses = [losses[i] for i in range(len(steps)) if steps[i] <= 100]

        converge_steps = [s for s in steps if s > 100]
        converge_losses = [losses[i] for i in range(len(steps)) if steps[i] > 100]

        plt.figure(figsize=(10, 6))
        plt.plot(initial_steps, initial_losses, label="Initial Loss Decline", linewidth=2)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Initial Loss Decline (First 100 Steps)")
        plt.xlim(0, 100)
        plt.legend()
        plt.grid(True)
        self.save_plot(plt, "5-1-3")
        plt.show()

        if converge_steps and converge_losses:
            plt.figure(figsize=(10, 6))
            plt.plot(converge_steps, converge_losses, label="Convergence Loss", linewidth=2, color='orange')
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.title("Loss Convergence (After 100 Steps)")
            plt.legend()
            plt.grid(True)
            self.save_plot(plt, "5-1-3-converge")
            plt.show()

    def plot_error_curves(self):
        """绘制误差曲线"""
        steps, l2_errors, h1_errors = [], [], []
        error_history_file = f"{self.method_prefix}_error_history.txt"
        with open(os.path.join(self.data_dir, error_history_file), "r") as f:
            lines = f.readlines()[1:]
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 3:
                    step, l2_err, h1_err = float(parts[0]), float(parts[1]), float(parts[2])
                    steps.append(int(step))
                    l2_errors.append(l2_err)
                    h1_errors.append(h1_err)

        if not steps or not l2_errors:
            print(f"No error data found in {error_history_file}")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(steps, l2_errors, label="L2 Error", color="blue", linewidth=2)
        plt.plot(steps, h1_errors, label="H1 Error", color="red", linewidth=2, linestyle="--")
        plt.xlabel("Step")
        plt.ylabel("Error")
        method_name = self.method_prefix.upper()
        plt.title(f"{method_name} Error Convergence")
        plt.legend()
        plt.grid(True)
        self.save_plot(plt, "5-1-4")
        plt.show()

    def plot_log_error_convergence(self, steps: list, l2_errors: list, h1_errors: list):
        """绘制对数误差收敛曲线"""
        plt.figure(figsize=(10, 6))
        plt.semilogy(steps, l2_errors, label="Log L2 Error", color="blue")
        plt.semilogy(steps, h1_errors, label="Log H1 Error", color="red", linestyle="--")
        plt.xlabel("Step")
        plt.ylabel("Log Error")
        method_name = self.method_prefix.upper()
        plt.title(f"{method_name} Log-Error Convergence")
        plt.legend()
        plt.grid(True)
        self.save_plot(plt, "5-1-7")
        plt.show()


def load_model_for_visualization(model_path: str, device: str):
    """加载模型用于可视化（简化版本，不依赖配置文件）"""
    # 直接使用默认参数加载模型
    params = {
        'width': 10, 'depth': 2, 'k': 5, 'penalty': 100.0,
        'step_size': 10, 'gamma': 0.9, 'radius': 1.0, 'nSample': 41, 'd': 10
    }
    params["device"] = device

    # 从配置中读取核函数参数
    kernel_config = params.get("kernel", {})
    nodes_count = kernel_config.get("nodes_count", 21)
    scale = kernel_config.get("scale", 0.5)

    # 生成节点（与训练时一致）
    nodes_x = torch.linspace(-params["radius"], params["radius"], nodes_count)
    nodes_y = torch.linspace(-params["radius"], params["radius"], nodes_count)
    nodes = torch.stack(torch.meshgrid(nodes_x, nodes_y, indexing='xy'), dim=-1).reshape(-1, 2)
    distances = torch.sqrt(torch.sum(nodes ** 2, dim=1))
    inside_disk = distances <= params["radius"]
    nodes = nodes[inside_disk].to(device)
    s = torch.tensor(scale, dtype=torch.float32).to(device)

    model = EnhancedRitzNet(params, nodes, s).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model, params


def main():
    """主函数"""
    setup_matplotlib()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 获取项目根目录
    current_file = os.path.abspath(__file__)
    scripts_dir = os.path.dirname(current_file)
    project_root = os.path.dirname(scripts_dir)
    output_dir = os.path.join(project_root, "output")

    # 尝试不同的模型文件和方法前缀
    model_files = [
        ("pinn_last_model.pt", "pinn"),
        ("rkdr_mc_last_model.pt", "rkdr_mc"),
        ("rkdr_coll_last_model.pt", "rkdr_coll"),
        ("rkdr_last_model.pt", "rkdr")  # 兼容旧文件名
    ]

    model = None
    method_prefix = None
    for model_file, prefix in model_files:
        model_path = os.path.join(output_dir, model_file)
        if os.path.exists(model_path):
            print(f"Loading {prefix} model from: {model_path}")
            model, params = load_model_for_visualization(model_path, device)
            method_prefix = prefix
            break
        else:
            print(f"Model file not found: {model_path}")

    if model is None:
        print("No trained model found in output directory!")
        print("Please run one of the training scripts first:")
        print("  - python scripts/train.py")
        print("  - python scripts/train_pinn.py")
        print("  - python scripts/train_coll.py")
        return

    # 读取训练时间信息
    train_time, test_time = 0.0, 0.0
    time_file = os.path.join(output_dir, f"{method_prefix}_training_time.txt")
    if os.path.exists(time_file):
        with open(time_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                if "Training Time:" in line:
                    train_time = float(line.split(":")[1].strip().split()[0])
                elif "Test Time:" in line:
                    test_time = float(line.split(":")[1].strip().split()[0])

    print("Creating visualizations...")
    visualizer = Visualizer(params, method_prefix)

    visualizer.plot_result(model, device, train_time, test_time)
    visualizer.plot_loss_curve()
    visualizer.plot_error_curves()
    visualizer.plot_log_error_convergence([], [], [])

    print("Visualization completed!")


if __name__ == "__main__":
    main()