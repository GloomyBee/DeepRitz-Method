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
    
    def __init__(self, params: dict):
        self.params = params
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
        self.figures_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
        os.makedirs(self.figures_dir, exist_ok=True)

    def save_plot(self, plt_obj, filename: str):
        """
        保存图表
        
        Args:
            plt_obj: matplotlib plt对象
            filename: 文件名
        """
        suffix = filename.split('-')[-1]
        full_filename = os.path.join(self.figures_dir, f"{filename.split('-')[0]}-{self.params['k']}-{suffix}.png")
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

        with open(os.path.join(self.data_dir, "pinn_solution_data.txt"), "w") as f:
            f.write("x y Predicted Exact\n")
            for i in range(n_sample):
                for j in range(n_sample):
                    f.write(f"{x[i,j]} {y[i,j]} {pred[i,j]} {exact_sol[i,j]}\n")

        plt.figure(figsize=(10, 6))
        plt.contourf(x, y, pred, levels=50, cmap='viridis')
        plt.colorbar(label="Predicted u(x, y)")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(
            f"RKDR Predicted Solution\n(width={self.params['width']}, depth={self.params['depth']}, penalty={self.params['penalty']}, step_size={self.params['step_size']})\n"
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
        with open(os.path.join(self.data_dir, "rkdr_loss_history.txt"), "r") as f:
            lines = f.readlines()[1:]
            for line in lines:
                step, loss = map(float, line.strip().split())
                steps.append(step)
                losses.append(loss)

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

        plt.figure(figsize=(10, 6))
        plt.plot(converge_steps, converge_losses, label="Convergence Loss", linewidth=2)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Convergence Loss (After Step 100)")
        plt.legend()
        plt.grid(True)
        self.save_plot(plt, "5-1-4")
        plt.show()

    def plot_absolute_error_distribution(self):
        """绘制绝对误差分布"""
        x, y, pred, exact = [], [], [], []
        with open(os.path.join(self.data_dir, "rkdr_solution_data.txt"), "r") as f:
            lines = f.readlines()[1:]
            for line in lines:
                xi, yi, predi, exacti = map(float, line.strip().split())
                x.append(xi)
                y.append(yi)
                pred.append(predi)
                exact.append(exacti)

        abs_error = np.abs(np.array(pred) - np.array(exact))
        x = np.array(x).reshape(self.params["nSample"], self.params["nSample"])
        y = np.array(y).reshape(self.params["nSample"], self.params["nSample"])
        abs_error = abs_error.reshape(self.params["nSample"], self.params["nSample"])

        l2_error = np.sqrt(np.mean(abs_error ** 2))
        print(f"L2 Error: {l2_error}")

        plt.figure(figsize=(10, 6))
        plt.contourf(x, y, abs_error, levels=50, cmap='viridis')
        plt.colorbar(label="Absolute Error")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"Absolute Error Distribution\n(L2 Error: {l2_error:.6f})")
        self.save_plot(plt, "5-1-2")
        plt.show()

    def plot_error_convergence(self, steps, l2_errors, h1_errors):
        """绘制误差收敛曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(steps, l2_errors, label="L2 Error", color="blue")
        plt.plot(steps, h1_errors, label="H1 Error", color="red", linestyle="--")
        plt.xlabel("Step")
        plt.ylabel("Error")
        plt.title("Error Convergence with Training Steps")
        plt.legend()
        plt.grid(True)
        self.save_plot(plt, "5-1-6")
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.semilogy(steps, l2_errors, label="Log L2 Error", color="blue")
        plt.semilogy(steps, h1_errors, label="Log H1 Error", color="red", linestyle="--")
        plt.xlabel("Step")
        plt.ylabel("Log Error")
        plt.title("Log-Error Convergence with Training Steps")
        plt.legend()
        plt.grid(True)
        self.save_plot(plt, "5-1-7")
        plt.show()


def load_model_for_visualization(model_path: str, device: str):
    """加载模型用于可视化"""
    # 加载配置文件（与训练时一致）
    config_dir = os.path.dirname(os.path.dirname(model_path))
    config_path = os.path.join(config_dir, "..", "config", "base_config.yaml")
    problem_config_path = os.path.join(config_dir, "..", "config", "poisson_2d.yaml")
    
    config = load_config(config_path, problem_config_path)
    params = config.to_dict()
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
    
    # 模型路径
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                             "output", "pinn_last_model.pt")
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return
    
    print("Loading model for visualization...")
    model, params = load_model_for_visualization(model_path, device)
    
    # 读取训练时间信息
    train_time, test_time = 0.0, 0.0
    time_file = os.path.join(os.path.dirname(model_path), "rkdr_training_time.txt")
    if os.path.exists(time_file):
        with open(time_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                if "Training Time:" in line:
                    train_time = float(line.split(":")[1].strip().split()[0])
                elif "Test Time:" in line:
                    test_time = float(line.split(":")[1].strip().split()[0])
    
    # 读取误差历史
    steps, l2_errors, h1_errors = [], [], []
    error_file = os.path.join(os.path.dirname(model_path), "rkdr_error_history.txt")
    if os.path.exists(error_file):
        with open(error_file, "r") as f:
            lines = f.readlines()[1:]
            for line in lines:
                step, l2_err, h1_err = map(float, line.strip().split())
                steps.append(int(step))
                l2_errors.append(l2_err)
                h1_errors.append(h1_err)
    
    print("Creating visualizations...")
    visualizer = Visualizer(params)
    
    visualizer.plot_result(model, device, train_time, test_time)
    visualizer.plot_loss_curve()
    visualizer.plot_absolute_error_distribution()
    
    if steps and l2_errors and h1_errors:
        visualizer.plot_error_convergence(steps, l2_errors, h1_errors)
    
    print("Visualization completed!")


if __name__ == "__main__":
    main()