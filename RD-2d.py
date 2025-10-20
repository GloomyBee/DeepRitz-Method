import numpy as np
import math, torch, time
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import os
from datetime import datetime

# 设置 Matplotlib 参数
plt.rc('text', usetex=False)
plt.rc('mathtext', fontset='cm')
plt.rc('font', family='Arial', size=16)
plt.rc('axes', titlesize=16, labelsize=16)
plt.rc('legend', fontsize=16)
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)

# 工具函数模块
class Utils:
    @staticmethod
    def sample_from_disk(radius, num_points):
        """在圆盘内均匀采样"""
        r = np.sqrt(np.random.uniform(0, 1, num_points)) * radius
        theta = np.random.uniform(0, 2 * np.pi, num_points)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.stack((x, y), axis=-1)

    @staticmethod
    def sample_from_surface(radius, num_points):
        """在圆盘边界上均匀采样"""
        theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        return np.stack((x, y), axis=-1)

    @staticmethod
    def ffun(data):
        """源项 f(x, y) = 4"""
        return 4.0 * torch.ones([data.shape[0], 1], dtype=torch.float)

    @staticmethod
    def exact(radius, data):
        """解析解 u(x, y) = 1 - (x^2 + y^2)"""
        return (radius**2 - torch.sum(data * data, dim=1)).unsqueeze(1)

    @staticmethod
    def compute_error(output, target, params):
        """计算相对 L2 误差"""
        error = output - target
        error = math.sqrt(torch.mean(error ** 2) * math.pi * params["radius"]**2)
        ref = math.sqrt(torch.mean(target ** 2) * math.pi * params["radius"]**2)
        return error / ref

# 模型模块
class RitzNet(nn.Module):
    def __init__(self, params):
        super(RitzNet, self).__init__()
        self.params = params
        self.device = params.get("device", "cpu")

        # 网络结构
        self.linear_in = nn.Linear(params["d"], params["width"])
        self.linear = nn.ModuleList([nn.Linear(params["width"], params["width"]) for _ in range(params["depth"])])
        self.linear_out = nn.Linear(params["width"], params["dd"])

    def forward(self, x):
        x = torch.tanh(self.linear_in(x))
        for layer in self.linear:
            x = torch.tanh(layer(x))
        return self.linear_out(x)

# 训练和测试模块
class Trainer:
    def __init__(self, model, device, params):
        self.model = model
        self.device = device
        self.params = params
        self.optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["decay"])
        self.scheduler = StepLR(self.optimizer, step_size=params["step_size"], gamma=params["gamma"])
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data2")
        os.makedirs(self.data_dir, exist_ok=True)

    def test(self):
        num_quad = self.params["numQuad"]
        data = torch.from_numpy(Utils.sample_from_disk(self.params["radius"], num_quad)).float().to(self.device)
        data.requires_grad = True

        output = self.model(data)
        target = Utils.exact(self.params["radius"], data).to(self.device)

        l2_error = Utils.compute_error(output, target, self.params)

        grad_output = torch.autograd.grad(output, data, grad_outputs=torch.ones_like(output), create_graph=True)[0]
        grad_target = torch.autograd.grad(target, data, grad_outputs=torch.ones_like(target), create_graph=True)[0]

        l2_grad_error = torch.sqrt(torch.mean((grad_output - grad_target) ** 2) * math.pi * self.params["radius"]**2)
        h1_error = torch.sqrt(l2_error ** 2 + l2_grad_error ** 2)

        return l2_error, h1_error

    def train(self):
        self.model.train()
        data_body = torch.from_numpy(Utils.sample_from_disk(self.params["radius"], self.params["bodyBatch"])).float().to(self.device)
        data_body.requires_grad = True
        data_boundary = torch.from_numpy(Utils.sample_from_surface(self.params["radius"], self.params["bdryBatch"])).float().to(self.device)

        steps, l2_errors, h1_errors = [], [], []
        loss_window = []
        window_size = self.params.get("window_size", 100)
        tolerance = self.params.get("tolerance", 1e-5)

        with open(os.path.join(self.data_dir, "dr_loss_history.txt"), "w") as f:
            f.write("Step Loss\n")
            for step in range(self.params["trainStep"]):
                output_body = self.model(data_body)
                grad_output = torch.autograd.grad(output_body, data_body, grad_outputs=torch.ones_like(output_body),
                                                  retain_graph=True, create_graph=True, only_inputs=True)[0]
                dfdx = grad_output[:, 0:1]
                dfdy = grad_output[:, 1:2]

                f_term = Utils.ffun(data_body).to(self.device)
                loss_body = torch.mean(0.5 * (dfdx ** 2 + dfdy ** 2) - f_term * output_body) * math.pi * self.params["radius"]**2

                output_boundary = self.model(data_boundary)
                target_boundary = Utils.exact(self.params["radius"], data_boundary).to(self.device)
                loss_boundary = torch.mean((output_boundary - target_boundary) ** 2) * self.params["penalty"] * 2 * math.pi * self.params["radius"]

                loss = loss_body + loss_boundary

                loss_window.append(loss.item())
                if len(loss_window) > window_size:
                    loss_window.pop(0)
                if len(loss_window) == window_size:
                    loss_diff = max(loss_window) - min(loss_window)
                    if loss_diff < tolerance:
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

        with open(os.path.join(self.data_dir, "dr_error_history.txt"), "w") as f:
            f.write("Step L2_Error H1_Error\n")
            for step, l2_err, h1_err in zip(steps, l2_errors, h1_errors):
                f.write(f"{step} {l2_err} {h1_err}\n")

        return steps, l2_errors, h1_errors

# 可视化模块
class Visualizer:
    def __init__(self, params):
        self.params = params
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data2")
        self.figures_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures2")
        os.makedirs(self.figures_dir, exist_ok=True)

    def save_plot(self, plt_obj, filename):
        suffix = filename.split('-')[-1]
        full_filename = os.path.join(self.figures_dir, f"{filename.split('-')[0]}-{self.params['k']}-{suffix}.png")
        plt_obj.savefig(full_filename, dpi=300, bbox_inches='tight')
        print(f"Image saved to: {full_filename}")

    def plot_result(self, model, device, train_time, test_time):
        n_sample = 100  # 固定为 100，与原始代码一致
        r_list = np.linspace(0, self.params["radius"], n_sample)
        theta_list = np.linspace(0, 2 * np.pi, n_sample)
        xx, yy = np.meshgrid(r_list, theta_list)
        x = xx * np.cos(yy)
        y = xx * np.sin(yy)
        coords = np.stack((x.flatten(), y.flatten()), axis=-1)
        coords_tensor = torch.from_numpy(coords).float().to(device)

        pred = model(coords_tensor).detach().cpu().numpy().reshape(n_sample, n_sample)
        exact_sol = self.params["radius"]**2 - (x**2 + y**2)

        with open(os.path.join(self.data_dir, "dr_solution_data.txt"), "w") as f:
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
            f"DR Predicted Solution\n(width={self.params['width']}, depth={self.params['depth']}, penalty={self.params['penalty']}, step_size={self.params['step_size']})\n"
            f"Train Time: {train_time:.2f}s, Test Time: {test_time:.4f}s"
        )
        self.save_plot(plt, "3-1-1")
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.contourf(x, y, exact_sol, levels=50, cmap='viridis')
        plt.colorbar(label="Exact u(x, y)")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Exact Solution")
        self.save_plot(plt, "3-1-8")
        plt.show()

    def plot_loss_curve(self):
        steps, losses = [], []
        with open(os.path.join(self.data_dir, "dr_loss_history.txt"), "r") as f:
            lines = f.readlines()[1:]
            for line in lines:
                step, loss = map(float, line.strip().split())
                steps.append(step)
                losses.append(loss)

        initial_steps = [s for s in steps if s <= 100]
        initial_losses = losses[:len(initial_steps)]

        converge_steps = [s for s in steps if s >= 100]
        converge_losses = losses[len(steps) - len(converge_steps):]

        plt.figure(figsize=(10, 6))
        plt.plot(initial_steps, initial_losses, label="Initial Loss Decline", linewidth=2)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Initial Loss Decline (First 100 Steps)")
        plt.xlim(0, 100)
        plt.legend()
        plt.grid(True)
        self.save_plot(plt, "3-1-2")
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(converge_steps, converge_losses, label="Convergence Loss", linewidth=2)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Convergence Loss (After Step 100)")
        plt.legend()
        plt.grid(True)
        self.save_plot(plt, "3-1-3")
        plt.show()

    def plot_h1_error_distribution(self, model, device, h1_error):
        n_sample = 100
        r_list = np.linspace(0, self.params["radius"], n_sample)
        theta_list = np.linspace(0, 2 * np.pi, n_sample)
        xx, yy = np.meshgrid(r_list, theta_list)
        x = xx * np.cos(yy)
        y = xx * np.sin(yy)
        coords = np.stack((x.flatten(), y.flatten()), axis=-1)
        coords_tensor = torch.from_numpy(coords).float().to(device)
        coords_tensor.requires_grad = True

        pred = model(coords_tensor)
        exact_sol = Utils.exact(self.params["radius"], coords_tensor).to(device)

        absolute_error = torch.abs(pred - exact_sol).detach().cpu().numpy().reshape(n_sample, n_sample)

        grad_pred = torch.autograd.grad(pred, coords_tensor, grad_outputs=torch.ones_like(pred), create_graph=True)[0]
        grad_exact = torch.autograd.grad(exact_sol, coords_tensor, grad_outputs=torch.ones_like(exact_sol), create_graph=True)[0]
        grad_error = torch.sqrt(torch.sum((grad_pred - grad_exact) ** 2, dim=1)).detach().cpu().numpy().reshape(n_sample, n_sample)

        with open(os.path.join(self.data_dir, "dr_absolute_error_distribution.txt"), "w") as f:
            f.write("x y Absolute_Error\n")
            for i in range(n_sample):
                for j in range(n_sample):
                    f.write(f"{x[i,j]} {y[i,j]} {absolute_error[i,j]}\n")

        with open(os.path.join(self.data_dir, "dr_gradient_error_distribution.txt"), "w") as f:
            f.write("x y Gradient_Error\n")
            for i in range(n_sample):
                for j in range(n_sample):
                    f.write(f"{x[i,j]} {y[i,j]} {grad_error[i,j]}\n")

        plt.figure(figsize=(10, 6))
        plt.contourf(x, y, absolute_error, levels=50, cmap='viridis')
        plt.colorbar(label="Absolute Error")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"Absolute Error Distribution\n(H1 Error: {h1_error:.6f})")
        self.save_plot(plt, "3-1-4")
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.contourf(x, y, grad_error, levels=50, cmap='viridis')
        plt.colorbar(label="Gradient Error")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"Gradient Error Distribution\n(H1 Error: {h1_error:.6f})")
        self.save_plot(plt, "3-1-5")
        plt.show()

    def plot_error_convergence(self, steps, l2_errors, h1_errors):
        plt.figure(figsize=(10, 6))
        plt.plot(steps, l2_errors, label="L2 Error", color="blue")
        plt.plot(steps, h1_errors, label="H1 Error", color="red", linestyle="--")
        plt.xlabel("Step")
        plt.ylabel("Error")
        plt.title("Error Convergence with Training Steps")
        plt.legend()
        plt.grid(True)
        self.save_plot(plt, "3-1-6")
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.semilogy(steps, l2_errors, label="Log L2 Error", color="blue")
        plt.semilogy(steps, h1_errors, label="Log H1 Error", color="red", linestyle="--")
        plt.xlabel("Step")
        plt.ylabel("Log Error")
        plt.title("Log-Error Convergence with Training Steps")
        plt.legend()
        plt.grid(True)
        self.save_plot(plt, "3-1-7")
        plt.show()

# 主函数
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        dummy_tensor = torch.zeros(1).cuda()
        dummy_tensor += 1

    params = dict()
    params["radius"] = 1
    params["d"] = 2
    params["dd"] = 1
    params["bodyBatch"] = 1024
    params["bdryBatch"] = 1024
    params["lr"] = 0.0005
    params["width"] = 100
    params["depth"] = 3
    params["numQuad"] = 40000
    params["trainStep"] = 200000
    params["penalty"] = 500
    params["diff"] = 0.001
    params["writeStep"] = 50
    params["sampleStep"] = 200
    params["step_size"] = 1000
    params["gamma"] = 0.5
    params["decay"] = 0.0001
    params["window_size"] = 200
    params["tolerance"] = 5e-3
    params["k"] = 7
    params["device"] = device

    start_time = time.time()
    model = RitzNet(params).to(device)
    print(f"Generating network costs {time.time() - start_time:.2f} seconds.")

    trainer = Trainer(model, device, params)
    visualizer = Visualizer(params)

    start_train_time = time.time()
    steps, l2_errors, h1_errors = trainer.train()
    train_time = time.time() - start_train_time
    print(f"Training costs {train_time:.2f} seconds.")

    model.eval()
    start_test_time = time.time()
    l2_error, h1_error = trainer.test()
    test_time = time.time() - start_test_time
    print(f"Testing costs {test_time:.4f} seconds.")
    print(f"The L2 error (of the last model) is {l2_error}.")
    print(f"The H1 error (of the last model) is {h1_error}.")
    print(f"The number of parameters is {sum(p.numel() for p in model.parameters())}.")

    torch.save(model.state_dict(), os.path.join(trainer.data_dir, "dr_last_model.pt"))
    with open(os.path.join(trainer.data_dir, "dr_training_time.txt"), "w") as f:
        f.write(f"Training Time: {train_time:.2f} seconds\n")
        f.write(f"Test Time: {test_time:.4f} seconds\n")

    visualizer.plot_result(model, device, train_time, test_time)
    visualizer.plot_loss_curve()
    visualizer.plot_h1_error_distribution(model, device, h1_error)
    visualizer.plot_error_convergence(steps, l2_errors, h1_errors)

if __name__ == "__main__":
    main()