"""
训练脚本 - RKDR方法入口
"""

import torch
import time
import sys
import os
import math
import argparse  # 导入 argparse 模块用于处理命令行参数

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pdes.poisson import Poisson2D
from core.models.mlp import EnhancedRitzNet
from core.trainer.trainer_coll import CollocationTrainer
from core.utils import setup_matplotlib, save_model, save_training_info
from config.config_loader import load_config


def main():
    """主函数 - RKDR训练"""

    setup_matplotlib()

    # 加载配置
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "config", "base_config.yaml")
    problem_config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                       "config", "poisson_2d.yaml")

    config = load_config(config_path, problem_config_path)

    # 设备设置
    device_str = config.get("device", "auto")
    if device_str == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print(f"Using device: {device}")

    # 简单的GPU预热
    if torch.cuda.is_available():
        dummy_tensor = torch.zeros(1).to(device)
        dummy_tensor += 1

    # 参数设置 - 从配置加载
    params = config.to_dict()

    # 更新设备参数
    params["device"] = device

    # 创建PDE实例
    pde = Poisson2D(radius=params["radius"])

    # 生成径向基核节点
    kernel_config = params.get("kernel", {})
    nodes_count = kernel_config.get("nodes_count", 21)
    scale = kernel_config.get("scale", 0.5)

    nodes_x = torch.linspace(-params["radius"], params["radius"], nodes_count)
    nodes_y = torch.linspace(-params["radius"], params["radius"], nodes_count)
    nodes = torch.stack(torch.meshgrid(nodes_x, nodes_y, indexing='xy'), dim=-1).reshape(-1, 2)
    distances = torch.sqrt(torch.sum(nodes ** 2, dim=1))
    inside_disk = distances <= params["radius"]
    nodes = nodes[inside_disk].to(device)
    s = torch.tensor(scale, dtype=torch.float32).to(device)

    # 创建模型
    model = EnhancedRitzNet(params, nodes, s).to(device)
    model.pde = pde  # 将PDE实例附加到模型

    # 创建训练器
    trainer = CollocationTrainer(model, device, params)

    # 开始配点法训练
    print("\nStarting RKDR training with Collocation method...")
    start_time = time.time()
    steps, l2_errors, h1_errors = trainer.train()  # 调用配点法训练
    train_time = time.time() - start_time
    print(f"Training finished in {train_time:.2f} seconds.")
    output_prefix = "rkdr_coll"

    # 测试
    model.eval()
    start_test_time = time.time()
    l2_error, h1_error = trainer.test()
    test_time = time.time() - start_test_time
    print(f"\n--- Final Model Evaluation ---")
    print(f"Testing costs {test_time:.4f} seconds.")
    print(f"The L2 error is: {l2_error:.6f}")
    print(f"The H1 error is: {h1_error:.6f}")
    print(f"The number of parameters is: {sum(p.numel() for p in model.parameters())}")

    # 保存模型和训练信息 (使用带前缀的文件名)
    model_path = os.path.join(trainer.data_dir, f"{output_prefix}_last_model.pt")
    info_path = os.path.join(trainer.data_dir, f"{output_prefix}_training_info.txt")

    save_model(model, model_path)
    save_training_info(train_time, test_time, info_path)

    # 保存solution_data
    n_sample = params.get("nSample", 100)
    r_list = torch.linspace(0, params["radius"], n_sample)
    theta_list = torch.linspace(0, 2 * math.pi, n_sample)
    xx, yy = torch.meshgrid(r_list, theta_list, indexing='xy')
    x = xx * torch.cos(yy)
    y = xx * torch.sin(yy)
    coords = torch.stack([x.flatten(), y.flatten()], dim=-1)
    coords_tensor = coords.to(device)

    with torch.no_grad():
        pred = model(coords_tensor).cpu().numpy()
        exact = pde.exact_solution(coords_tensor).cpu().numpy()
        x_np = x.cpu().numpy()
        y_np = y.cpu().numpy()

    solution_file = os.path.join(trainer.data_dir, f"{output_prefix}_solution_data.txt")
    with open(solution_file, "w") as f:
        f.write("x y Predicted Exact\n")
        for i in range(n_sample):
            for j in range(n_sample):
                idx = i * n_sample + j
                f.write(f"{x_np[i,j]} {y_np[i,j]} {pred[idx,0]} {exact[idx,0]}\n")

    print(f"Solution data saved to {solution_file}")
    print(f"\nModel saved to {model_path}")
    print("Training completed successfully!")


if __name__ == "__main__":
    main()

