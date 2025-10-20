"""
train_pinn.py: PINN方法训练入口
"""

import torch
import time
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pdes.poisson import Poisson2D
from core.models.mlp import EnhancedRitzNet
from core.trainer_pinn import PINNTrainer
from core.utils import setup_matplotlib, save_model, save_training_info
from config.config_loader import load_config


def main():
    """主函数 - PINN训练"""
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

    if torch.cuda.is_available():
        dummy_tensor = torch.zeros(1).cuda()
        dummy_tensor += 1

    # 参数设置 - 从配置加载
    params = config.to_dict()

    # 更新设备参数
    params["device"] = device

    # 确保训练步数足够（如果配置中的trainStep太小，使用默认值）
    #if params.get("trainStep", 0) < 1000:
    #    params["trainStep"] = 20000

    # 检查边界损失权重
    if "penalty" not in params:
        params["penalty"] = 100.0  # 默认值，PINN边界损失权重

    # 创建PDE实例
    pde = Poisson2D(radius=params["radius"])

    # 生成径向基核节点
    # 从配置中读取核函数参数
    kernel_config = params.get("kernel", {})
    nodes_count = kernel_config.get("nodes_count", 21)
    scale = kernel_config.get("scale", 0.5)

    nodes_x = torch.linspace(-1.0, 1.0, nodes_count)
    nodes_y = torch.linspace(-1.0, 1.0, nodes_count)
    nodes = torch.stack(torch.meshgrid(nodes_x, nodes_y, indexing='xy'), dim=-1).reshape(-1, 2)
    distances = torch.sqrt(torch.sum(nodes ** 2, dim=1))
    inside_disk = distances <= params["radius"]
    nodes = nodes[inside_disk].to(device)
    s = torch.tensor(scale, dtype=torch.float32).to(device)

    # 创建模型（复用EnhancedRitzNet，但忽略核增强）
    model = EnhancedRitzNet(params, nodes, s).to(device)
    model.pde = pde  # 将PDE实例附加到模型

    # 创建训练器
    trainer = PINNTrainer(model, device, params)

    # 训练
    print("Starting PINN training...")
    start_time = time.time()
    steps, l2_errors, h1_errors = trainer.train()
    train_time = time.time() - start_time
    print(f"Training costs {train_time} seconds.")

    # 测试
    model.eval()
    start_test_time = time.time()
    l2_error, h1_error = trainer.test()
    test_time = time.time() - start_test_time
    print(f"Testing costs {test_time:.4f} seconds.")
    print(f"The L2 error (of the last model) is {l2_error}.")
    print(f"The H1 error (of the last model) is {h1_error}.")
    print(f"The number of parameters is {sum(p.numel() for p in model.parameters())}.")

    # 保存模型和训练信息
    save_model(model, os.path.join(trainer.data_dir, "pinn_last_model.pt"))
    save_training_info(train_time, test_time, os.path.join(trainer.data_dir, "pinn_training_time.txt"))

    print("Training completed successfully!")


if __name__ == "__main__":
    main()