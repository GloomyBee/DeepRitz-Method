"""
评估脚本 - 评估已训练模型
"""

import torch
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pdes.poisson import Poisson2D
from core.models.mlp import EnhancedRitzNet
from core.utils import setup_matplotlib
from config.config_loader import load_config


def load_trained_model(model_path: str, device: str):
    """
    加载训练好的模型
    
    Args:
        model_path: 模型文件路径
        device: 设备
        
    Returns:
        加载的模型
    """
    # 加载配置（需要与训练时一致）
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                           "config", "base_config.yaml")
    problem_config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                    "config", "poisson_2d.yaml")
    
    config = load_config(config_path, problem_config_path)
    
    # 参数设置 - 从配置加载
    params = config.to_dict()
    
    # 更新设备参数
    params["device"] = device

    pde = Poisson2D(radius=params["radius"])
    
    # 生成节点（需要与训练时一致）
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
    
    # 创建模型并加载权重
    model = EnhancedRitzNet(params, nodes, s).to(device)
    model.pde = pde
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model


def evaluate_model(model, data_dir: str):
    """
    评估模型性能
    
    Args:
        model: 已训练的模型
        data_dir: 数据目录
    """
    from core.trainer.trainer import Trainer
    from core.data_utils.sampler import Utils
    import math
    
    device = next(model.parameters()).device
    params = model.params
    
    # 创建临时训练器用于测试
    trainer = Trainer(model, device, params)
    
    # 测试
    l2_error, h1_error = trainer.test()
    
    print(f"Model Evaluation Results:")
    print(f"L2 Error: {l2_error:.6f}")
    print(f"H1 Error: {h1_error:.6f}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")


def main():
    """主函数"""
    setup_matplotlib()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 模型路径
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                             "output", "rkdr_last_model.pt")
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return
    
    print("Loading trained model...")
    model = load_trained_model(model_path, device)
    
    print("Evaluating model...")
    evaluate_model(model, os.path.dirname(model_path))
    
    print("Evaluation completed!")


if __name__ == "__main__":
    main()