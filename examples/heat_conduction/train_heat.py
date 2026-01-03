"""
热传导问题训练脚本 - 遵循框架标准

参考 scripts/train_coll.py 的实现模式
"""

import torch
import time
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from core.models.mlp import EnhancedRitzNet
from core.utils import setup_matplotlib, save_model, save_training_info
from config.config_loader import load_config
from heat_pde import HeatPDE
from heat_trainer import HeatTrainer


def main():
    """主函数 - 热传导训练"""

    setup_matplotlib()

    print("=" * 80)
    print("热传导问题训练 - 使用DeepRitz框架")
    print("=" * 80)

    # 加载配置
    config_path = os.path.join(os.path.dirname(__file__), "heat_config.yaml")
    config = load_config(config_path)

    # 设备设置
    device_str = config.get("device", "auto")
    if device_str == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print(f"\\n使用设备: {device}")

    # GPU预热
    if torch.cuda.is_available():
        dummy_tensor = torch.zeros(1).to(device)
        dummy_tensor += 1

    # 参数设置
    params = config.to_dict()
    params["device"] = device

    # 打印配置
    print(f"\\n配置参数:")
    print(f"  导热系数 k: {params['physics']['k_thermal']} W/(m·K)")
    print(f"  体积热源 s: {params['physics']['s_source']} W/m³")
    print(f"  左边界温度: {params['physics']['T_left']} °C")
    print(f"  上边界热流: {params['physics']['q_top']} W/m²")
    print(f"  网络宽度: {params['width']}")
    print(f"  网络深度: {params['depth']}")
    print(f"  训练步数: {params['trainStep']}")
    print(f"  输出频率: 每{params['writeStep']}步")

    # 创建PDE实例
    pde = HeatPDE(
        k=params['physics']['k_thermal'],
        s=params['physics']['s_source'],
        T_left=params['physics']['T_left']
    )

    print(f"\\nPDE问题: {pde.name}")
    print(f"  几何区域: 梯形 (顶点: (0,0), (2,0), (1,1), (0,1))")
    print(f"  面积: {pde.area} m²")

    # 生成核函数节点（梯形区域内）
    kernel_config = params.get("kernel", {})
    nodes_per_dim = kernel_config.get("nodes_count", 10)
    scale = kernel_config.get("scale", 0.5)

    x_nodes = torch.linspace(0, 2, nodes_per_dim)
    y_nodes = torch.linspace(0, 1, nodes_per_dim)
    nodes = torch.stack(torch.meshgrid(x_nodes, y_nodes, indexing='xy'), dim=-1).reshape(-1, 2)

    # 过滤掉梯形外的节点
    mask = (nodes[:, 0] + nodes[:, 1] <= 2.0)
    nodes = nodes[mask].to(device)
    s = torch.tensor(scale, dtype=torch.float32).to(device)

    print(f"\\n核函数节点数: {len(nodes)}")

    # 创建模型
    model = EnhancedRitzNet(params, nodes, s).to(device)
    model.pde = pde

    print(f"\\n网络结构:")
    print(f"  输入维度: {params['d']}")
    print(f"  隐藏层宽度: {params['width']}")
    print(f"  网络深度: {params['depth']}")
    print(f"  输出维度: {params['dd']}")
    print(f"  总参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 创建训练器
    trainer = HeatTrainer(model, device, params)

    # 训练
    print("\\n开始训练...")
    start_time = time.time()
    steps, l2_errors, h1_errors = trainer.train()
    train_time = time.time() - start_time
    print(f"训练耗时: {train_time:.2f}秒")

    # 评估残差
    print("\\n" + "=" * 80)
    print("评估PDE残差")
    print("=" * 80)
    model.eval()
    start_test_time = time.time()
    residual = trainer.evaluate_residual(n_test=params["numQuad"])
    test_time = time.time() - start_test_time

    print(f"测试耗时: {test_time:.4f}秒")
    print(f"平均PDE残差: {residual:.6f}")
    print(f"网络参数量: {sum(p.numel() for p in model.parameters())}")

    # 保存模型和训练信息
    output_prefix = "heat"
    model_path = os.path.join(trainer.data_dir, f"{output_prefix}_model.pt")
    info_path = os.path.join(trainer.data_dir, f"{output_prefix}_training_info.txt")

    save_model(model, model_path)
    save_training_info(train_time, test_time, info_path)

    # 保存配置到checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': params,
        'residual': residual,
        'train_time': train_time
    }
    torch.save(checkpoint, model_path)

    print(f"\\n模型已保存到: {model_path}")
    print("训练完成！")

    print("\\n下一步:")
    print("  运行 visualize_heat.py 查看结果")


if __name__ == "__main__":
    main()
