import sys
import os
import torch
from heat_pde import HeatPDE
from core.models.mlp import EnhancedRitzNet
from heat_trainer_hr import HeatTrainerHR # 导入新的训练器
from config.config_loader import load_config

def main():
    # ... (前置代码同 train_heat_hr.py)

    # 加载配置
    config_path = os.path.join(os.path.dirname(__file__), "heat_config.yaml")
    config = load_config(config_path)
    
    # === 关键修改 1: 强制修改输出维度 ===
    # 我们需要输出 [T, qx, qy]
    config["dd"] = 3 
    
    # 转换为字典
    params = config.to_dict()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params["device"] = device
    
    # ... (PDE 初始化代码同 train_heat_hr.py) ...
    pde = HeatPDE(...)
    
    # ... (Kernel 生成代码同 train_heat_hr.py) ...
    
    # 创建模型 (参数 params['dd'] 已经是 3 了)
    model = EnhancedRitzNet(params, nodes, s).to(device)
    model.pde = pde
    
    print(f"网络输出维度: {params['dd']} (T, qx, qy)")
    
    # === 关键修改 2: 使用新的训练器 ===
    trainer = HeatTrainerHR(model, device, params)
    
    trainer.train()
    
    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': params
    }, "output/heat_hr_model.pt")

if __name__ == "__main__":
    main()