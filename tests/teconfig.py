import os
import torch
from config.config_loader import load_config


# pprint 已经不需要了

def main():
    """主函数 - RKDR训练"""

    # 1. 加载配置
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, "config", "base_config.yaml")
    problem_config_path = os.path.join(base_dir, "config", "poisson_2d.yaml")

    config = load_config(config_path, problem_config_path)

    # 2. 设备设置
    device_str = config.get("device", "auto")
    if device_str == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    if torch.cuda.is_available():
        dummy_tensor = torch.zeros(1).cuda()
        dummy_tensor += 1

    # 3. 参数设置
    params = config.to_dict()
    params["device"] = device

    if params.get("trainStep", 0) < 1000:
        params["trainStep"] = 20000

    # 4. (假设这里开始调用训练函数)
    print("✅ Configuration prepared. Final params dictionary is ready for training.")
    print(params)


if __name__ == "__main__":
    main()

