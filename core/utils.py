"""
通用工具函数
"""
import torch
import os
import matplotlib.pyplot as plt
from typing import Any, Dict


def setup_matplotlib():
    """设置Matplotlib参数"""
    plt.rc('text', usetex=False)
    plt.rc('mathtext', fontset='cm')
    plt.rc('font', family='Arial', size=16)
    plt.rc('axes', titlesize=16, labelsize=16)
    plt.rc('legend', fontsize=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)


def create_output_directory(base_dir: str = None) -> str:
    """
    创建输出目录
    
    Args:
        base_dir: 基础目录路径
        
    Returns:
        输出目录路径
    """
    if base_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_model(model, filepath: str):
    """
    保存模型
    
    Args:
        model: 要保存的模型
        filepath: 保存路径
    """
    torch.save(model.state_dict(), filepath)


def load_model(model, filepath: str):
    """
    加载模型
    
    Args:
        model: 模型实例
        filepath: 模型文件路径
    """
    model.load_state_dict(torch.load(filepath))
    model.eval()


def save_training_info(train_time: float, test_time: float, filepath: str):
    """
    保存训练信息
    
    Args:
        train_time: 训练时间
        test_time: 测试时间
        filepath: 保存路径
    """
    with open(filepath, "w") as f:
        f.write(f"Training Time: {train_time} seconds\n")
        f.write(f"Test Time: {test_time} seconds\n")