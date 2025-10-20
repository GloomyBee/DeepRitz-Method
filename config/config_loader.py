"""
配置管理模块
"""

import yaml
import os
from typing import Dict, Any, Optional


class Config:
    """配置管理类"""
    
    def __init__(self, base_config_path: str, problem_config_path: Optional[str] = None):
        """
        初始化配置
        
        Args:
            base_config_path: 基础配置文件路径
            problem_config_path: 问题特定配置文件路径（可选）
        """
        self.base_config_path = base_config_path
        self.problem_config_path = problem_config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        加载配置文件
        
        Returns:
            合并后的配置字典
        """
        # 加载基础配置
        with open(self.base_config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 如果有问题特定配置，合并配置
        if self.problem_config_path and os.path.exists(self.problem_config_path):
            with open(self.problem_config_path, 'r', encoding='utf-8') as f:
                problem_config = yaml.safe_load(f)
                self._merge_config(config, problem_config)
        
        return config
    
    def _merge_config(self, base_config: Dict[str, Any], override_config: Dict[str, Any]):
        """
        递归合并配置字典
        
        Args:
            base_config: 基础配置
            override_config: 覆盖配置
        """
        for key, value in override_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                self._merge_config(base_config[key], value)
            else:
                base_config[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            key: 配置键
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """
        设置配置值
        
        Args:
            key: 配置键
            value: 配置值
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """
        获取配置字典
        
        Returns:
            配置字典
        """
        return self.config.copy()
    
    def save(self, filepath: str):
        """
        保存配置到文件
        
        Args:
            filepath: 保存路径
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)


def load_config(base_config_path: str, problem_config_path: Optional[str] = None) -> Config:
    """
    加载配置的便捷函数
    
    Args:
        base_config_path: 基础配置文件路径
        problem_config_path: 问题特定配置文件路径（可选）
        
    Returns:
        配置对象
    """
    return Config(base_config_path, problem_config_path)


def get_default_config() -> Dict[str, Any]:
    """
    获取默认配置
    
    Returns:
        默认配置字典
    """
    return {
        "window_size": 200,
        "tolerance": 5e-3,
        "k": 5,
        "d": 2,
        "dd": 1,
        "radius": 1.0,
        "bodyBatch": 4096,
        "bdryBatch": 4096,
        "lr": 0.001,
        "preLr": 0.01,
        "width": 100,
        "depth": 3,
        "numQuad": 40000,
        "trainStep": 20000,
        "penalty": 1000,
        "preStep": 0,
        "diff": 0.01,
        "writeStep": 10,
        "sampleStep": 200,
        "step_size": 200,
        "gamma": 0.5,
        "decay": 0.0001,
        "device": "auto",
        "nSample": 100
    }