# tests/test_config_loader.py

import pytest
import yaml
import os
from config.config_loader import Config  # 假设你的项目根目录在 PYTHONPATH 中

# 1. 定义测试中使用的 YAML 内容
BASE_CONFIG_CONTENT = {
    "training": {
        "lr": 0.01,
        "epochs": 100,
        "optimizer": "Adam"
    },
    "model": {
        "name": "MLP",
        "depth": 3
    }
}

PROBLEM_CONFIG_CONTENT = {
    "training": {
        "lr": 0.005,  # 覆盖基础配置
        "batch_size": 128  # 新增配置
    },
    "pde": {
        "name": "Poisson",
        "dim": 2
    }
}


# 2. 创建一个 Fixture 来准备配置文件
@pytest.fixture
def config_paths(tmp_path):
    """
    一个 Pytest fixture，它会在一个临时目录中创建
    base_config.yaml 和 problem_config.yaml，并返回它们的路径。
    """
    base_path = tmp_path / "base_config.yaml"
    prob_path = tmp_path / "problem_config.yaml"

    with open(base_path, 'w', encoding='utf-8') as f:
        yaml.dump(BASE_CONFIG_CONTENT, f)

    with open(prob_path, 'w', encoding='utf-8') as f:
        yaml.dump(PROBLEM_CONFIG_CONTENT, f)

    return base_path, prob_path


def test_load_base_config_only(config_paths):
    """测试只加载基础配置文件"""
    base_path, _ = config_paths

    # Act: 创建 Config 对象
    cfg = Config(base_config_path=str(base_path))

    # Assert: 验证配置是否正确加载
    assert cfg.get("model.name") == "MLP"
    assert cfg.get("training.lr") == 0.01
    assert cfg.get("pde") is None  # problem 特有的配置不应该存在


def test_merge_configs(config_paths):
    """测试基础配置和问题特定配置的合并逻辑"""
    base_path, prob_path = config_paths

    # Act
    cfg = Config(base_config_path=str(base_path), problem_config_path=str(prob_path))

    # Assert
    # 1. 测试覆盖：lr 应该被 problem_config 覆盖
    assert cfg.get("training.lr") == 0.005
    # 2. 测试保留：epochs 应该保留 base_config 中的值
    assert cfg.get("training.epochs") == 100
    # 3. 测试新增（嵌套）：batch_size 应该被添加
    assert cfg.get("training.batch_size") == 128
    # 4. 测试新增（顶级）：pde 应该被添加
    assert cfg.get("pde.name") == "Poisson"
    assert cfg.get("model.name") == "MLP"  # base 中的其他配置不受影响


def test_get_method(config_paths):
    """测试 get 方法的各种情况"""
    base_path, _ = config_paths
    cfg = Config(str(base_path))

    # 测试获取存在的嵌套键
    assert cfg.get("training.optimizer") == "Adam"

    # 测试获取不存在的键，应返回默认值 None
    assert cfg.get("non_existent_key") is None

    # 测试获取不存在的键，应返回指定的默认值
    assert cfg.get("non_existent_key", "default_value") == "default_value"

    # 测试获取不存在的嵌套键
    assert cfg.get("model.non_existent_param", default=42) == 42


def test_set_method(config_paths):
    """测试 set 方法的功能"""
    base_path, _ = config_paths
    cfg = Config(str(base_path))

    # 1. 覆盖一个已存在的值
    cfg.set("training.lr", 0.99)
    assert cfg.get("training.lr") == 0.99

    # 2. 添加一个新值
    cfg.set("training.patience", 10)
    assert cfg.get("training.patience") == 10

    # 3. 添加一个全新的嵌套值（这会考验它能否创建中间的字典）
    cfg.set("new_section.param.value", "hello")
    assert cfg.get("new_section.param.value") == "hello"
    assert isinstance(cfg.get("new_section"), dict)

def test_file_not_found():
    """测试当基础配置文件不存在时，是否抛出 FileNotFoundError"""
    with pytest.raises(FileNotFoundError):
        # 我们可以传入一个绝对不可能存在的路径
        Config("/path/to/non/existent/file.yaml")


def test_save_and_load_roundtrip(config_paths, tmp_path):
    """测试保存配置后，内容是否正确"""
    base_path, prob_path = config_paths
    cfg = Config(str(base_path), str(prob_path))

    # 定义保存路径
    save_path = tmp_path / "saved_config.yaml"

    # Act: 保存配置
    cfg.save(str(save_path))

    # Assert: 检查文件是否被创建
    assert os.path.exists(save_path)

    # 重新加载保存的文件，验证其内容
    reloaded_cfg = Config(str(save_path))

    assert reloaded_cfg.to_dict() == cfg.to_dict()
    assert reloaded_cfg.get("training.lr") == 0.005
