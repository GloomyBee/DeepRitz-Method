# DeepRitz Framework

基于Deep Ritz方法的科学计算框架，用于求解偏微分方程。该项目提供了完整的模块化架构，支持多种求解方法包括RKDR（Radial Kernel Distribution Regression）和PINN方法。

## 项目概述

该项目实现了一个多方法的深度学习框架，特别针对泊松方程问题进行了优化。框架采用模块化设计，支持不同的数值求解方法，便于研究比较和扩展。

### 主要特性

- 🧠 **RKDR方法**: 基于径向核函数分布回归的改进Deep Ritz方法
- 🧮 **PINN方法**: 基于物理信息的神经网络方法
- 📐 **配点法**: 基于数值积分的高精度训练方法
- 🏗️ **模块化架构**: 清晰的代码结构，便于维护和扩展
- ⚙️ **配置管理**: 使用YAML文件管理实验参数
- 📊 **可视化工具**: 完整的结果可视化和分析工具
- 🔧 **易于扩展**: 基于抽象基类设计，支持添加新的PDE问题

## 项目结构

```
DeepRitz/
├── config/                  # 配置文件
│   ├── base_config.yaml      # 基础配置文件
│   ├── poisson_2d.yaml      # 泊松方程特定配置
│   └── config_loader.py     # 配置加载管理
│
├── core/                   # 核心源代码包
│   ├── pdes/               # PDE定义模块
│   │   ├── base_pde.py     # PDE抽象基类
│   │   └── poisson.py      # 泊松方程实现
│   ├── models/             # 神经网络模型
│   │   ├── base_model.py   # 模型基类
│   │   └── mlp.py          # 增强Ritz网络
│   ├── data_utils/        # 数据处理工具
│   │   └── sampler.py     # 采样和计算工具
│   ├── loss/               # 损失函数模块
│   │   ├── losses.py       # DeepRitz损失函数（蒙特卡洛）
│   │   ├── losses_pinn.py  # PINN损失函数
│   │   └── losses_coll.py # 配点法损失函数
│   ├── trainer/            # 训练器模块
│   │   ├── trainer.py      # DeepRitz训练器
│   │   ├── trainer_pinn.py # PINN训练器
│   │   └── trainer_coll.py # 配点法训练器
│   └── utils.py           # 通用工具
│
├── scripts/              # 执行脚本
│   ├── train.py          # DeepRitz训练（蒙特卡洛）
│   ├── train_pinn.py     # PINN训练
│   ├── train_coll.py     # RKDR训练（配点法）
│   ├── evaluate.py       # 模型评估
│   └── visualize.py      # 结果可视化
│
├── tests/                # 测试文件
├── output/               # 训练输出（自动创建）
├── figures/              # 图表输出（自动创建）
├── RD-2d.py            # 原始DeepRitz实现
└── RKDR-2d.py          # 原始RKDR实现
```

## 核心组件

### 1. PDE模块 (`core/pdes/`)

#### `base_pde.py`
所有偏微分方程的抽象基类，定义了标准接口：

```python
class BasePDE(ABC):
    @abstractmethod
    def source_term(self, data: torch.Tensor) -> torch.Tensor
    
    @abstractmethod
    def exact_solution(self, data: torch.Tensor) -> torch.Tensor
    
    @abstractmethod
    def boundary_condition(self, data: torch.Tensor) -> torch.Tensor
```

#### `poisson.py`
二维泊松方程的具体实现：

- **源项**: f(x, y) = 4
- **解析解**: u(x, y) = 1 - (x² + y²)
- **边界条件**: Dirichlet边界条件

### 2. 模型模块 (`core/models/`)

#### `base_model.py`
神经网络模型的基础接口，提供统一的模型结构标准。

#### `mlp.py`
增强的Ritz网络实现，包含高斯核特征：

```python
class EnhancedRitzNet(BaseModel):
    def compute_kernel_features(self, x: torch.Tensor) -> torch.Tensor:
        # 计算高斯核特征
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 前向传播，结合核特征和神经网络
```

### 3. 数据处理 (`core/data_utils/`)

#### `sampler.py`
采样和计算工具类：

- `sample_from_disk()`: 圆盘内均匀采样
- `sample_from_surface()`: 边界采样
- `gaussian_kernel()`: 高斯核函数
- `compute_error()`: 相对L2误差计算

### 4. 损失函数 (`core/loss/`)

#### `losses.py` - DeepRitz损失函数（蒙特卡洛积分）
- `compute_energy_loss()`: 能量泛函损失（蒙特卡洛）
- `compute_boundary_loss()`: 边界条件损失
- `compute_total_loss()`: 总损失计算

#### `losses_pinn.py` - PINN损失函数
- `compute_pde_loss()`: PDE残差损失
- `compute_bc_loss()`: 边界条件损失
- `compute_total_pinn_loss()`: 加权总损失

#### `losses_coll.py` - 配点法损失函数（数值积分）
- `compute_energy_loss_quadrature()`: 能量泛函损失（配点法）
- `compute_boundary_loss_quadrature()`: 边界条件损失
- `compute_total_loss_quadrature()`: 总损失计算

### 5. 训练器 (`core/trainer/`)

#### `trainer.py` - DeepRitz训练器
- 标准DeepRitz方法训练逻辑
- 蒙特卡洛积分实现
- 收敛判断和评估机制

#### `trainer_pinn.py` - PINN训练器
- 物理信息神经网络训练
- 基于PDE残差的损失优化
- 自动微分计算二阶导数

#### `trainer_coll.py` - 配点法训练器
- RKDR方法专用训练器
- 固定配点和数值积分
- 高精度计算实现

### 6. 工具函数 (`core/utils.py`)

- Matplotlib参数设置
- 输出目录管理
- 模型保存/加载
- 训练信息记录

## 快速开始

### 环境要求

```bash
Python >= 3.8
PyTorch >= 1.8.0
NumPy >= 1.19.0
Matplotlib >= 3.3.0
PyYAML >= 5.4.0
```

### 安装依赖

```bash
# 创建虚拟环境
python -m venv venv_linux
source venv_linux/bin/activate  # Linux/Mac
# 或 venv_linux\Scripts\activate  # Windows

# 安装依赖
pip install torch numpy matplotlib pyyaml
```

### 基本使用

#### 1. DeepRitz方法（蒙特卡洛积分）

```bash
python scripts/train.py
```

#### 2. RKDR方法（配点法）

```bash
python scripts/train_coll.py
```

#### 3. PINN方法（物理信息神经网络）

```bash
python scripts/train_pinn.py
```

所有训练脚本将：
- 自动创建输出目录
- 记录训练损失和误差历史
- 保存训练好的模型
- 生成训练时间统计
- 支持GPU加速

#### 4. 评估模型

```bash
python scripts/evaluate.py
```

评估脚本将：
- 加载训练好的模型
- 计算L2和H1误差
- 显示模型参数数量
- 输出评估结果

#### 5. 可视化结果

```bash
python scripts/visualize.py
```

可视化将生成：
- 预测解和解析解对比图
- 损失收敛曲线
- 绝对误差分布
- 误差收敛曲线

### 训练方法对比

| 方法 | 积分方式 | 训练器 | 损失函数 | 特点 |
|------|----------|---------|----------|------|
| DeepRitz | 蒙特卡洛 | Trainer | losses.py | 标准方法，随机采样 |
| RKDR | 数值积分 | CollocationTrainer | losses_coll.py | 高精度，固定配点 |
| PINN | 无（残差） | PINNTrainer | losses_pinn.py | 基于物理约束 |

## 配置系统

### 基础配置 (`config/base_config.yaml`)

包含所有通用参数：
- 网络结构参数（宽度、深度）
- 训练参数（学习率、批大小）
- 采样参数（点数、频率）
- 设备和输出设置

### 问题特定配置 (`config/poisson_2d.yaml`)

针对泊松方程的特定设置：
- PDE类型和维度
- 源项和边界条件
- 域和核函数参数

### 配置使用示例

```python
from config.config_loader import load_config

# 加载配置
config = load_config("config/base_config.yaml", "config/poisson_2d.yaml")

# 获取参数
learning_rate = config.get("training.learning_rate")
network_width = config.get("network.hidden_width")
```

## 扩展开发

### 添加新的PDE问题

1. 在 `core/pdes/` 中创建新的PDE类，继承 `BasePDE`
2. 实现 `source_term()`, `exact_solution()`, `boundary_condition()` 方法
3. 在 `core/pdes/__init__.py` 中导入新类

```python
# core/pdes/wave.py
from .base_pde import BasePDE

class WaveEquation2D(BasePDE):
    def source_term(self, data):
        # 实现源项
        pass
    
    def exact_solution(self, data):
        # 实现解析解
        pass
    
    def boundary_condition(self, data):
        # 实现边界条件
        pass
```

### 添加新的模型架构

1. 在 `core/models/` 中创建新的模型类，继承 `BaseModel`
2. 实现 `forward()` 方法
3. 在 `core/models/__init__.py` 中导入新类

```python
# core/models/fno.py
from .base_model import BaseModel

class FourierNeuralOperator(BaseModel):
    def __init__(self, params):
        super().__init__(params)
        # 初始化网络层
    
    def forward(self, x):
        # 实现前向传播
        pass
```

### 添加新的损失函数

1. 在 `core/losses.py` 中添加新的损失计算函数
2. 更新 `trainer.py` 中的损失计算逻辑

### 创建新的实验配置

1. 在 `config/` 中创建新的YAML配置文件
2. 基于 `base_config.yaml` 覆盖特定参数

## 输出文件说明

### 训练输出

#### DeepRitz方法（蒙特卡洛）
- `rkdr_mc_last_model.pt`: 训练好的模型权重
- `rkdr_mc_loss_history.txt`: 训练损失历史
- `rkdr_mc_error_history.txt`: 误差历史
- `rkdr_mc_training_info.txt`: 训练信息统计

#### RKDR方法（配点法）
- `rkdr_coll_last_model.pt`: 训练好的模型权重
- `rkdr_coll_loss_history.txt`: 训练损失历史
- `rkdr_coll_error_history.txt`: 误差历史
- `rkdr_coll_training_info.txt`: 训练信息统计

#### PINN方法
- `pinn_last_model.pt`: 训练好的模型权重
- `pinn_loss_history.txt`: 训练损失历史
- `pinn_error_history.txt`: 误差历史
- `pinn_training_time.txt`: 训练时间统计

### 可视化输出
- `5-1-1-*.png`: 预测解图像
- `5-1-2-*.png`: 绝对误差分布
- `5-1-3-*.png`: 初始损失下降
- `5-1-4-*.png`: 收敛阶段损失
- `5-1-5-*.png`: H1误差分布
- `5-1-6-*.png`: 误差收敛曲线
- `5-1-7-*.png`: 对数误差收敛曲线
- `5-1-8-*.png`: 解析解图像

### 解数据文件
- `*_solution_data.txt`: 解数据（预测值vs解析解）
- `*_error_history.txt`: L2和H1误差历史记录

## 性能优化

### GPU支持
框架自动检测并使用GPU：

```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

### 内存管理
- 使用动态采样减少内存占用
- 支持批次大小调整
- 自动梯度计算优化

### 收敛控制
- 基于损失窗口的收敛判断
- 可配置的容忍度参数
- 早停机制防止过训练

## 调试和测试

### 常见问题
1. **CUDA内存不足**: 减小 `bodyBatch` 和 `bdryBatch`
2. **训练不收敛**: 调整学习率或增加网络容量
3. **梯度爆炸**: 检查损失函数和权重初始化
4. **导入错误**: 确保使用正确的训练器类和导入路径

### 测试策略
框架设计了测试目录结构，支持：
- 单元测试：测试各个模块功能
- 集成测试：测试完整训练流程
- 回归测试：确保修改不破坏现有功能

### 🔧 已知问题修复（2025-10-26）
项目已修复以下主要逻辑问题：
- **循环依赖问题**: 统一了导入路径，使用相对导入
- **重复类定义问题**: 重命名训练器类，避免命名冲突
- **损失函数混乱**: 分离了不同方法的损失函数，使用明确命名
- **脚本导入错误**: 修复了错误的导入路径
- **训练方法匹配**: 修复了训练方法调用逻辑
- **可视化硬编码**: 从配置文件加载参数，而非硬编码

### 🔧 架构优势
- **模块化设计**: 清晰的职责分离，便于维护
- **配置统一**: 所有组件使用相同的配置系统
- **方法多样**: 支持多种求解方法，便于比较研究
- **扩展性强**: 基于抽象基类，易于添加新功能

## 贡献指南

1. 遵循现有的代码结构和命名约定
2. 添加适当的文档字符串和注释
3. 确保新功能有对应的测试
4. 更新相关文档

## 许可证

本项目遵循学术研究用途许可证。

## 引用

如果您在工作中使用了本框架，请引用相关的Deep Ritz方法论文。

---

**注意**: 原始的 `RKDR-2d.py` 和 `RD-2d.py` 文件保留在根目录中，确保向后兼容性。新的模块化框架提供了更好的维护性、扩展性和多种求解方法支持，建议在新项目中使用。

**最新更新**: 2025-10-26，项目已完成主要逻辑问题修复，架构更加稳定，支持多种训练方法和完整的可视化流程。