# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目意识

### 🔄 项目意识和上下文
- **在新对话开始时始终阅读 `INITIAL.md`**，以了解项目的架构、目标、风格和约束。
- **在开始新任务之前检查 `TASK.md`**。如果任务未列出，请添加简要描述和今天的日期。
- **使用 `INITIAL.md` 中描述的一致命名约定、文件结构和架构模式**。
- **执行 Python 命令时使用 venv_linux**（虚拟环境），包括单元测试。

### 🧱 代码结构和模块化
- **永远不要创建超过 500 行代码的文件。** 如果文件接近此限制，请通过将其拆分为模块或辅助文件来重构。
- **将代码组织成清晰分离的模块**，按功能或职责分组。
  当前项目结构：
    - `core/` - 核心算法实现
      - `pdes/` - PDE问题定义（poisson.py等）
      - `models/` - 神经网络模型（mlp.py等）
      - `loss/` - 损失函数（losses.py, losses_coll.py等）
      - `trainer/` - 训练器（trainer.py, trainer_pinn.py, trainer_coll.py）
      - `data_utils/` - 数据采样和工具（sampler.py）
    - `scripts/` - 训练和评估脚本
    - `config/` - 配置文件管理
- **使用清晰、一致的导入**（在包内优先使用相对导入）。
- **使用配置管理系统**：所有参数通过 `config.config_loader.load_config` 统一加载。

### 🧪 测试和可靠性
- **当前项目正在建立测试框架**，测试目录已规划但尚未完全实现。
- **手动测试**：直接运行各个脚本文件进行功能验证
- **核心模块测试**：可以单独测试 `core/` 目录下的模块
- **配置测试**：验证 `config/config_loader.py` 的配置加载功能

### 🔧 已修复的逻辑问题（2025-10-26）
项目已修复以下主要逻辑问题：
1. **循环依赖问题** - 统一了 `trainer_coll.py` 中的导入路径，使用相对导入
2. **重复类定义问题** - 重命名 `trainer_coll.py` 中的类为 `CollocationTrainer`，避免与 `Trainer` 冲突
3. **损失函数混乱** - 清理了 `losses_coll.py`，使用明确的配点法专用函数名（`_quadrature` 前缀）
4. **脚本导入路径** - 修复了 `train_pinn.py` 中错误的导入路径
5. **训练方法匹配** - 修复了 `train_coll.py` 中的训练方法调用逻辑
6. **可视化硬编码** - 修改 `visualize.py` 从配置文件加载参数，而不是硬编码
7. **工具函数导入** - 确认 `mlp.py` 中的工具函数导入正确

### 🔧 关键设计模式
- **训练器分离**：`Trainer`（标准DeepRitz）、`PINNTrainer`（物理信息神经网络）、`CollocationTrainer`（配点法）
- **损失函数分离**：蒙特卡洛积分版本（`losses.py`）vs 数值积分版本（`losses_coll.py`）
- **配置统一管理**：通过 `config/config_loader.py` 统一加载基础配置和问题特定配置

### ✅ 任务完成
- **完成任务后立即在 `TASK.md` 中标记已完成的任务**。
- 将开发过程中发现的新子任务或待办事项添加到 `TASK.md` 的"工作中发现"部分。

### 📎 风格和约定
- **使用 Python** 作为主要语言。
- **遵循 PEP8**，使用类型提示。
- **使用 `pydantic` 进行数据验证**。
- 如果适用，使用 `FastAPI` 用于 API，使用 `SQLAlchemy` 或 `SQLModel` 用于 ORM。
- 使用 Google 风格**为每个函数编写文档字符串**：
  ```python
  def example():
      """
      简要摘要。

      Args:
          param1 (type): 描述。

      Returns:
          type: 描述。
      """
  ```

### 📚 文档和可解释性
- **在添加新功能、依赖项更改或修改设置步骤时更新 `README.md`**。
- **注释非显而易见的代码**，并确保一切对中级开发人员都是可以理解的。
- 编写复杂逻辑时，**添加内联 `# 原因：` 注释**，解释为什么，而不仅仅是什么。

### 🧠 AI 行为规则
- **永远不要假设缺失的上下文。如果不确定，请提出问题。**
- **永远不要臆造库或函数** – 仅使用已知的、经过验证的 Python 包。
- **在代码或测试中引用文件路径和模块名称之前，始终确认它们存在**。
- **永远不要删除或覆盖现有代码**，除非明确指示或作为 `TASK.md` 中任务的一部分。

### 重要指令提醒
做被要求的事情；不多不少。
除非对实现目标绝对必要，否则永远不要创建文件。
始终优先编辑现有文件而不是创建新文件。
永远不要主动创建文档文件（*.md）或 README 文件。只有在用户明确请求时才创建文档文件。

## 项目概述

这是一个实现Deep Ritz方法的科学计算项目，用于求解偏微分方程。项目包含两个主要部分：
1. **原始DeepRitz实现**（根目录文件）- 包含基础的Deep Ritz方法实现
2. **改进的RKDR方法**（根目录文件）- 包含Radial Kernel Distribution Regression方法的实现

## 核心架构

项目已成功从单文件实现向模块化架构迁移：

### 当前状态
- **根目录包含原始实现**：`RD-2d.py`（标准DeepRitz方法）和`RKDR-2d.py`（改进的RKDR方法）
- **模块化架构已完成**：`core/`、`scripts/`、`config/` 目录已实现完整功能
- **输出目录**：`output/` 用于存储训练结果和模型文件
- **图表目录**：`figures/` 用于存储可视化结果

### 核心模块功能
- **`core/pdes/`**：PDE问题定义，包含基类`BasePDE`和具体实现`Poisson2D`
- **`core/models/`**：神经网络模型，包含基类`BaseModel`和增强网络`EnhancedRitzNet`
- **`core/loss/`**：损失函数，分离为蒙特卡洛版本和配点法版本
- **`core/trainer/`**：训练器，包含三种训练方法（DeepRitz、PINN、Collocation）
- **`core/data_utils/`**：数据采样工具，包含`Utils`类用于采样和误差计算
- **`scripts/`**：训练、评估和可视化脚本入口

### 技术栈
- **深度学习框架**：PyTorch
- **数值计算**：NumPy
- **可视化**：Matplotlib
- **主要依赖**：torch, numpy, matplotlib

### 核心算法
1. **Deep Ritz方法**：将PDE问题转化为变分形式，使用神经网络逼近解
2. **RKDR方法**：Radial Kernel Distribution Regression，使用径向核函数改进的Deep Ritz方法
3. **PINN方法**：Physics-Informed Neural Networks，基于PDE残差的训练方法
4. **配点法**：使用固定配点和数值积分的训练方法

## 开发命令

### 虚拟环境设置
```bash
# 创建虚拟环境
python -m venv venv_linux

# 激活虚拟环境 (Linux/WSL)
source venv_linux/bin/activate

# 激活虚拟环境 (Windows)
venv_linux\Scripts\activate

# 安装基本依赖
pip install torch numpy matplotlib
```

### 运行现有代码
```bash
# 运行标准DeepRitz方法（使用蒙特卡洛积分）
python scripts/train.py

# 运行改进的RKDR方法（配点法）
python scripts/train_coll.py

# 运行PINN方法
python scripts/train_pinn.py

# 评估已训练模型
python scripts/evaluate.py

# 可视化结果
python scripts/visualize.py

# 原始单文件实现（保留作为参考）
python RD-2d.py
python RKDR-2d.py
```

### 测试
```bash
# 手动测试：直接运行各个脚本文件
# 核心模块测试：单独测试 core/ 目录下的模块
# 配置测试：验证 config/config_loader.py 的配置加载功能
```

### 训练脚本说明
- **`scripts/train.py`** - 标准DeepRitz方法，使用蒙特卡洛积分
- **`scripts/train_coll.py`** - RKDR方法，使用配点法数值积分
- **`scripts/train_pinn.py`** - PINN方法，基于物理信息神经网络
- **`scripts/evaluate.py`** - 评估已训练模型的性能
- **`scripts/visualize.py`** - 生成训练结果可视化图表

### ⚠️ 重要注意事项
1. **配置一致性**：所有脚本都依赖 `config/` 目录下的配置文件，确保配置文件存在且正确
2. **训练器选择**：每种训练方法使用不同的训练器类，导入路径已在2025-10-26修复
3. **损失函数**：蒙特卡洛积分和配点法使用不同的损失函数版本，函数名已明确区分
4. **参数传递**：模型和训练器使用相同的参数配置，确保训练和评估时参数一致
5. **输出文件**：不同训练方法生成不同前缀的输出文件，避免冲突（如 `rkdr_mc_` vs `rkdr_coll_` vs `pinn_`）

