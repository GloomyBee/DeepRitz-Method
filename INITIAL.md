## 功能：
- 基于根目录的RKDR-2d.py以及RD-2d.py重构项目，已完成规范的模块化项目结构	
DeepRitz/
├── config/                  # 配置文件，用于管理实验的超参数
│   ├── base_config.yaml      # (S) 基础配置文件，包含通用参数
│   ├── poisson_2d.yaml      # (S) 泊松方程2D问题特定配置
│   └── config_loader.py     # (S) 配置加载和管理模块
│
├── core/                   # 核心源代码包
│   ├── __init__.py          # (S) 包初始化文件
│   ├── pdes/               # ★ PDE (物理问题) 定义模块
│   │   ├── __init__.py
│   │   ├── base_pde.py     # (S,O) 定义所有PDE的抽象基类接口
│   │   └── poisson.py      # (S) 泊松方程的具体实现
│   │
│   ├── models/             # (S) 神经网络模型定义
│   │   ├── __init__.py
│   │   ├── base_model.py   # (L) 定义模型的基础接口
│   │   └── mlp.py        # (S) 增强的Ritz网络实现（含高斯核特征）
│   │
│   ├── data_utils/        # (S, DRY) 数据处理和采样工具
│   │   ├── __init__.py
│   │   └── sampler.py     # (S) 采样工具函数和计算工具
│   │
│   ├── losses.py          # (S, DRY) 损失函数定义 (能量泛函、边界条件损失)
│   ├── trainer.py         # (S) 核心训练逻辑 (Trainer类)
│   └── utils.py           # (DRY) 通用工具函数 (Matplotlib设置、目录管理等)
│
├── scripts/               # 可执行脚本
│   ├── train.py          # (S) RKDR训练入口脚本
│   ├── evaluate.py       # (S) 评估已训练模型的入口脚本
│   └── visualize.py     # (S) 对结果进行可视化的脚本
│
├── tests/                # 单元测试和集成测试 (目录已创建，待填充)
│   ├── test_models.py    # (O) 模型测试
│   ├── test_pdes.py     # (O) PDE测试
│   └── test_sampler.py  # (O) 采样器测试
│
├── output/              # 训练输出目录 (自动创建)
│   ├── *.pt            # 训练的模型文件
│   ├── *.txt           # 训练历史和误差数据
│   └── *.png           # 生成的图表文件
│
├── figures/             # 可视化图表输出目录 (自动创建)
│
├── .gitignore           # Git忽略文件配置
├── CLAUDE.md            # Claude Code项目指导文件
├── INITIAL.md           # 项目架构说明文档
├── TASK.md             # 任务管理文档
├── README.md           # 项目说明文档
├── RD-2d.py           # (L) 原始DeepRitz方法实现（保留）
└── RKDR-2d.py         # (L) 原始RKDR方法实现（保留）


## 其他考虑：

### 设计原则 (S = 单一职责, O = 开闭原则, DRY = 不重复原则)
- **单一职责 (S)**: 每个模块只负责一个明确的功能，如PDE定义、模型定义、数据处理等
- **开闭原则 (O)**: 通过抽象基类设计，支持扩展新的PDE类型和模型架构，无需修改现有代码
- **不重复原则 (DRY)**: 通用功能抽取为独立模块，避免代码重复

### 架构优势
1. **模块化**: 代码按功能清晰分离，便于维护和扩展
2. **可配置**: 使用YAML配置文件管理超参数，支持不同实验配置
3. **可扩展**: 基于抽象基类的设计，便于添加新的PDE问题、模型架构和损失函数
4. **可重用**: 核心组件可独立使用和测试
5. **标准化**: 遵循Python项目标准结构，便于团队协作

### 使用方式
```bash
# 训练RKDR模型
python scripts/train.py

# 评估训练好的模型
python scripts/evaluate.py

# 可视化结果
python scripts/visualize.py
```

### 兼容性
- 保留了原始的RKDR-2d.py和RD-2d.py文件，确保向后兼容
- 新框架与原始实现功能完全一致，没有添加任何多余功能
- 支持无缝迁移到模块化架构
