# DeepRitz单文件实现说明

## 文件信息

- **文件名**: `deepritz_standalone.py`
- **总行数**: 1047行
- **语言**: Python 3.7+
- **依赖**: PyTorch, NumPy, Matplotlib

## 核心特性

### 1. 完整的算法实现
- ✅ 数据采样（圆盘内部和边界）
- ✅ PDE问题定义（Poisson方程）
- ✅ 全连接残差神经网络（ResNet）
- ✅ 损失函数计算（能量泛函）
- ✅ 训练器（完整训练流程）
- ✅ 可视化工具（训练曲线和解对比）

### 2. 详细的数学注释
每个计算步骤都包含：
- 数学公式推导
- 蒙特卡洛积分原理
- 变分形式说明
- 中间结果解释

### 3. 相对误差计算（科研标准）

**重要改进**：所有误差计算使用相对误差，更适合科研汇报。

#### 相对L2误差
```
相对L2误差 = ||u - u_exact||_L² / ||u_exact||_L²
```

#### 相对H1误差
```
相对H1误差 = ||u - u_exact||_H¹ / ||u_exact||_H¹

其中 ||u||_H¹² = ||u||_L²² + ||∇u||_L²²
```

**优点**：
- 不受解的量级影响
- 更直观地反映相对精度
- 符合科研论文的标准报告方式
- 便于不同问题之间的比较

### 4. 网络结构

采用全连接残差网络（ResNet）：
```
Input(2D) → Linear(width) → Tanh →
[ResBlock × (depth-1)] →
Linear(1) → Output
```

**残差块结构**：
```
x → Linear → Tanh → Linear → (+) → output
                              ↑
                              x (skip connection)
```

## 使用方法

### 基本运行
```bash
python deepritz_standalone.py
```

### 自定义配置
修改`main()`函数中的`config`字典：
```python
config = {
    'radius': 1.0,           # 域半径
    'width': 100,            # 网络宽度
    'depth': 3,              # 网络深度
    'train_steps': 5000,     # 训练步数
    'lr': 0.001,             # 学习率
    'body_batch': 4096,      # 内部采样点数
    'boundary_batch': 4096,  # 边界采样点数
    'penalty': 1000,         # 边界惩罚系数
    # ... 更多参数
}
```

### 作为模块导入
```python
from deepritz_standalone import (
    RitzNet,           # 神经网络模型
    Poisson2D,         # PDE问题定义
    DeepRitzTrainer,   # 训练器
    DataSampler,       # 数据采样工具
    LossComputer,      # 损失计算器
    Visualizer         # 可视化工具
)

# 创建自定义问题
pde = Poisson2D(radius=1.0)
model = RitzNet(width=100, depth=3)
trainer = DeepRitzTrainer(model, pde, device='cuda', config=config)
```

## 输出结果

### 1. 控制台输出
```
Step 0/5000 (耗时: 2.3s)
  损失: 0.123456
  相对L2误差: 0.045678
  相对H1误差: 0.067890
  学习率: 0.001000

...

训练结果总结
================================================================================
最终损失: 0.001234
最终相对L2误差: 0.002345 (0.23%)
最终相对H1误差: 0.003456 (0.35%)
```

### 2. 生成的图表

#### training_history.png
包含3个子图：
- 训练损失曲线（对数坐标）
- 相对L2误差曲线（对数坐标）
- 相对H1误差曲线（对数坐标）

#### solution_comparison.png
包含4个子图：
- 网络解 u_θ(x,y)
- 精确解 u_exact(x,y)
- 绝对误差分布 |u_θ - u_exact|
- 误差分布直方图（含统计信息）

## 算法流程

```
1. 初始化
   ├─ 创建PDE问题
   ├─ 创建神经网络
   └─ 设置优化器

2. 训练循环 (每个step)
   ├─ 采样训练数据
   │  ├─ 内部点采样（蒙特卡洛）
   │  └─ 边界点采样（均匀分布）
   │
   ├─ 前向传播
   │  ├─ 计算网络输出 u_θ(x)
   │  └─ 自动微分计算梯度 ∇u_θ
   │
   ├─ 损失计算
   │  ├─ 能量损失：∫[1/2|∇u|² - fu]dx
   │  ├─ 边界损失：λ∫|u-g|²ds
   │  └─ 总损失 = 能量损失 + 边界损失
   │
   ├─ 反向传播
   │  └─ 更新网络参数
   │
   └─ 周期性评估
      ├─ 计算相对L2误差
      ├─ 计算相对H1误差
      └─ 打印训练进度

3. 结果可视化
   ├─ 训练历史图
   └─ 解对比图
```

## 数学原理

### 变分形式
Poisson方程的变分形式：
```
最小化能量泛函：
E(u) = ∫_Ω [1/2|∇u|² - fu] dx + λ∫_∂Ω |u-g|² ds
```

### 蒙特卡洛积分
```
∫_Ω f(x) dx ≈ |Ω| · (1/N) Σᵢ f(xᵢ)

其中：
- |Ω| = πR² (圆盘面积)
- xᵢ ~ Uniform(Ω) (均匀采样)
```

### 误差度量

#### 相对L2误差
```
Relative L2 Error = ||u - u_exact||_L² / ||u_exact||_L²
                  = sqrt(∫|u - u_exact|²dx) / sqrt(∫|u_exact|²dx)
```

#### 相对H1误差
```
Relative H1 Error = ||u - u_exact||_H¹ / ||u_exact||_H¹

其中 H¹范数：
||u||_H¹ = sqrt(||u||_L²² + ||∇u||_L²²)
```

## 测试问题

**精确解**：
```
u(x,y) = sin(πx)sin(πy)
```

**源项**（由-Δu计算）：
```
f(x,y) = 2π²sin(πx)sin(πy)
```

**边界条件**：
```
u = g(x,y) on ∂Ω
```

## 性能建议

### GPU加速
```python
# 自动检测并使用GPU
config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
```

### 训练参数调优
- **学习率**：初始0.001，每500步衰减0.5
- **批大小**：内部4096点，边界4096点
- **边界惩罚**：λ=1000（根据问题调整）
- **重采样频率**：每200步重新采样

### 收敛标准
- 相对L2误差 < 1% (0.01)
- 相对H1误差 < 2% (0.02)
- 损失函数稳定下降

## 常见问题

### Q1: 如何切换到绝对误差？
```python
# 在compute_l2_error调用时设置relative=False
l2_error = DataSampler.compute_l2_error(
    output, target, radius, relative=False
)
```

### Q2: 如何修改网络深度？
```python
config['depth'] = 5  # 增加到5层
```

### Q3: 如何处理其他PDE问题？
继承`Poisson2D`类并重写：
- `source_term()` - 源项
- `exact_solution()` - 精确解
- `boundary_condition()` - 边界条件

## 引用

如果使用本代码，请引用：
```
DeepRitz项目组 (2026). Deep Ritz方法求解Poisson方程 - 单文件实现.
```

## 许可证

本代码仅供学习和研究使用。

---

**最后更新**: 2026-01-02
**版本**: 1.1 (添加相对误差计算)
