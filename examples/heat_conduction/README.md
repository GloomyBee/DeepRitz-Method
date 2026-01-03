# 热传导问题示例

使用DeepRitz框架求解二维稳态热传导方程。

## 问题描述

### 控制方程
稳态热传导方程：
```
-k∇²T = s  in Ω
```

其中：
- `k = 20.0 W/(m·K)` - 导热系数
- `s = 50.0 W/m³` - 体积热源
- `T` - 温度场

### 几何区域
梯形区域 Ω，顶点坐标：
- (0, 0) - 左下角
- (2, 0) - 右下角
- (1, 1) - 右上角
- (0, 1) - 左上角

面积：1.5 m²

### 边界条件

| 边界 | 类型 | 条件 |
|------|------|------|
| 左边 (x=0) | Dirichlet | T = 20°C |
| 上边 (y=1, 0≤x≤1) | Neumann | q = -100 W/m² |
| 底边 (y=0) | 绝热 | q = 0 |
| 斜边 (x+y=2) | 绝热 | q = 0 |

## 文件结构

```
examples/heat_conduction/
├── heat_pde.py              # HeatPDE类定义（继承BasePDE）
├── trapezoid_sampler.py     # 梯形区域采样器
├── heat_trainer.py          # 热传导训练器（继承BaseTrainer）
├── heat_config.yaml         # 配置文件
├── train_heat.py            # 训练脚本
├── visualize_heat.py        # 可视化脚本
└── README.md                # 本文档
```

## 快速开始

### 1. 测试PDE接口

```bash
cd examples/heat_conduction
python heat_pde.py
```

### 2. 测试采样器

```bash
python trapezoid_sampler_hr.py
```

这将生成 `trapezoid_sampling_test.png` 展示采样效果。

### 3. 训练模型

```bash
python train_heat_hr.py
```

训练过程：
- 使用Adam优化器
- 学习率：0.001，每500步衰减0.5倍
- 训练5000步（约2-5分钟，取决于硬件）
- 模型保存到 `../../output/heat_model.pt`

### 4. 可视化结果

```bash
python visualize_heat.py
```

生成的图表包括：
1. 温度分布云图
2. 训练损失曲线
3. 温度等值线图
4. 温度统计直方图
5. x方向温度剖面
6. y方向温度剖面

结果保存到 `../../output/heat_results.png`

## 配置参数

编辑 `heat_config.yaml` 调整参数：

### 物理参数
```yaml
problem:
  k: 20.0          # 导热系数
  s: 50.0          # 体积热源
  T_left: 20.0     # 左边界温度
  q_top: -100.0    # 上边界热流
```

### 网络参数
```yaml
network:
  hidden_width: 50   # 隐藏层宽度
  depth: 4           # 残差块数量
```

### 训练参数
```yaml
training:
  train_steps: 5000  # 训练步数
  lr: 0.001          # 学习率
  bodyBatch: 5000    # 内部点批大小
  bdryBatch: 1000    # 边界点批大小
```

## 框架使用示范

本示例展示了如何正确使用DeepRitz框架：

### ✅ 继承框架基类

```python
# heat_pde.py
from core.pdes.base_pde import BasePDE

class HeatPDE(BasePDE):
    def source_term(self, data):
        # 实现源项
        pass

    def exact_solution(self, data):
        # 无解析解返回None
        return None

    def boundary_condition(self, data):
        # 实现边界条件
        pass
```

### ✅ 使用框架模型

```python
# train_heat_hr.py
from core.models.mlp import EnhancedRitzNet

model = EnhancedRitzNet(
    input_dim=2,
    output_dim=1,
    width=50,
    depth=4,
    pde=pde
)
```

### ✅ 继承训练器基类

```python
# heat_trainer_hr.py
from core.trainer.base_trainer import BaseTrainer

class HeatTrainer(BaseTrainer):
    def _compute_loss(self, ...):
        # 实现损失计算
        pass

    def train(self):
        # 实现训练循环
        pass
```

### ✅ 使用YAML配置

```python
# train_heat_hr.py
import yaml

with open('heat_config.yaml', 'r') as f:
    config = yaml.safe_load(f)
```

## 扩展建议

### 添加新的边界条件

在 `heat_pde.py` 中添加边界判断方法：

```python
def is_on_custom_boundary(self, data, tol=1e-6):
    # 实现自定义边界判断
    pass
```

在 `heat_trainer.py` 中添加对应的损失项：

```python
def _compute_loss(self, ...):
    # 添加新的边界损失
    custom_bc_loss = ...
    total_loss += custom_bc_loss
```

### 更换几何区域

修改 `trapezoid_sampler.py` 中的采样逻辑：

```python
@staticmethod
def sample_domain(n_samples):
    # 实现新的区域采样
    pass
```

### 添加时间依赖

扩展为瞬态热传导问题：
1. 修改 `HeatPDE` 添加时间维度
2. 修改采样器同时采样时空点
3. 修改损失函数包含时间导数项

## 性能优化

### GPU加速

在 `heat_config.yaml` 中设置：
```yaml
device: "cuda"
```

### 调整批大小

根据GPU内存调整：
```yaml
sampling:
  bodyBatch: 10000   # 增大以充分利用GPU
  bdryBatch: 2000
```

### 学习率调度

修改学习率衰减策略：
```yaml
training:
  step_size: 1000    # 更晚衰减
  gamma: 0.8         # 更慢衰减
```

## 常见问题

### Q: 训练不收敛怎么办？

A: 尝试以下方法：
1. 增大边界惩罚系数 `penalty`
2. 减小学习率 `lr`
3. 增加网络容量 `hidden_width` 或 `depth`
4. 增加采样点数 `bodyBatch`

### Q: 如何验证结果正确性？

A:
1. 检查边界条件是否满足（左边界应接近20°C）
2. 检查PDE残差（运行训练脚本会自动评估）
3. 检查物理合理性（温度分布是否符合预期）

### Q: 如何处理无解析解的问题？

A: 本示例展示了处理方法：
1. `exact_solution()` 返回 `None`
2. 使用PDE残差评估收敛性
3. 通过物理约束验证结果

## 参考文献

1. E, W., & Yu, B. (2018). The deep Ritz method: a deep learning-based numerical algorithm for solving variational problems. *Communications in Mathematics and Statistics*, 6(1), 1-12.

2. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686-707.

## 许可证

本示例遵循DeepRitz框架的许可证。
