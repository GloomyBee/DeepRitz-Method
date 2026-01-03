# Deep Ritz MATLAB迁移项目 - 使用指南

## 目录

1. [项目简介](#项目简介)
2. [系统要求](#系统要求)
3. [安装配置](#安装配置)
4. [快速开始](#快速开始)
5. [使用MATLAB后端](#使用matlab后端)
6. [使用Python后端](#使用python后端)
7. [配置选项](#配置选项)
8. [API参考](#api参考)
9. [常见问题](#常见问题)
10. [故障排除](#故障排除)

---

## 项目简介

本项目实现了Deep Ritz方法的Python-MATLAB混合架构,结合了两种语言的优势:
- **Python/PyTorch**: 神经网络训练、GPU加速、自动微分
- **MATLAB**: 数值计算、PDE定义、矩阵运算

### 核心特性

✅ **双后端支持** - 可选择Python或MATLAB进行数值计算
✅ **自动回退** - MATLAB失败时自动切换到Python
✅ **配置驱动** - 通过YAML文件控制所有参数
✅ **完整测试** - 单元测试和集成测试覆盖
✅ **详细日志** - 完整的训练过程记录

---

## 系统要求

### 必需组件

- **Python**: 3.7+
- **PyTorch**: 1.8+
- **NumPy**: 1.19+
- **MATLAB**: R2019b+ (仅使用MATLAB后端时需要)

### 可选组件

- **CUDA**: 用于GPU加速(推荐)
- **MATLAB Engine API for Python**: 用于Python-MATLAB通信

---

## 安装配置

### 1. 克隆项目

```bash
git clone <repository-url>
cd DeepRitz
```

### 2. 创建虚拟环境

```bash
# Linux/Mac
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. 安装Python依赖

```bash
pip install torch numpy matplotlib pyyaml
```

### 4. 安装MATLAB Engine API (可选)

如果要使用MATLAB后端,需要安装MATLAB Engine API for Python:

```bash
# 1. 找到MATLAB安装目录
# 在MATLAB中运行: matlabroot

# 2. 进入Python引擎目录
cd "matlabroot/extern/engines/python"

# 3. 安装
python setup.py install
```

**验证安装**:

```python
import matlab.engine
print("MATLAB Engine API安装成功!")
```

### 5. 配置MATLAB路径

编辑 `config/base_config.yaml`:

```yaml
backend:
  type: "matlab"  # 或 "python"
  matlab_path: null  # 如果MATLAB不在PATH中,指定完整路径
  matlab_startup_timeout: 30
  fallback_to_python: true
```

---

## 快速开始

### 使用Python后端(默认)

```bash
# 1. 确保配置使用Python后端
# config/base_config.yaml:
#   backend:
#     type: "python"

# 2. 运行训练
python scripts/train.py

# 3. 查看结果
python scripts/visualize.py
```

### 使用MATLAB后端

```bash
# 1. 修改配置使用MATLAB后端
# config/base_config.yaml:
#   backend:
#     type: "matlab"

# 2. 运行训练
python scripts/train.py

# 3. 查看结果
python scripts/visualize.py
```

---

## 使用MATLAB后端

### 方法1: 通过配置文件

**步骤1**: 编辑 `config/base_config.yaml`

```yaml
backend:
  type: "matlab"
  matlab_path: null  # 或指定MATLAB路径,如 "C:/Program Files/MATLAB/R2023a"
  matlab_startup_timeout: 30
  fallback_to_python: true  # 推荐开启
```

**步骤2**: 运行训练脚本

```bash
python scripts/train.py
```

### 方法2: 通过代码

```python
from core.trainer.trainer_matlab import MatlabTrainer
from core.models.mlp import EnhancedRitzNet
from core.pdes.poisson import Poisson2D
from config.config_loader import Config

# 加载配置
config = Config('config/base_config.yaml')
params = config.config

# 创建模型
pde = Poisson2D(radius=params['radius'])
model = EnhancedRitzNet(
    d=params['d'],
    m=params['width'],
    depth=params['depth'],
    pde=pde
)

# 创建MATLAB训练器
trainer = MatlabTrainer(
    model=model,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    params=params,
    use_matlab=True  # 使用MATLAB后端
)

# 训练
steps, l2_errors, h1_errors = trainer.train()
```

### 手动控制MATLAB引擎

```python
from core.interface.matlab_engine_manager import MatlabEngineManager

# 方法1: 使用上下文管理器(推荐)
with MatlabEngineManager() as engine:
    # 调用MATLAB函数
    result = engine.call_function('Sampler.sample_from_disk', 1.0, 100)
    print(f"采样点数: {len(result)}")

# 方法2: 手动管理
engine = MatlabEngineManager()
try:
    engine.start_engine()
    result = engine.call_function('Poisson2D(1.0).source_term', points)
finally:
    engine.stop_engine()
```

### 数据转换示例

```python
from core.interface.data_converter import DataConverter
import torch
import numpy as np

# PyTorch → MATLAB
tensor = torch.randn(100, 2)
matlab_array = DataConverter.torch_to_matlab(tensor)

# NumPy → MATLAB
numpy_array = np.random.randn(100, 2)
matlab_array = DataConverter.numpy_to_matlab(numpy_array)

# MATLAB → PyTorch
tensor_back = DataConverter.matlab_to_torch(matlab_array, device='cuda')

# MATLAB → NumPy
numpy_back = DataConverter.matlab_to_numpy(matlab_array)
```

---

## 使用Python后端

### 标准训练流程

```python
from core.trainer.trainer import Trainer
from core.models.mlp import EnhancedRitzNet
from core.pdes.poisson import Poisson2D
from config.config_loader import Config

# 加载配置
config = Config('config/base_config.yaml')
params = config.config

# 创建模型
pde = Poisson2D(radius=params['radius'])
model = EnhancedRitzNet(
    d=params['d'],
    m=params['width'],
    depth=params['depth'],
    pde=pde
)

# 创建训练器
trainer = Trainer(
    model=model,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    params=params
)

# 训练
steps, l2_errors, h1_errors = trainer.train()

# 保存模型
torch.save(model.state_dict(), 'output/model.pt')
```

---

## 配置选项

### 完整配置文件示例

```yaml
# config/base_config.yaml

# 基础参数
window_size: 200
tolerance: 5e-2
k: 1
d: 2                    # 输入维度
dd: 1                   # 输出维度
radius: 1.0            # 域半径

# 批处理参数
bodyBatch: 4096        # 域内批大小
bdryBatch: 4096        # 边界批大小
numQuad: 40000         # 积分点数量

# 网络参数
width: 100             # 网络宽度
depth: 3               # 网络深度

# 训练参数
trainStep: 20000       # 训练步数
lr: 0.001             # 学习率
decay: 0.0001         # 权重衰减
penalty: 1000         # 边界惩罚系数
step_size: 200        # 学习率调度步数
gamma: 0.5            # 学习率衰减因子

# 采样参数
writeStep: 10         # 写入步数
sampleStep: 200       # 重新采样步数
recordStep: 100       # 记录步数

# 设备参数
device: "cuda"        # 设备选择 (auto, cpu, cuda)

# 输出参数
output_dir: "output"
figures_dir: "figures"

# 后端配置
backend:
  type: "python"           # "python" 或 "matlab"
  matlab_path: null        # MATLAB路径(可选)
  matlab_startup_timeout: 30
  fallback_to_python: true
```

### 配置参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `backend.type` | string | "python" | 后端类型: "python" 或 "matlab" |
| `backend.matlab_path` | string | null | MATLAB安装路径,null表示使用系统PATH |
| `backend.matlab_startup_timeout` | int | 30 | MATLAB引擎启动超时(秒) |
| `backend.fallback_to_python` | bool | true | MATLAB失败时是否回退到Python |
| `trainStep` | int | 20000 | 总训练步数 |
| `bodyBatch` | int | 4096 | 内部点批大小 |
| `bdryBatch` | int | 4096 | 边界点批大小 |
| `penalty` | float | 1000 | 边界条件惩罚系数 |
| `lr` | float | 0.001 | 学习率 |
| `device` | string | "cuda" | 计算设备 |

---

## API参考

### MatlabEngineManager

```python
class MatlabEngineManager:
    """MATLAB引擎管理器"""

    def __init__(self, matlab_path: Optional[str] = None,
                 startup_timeout: int = 30):
        """初始化引擎管理器"""

    def start_engine(self) -> None:
        """启动MATLAB引擎"""

    def stop_engine(self) -> None:
        """停止MATLAB引擎"""

    def call_function(self, func_name: str, *args, **kwargs) -> Any:
        """调用MATLAB函数"""

    def is_running(self) -> bool:
        """检查引擎是否运行"""
```

### DataConverter

```python
class DataConverter:
    """数据格式转换器"""

    @staticmethod
    def torch_to_matlab(tensor: torch.Tensor):
        """PyTorch张量 → MATLAB数组"""

    @staticmethod
    def numpy_to_matlab(array: np.ndarray):
        """NumPy数组 → MATLAB数组"""

    @staticmethod
    def matlab_to_torch(data, device: str = 'cpu') -> torch.Tensor:
        """MATLAB数组 → PyTorch张量"""

    @staticmethod
    def matlab_to_numpy(data) -> np.ndarray:
        """MATLAB数组 → NumPy数组"""
```

### MatlabTrainer

```python
class MatlabTrainer(BaseTrainer):
    """支持MATLAB后端的训练器"""

    def __init__(self, model, device: str, params: dict,
                 use_matlab: bool = True):
        """初始化训练器"""

    def train(self) -> Tuple[List[int], List[float], List[float]]:
        """训练模型,返回(steps, l2_errors, h1_errors)"""
```

---

## 常见问题

### Q1: 如何选择使用Python还是MATLAB后端?

**建议**:
- **开发和调试**: 使用Python后端(更快,更容易调试)
- **生产和性能**: 使用MATLAB后端(数值计算可能更快)
- **没有MATLAB**: 使用Python后端(完全功能)

### Q2: MATLAB引擎启动很慢怎么办?

**解决方案**:
1. 增加启动超时时间:
   ```yaml
   backend:
     matlab_startup_timeout: 60  # 增加到60秒
   ```

2. 使用持久化引擎(训练期间保持运行)

3. 启用回退机制:
   ```yaml
   backend:
     fallback_to_python: true
   ```

### Q3: 如何验证MATLAB后端是否正常工作?

```python
# 测试脚本
from core.interface.matlab_engine_manager import MatlabEngineManager

try:
    with MatlabEngineManager() as engine:
        result = engine.call_function('Sampler.sample_from_disk', 1.0, 10)
        print("✅ MATLAB后端工作正常!")
        print(f"采样点数: {len(result)}")
except Exception as e:
    print(f"❌ MATLAB后端失败: {e}")
```

### Q4: 训练时如何知道使用的是哪个后端?

查看训练日志:
```
INFO - MATLAB后端已启用
INFO - 开始训练(后端: MATLAB)
```

或检查输出文件名:
- MATLAB后端: `matlab_mc_loss_history.txt`
- Python后端: `python_mc_loss_history.txt`

### Q5: 如何在训练中途切换后端?

不建议在训练中途切换。如需切换:
1. 停止当前训练
2. 修改配置文件
3. 重新开始训练

### Q6: MATLAB和Python的计算结果一致吗?

理论上应该一致(相对误差<1e-6)。如果发现差异:
1. 检查MATLAB版本(需要R2019b+)
2. 运行数值一致性测试
3. 查看日志中的警告信息

---

## 故障排除

### 问题1: ImportError: No module named 'matlab.engine'

**原因**: MATLAB Engine API未安装

**解决**:
```bash
cd "matlabroot/extern/engines/python"
python setup.py install
```

### 问题2: MATLAB引擎启动失败

**可能原因**:
1. MATLAB未安装或不在PATH中
2. MATLAB许可证问题
3. 启动超时

**解决方案**:
```yaml
# 1. 指定MATLAB路径
backend:
  matlab_path: "C:/Program Files/MATLAB/R2023a"

# 2. 增加超时时间
backend:
  matlab_startup_timeout: 60

# 3. 启用回退
backend:
  fallback_to_python: true
```

### 问题3: 数据转换错误

**错误信息**: `ValueError: 输入维度不匹配`

**解决**:
```python
# 确保数据形状正确
# 内部点: [N, 2]
# 输出: [N, 1]
# 梯度: [N, 2]

# 使用ensure_2d确保维度
from core.interface.data_converter import DataConverter
data = DataConverter.ensure_2d(data)
```

### 问题4: CUDA out of memory

**解决**:
```yaml
# 减小批大小
bodyBatch: 2048  # 从4096减小
bdryBatch: 2048

# 或使用CPU
device: "cpu"
```

### 问题5: 训练不收敛

**检查清单**:
- [ ] 学习率是否合适? (尝试0.001-0.01)
- [ ] 惩罚系数是否合适? (尝试100-10000)
- [ ] 网络是否足够大? (尝试增加width或depth)
- [ ] 采样点是否足够? (尝试增加bodyBatch)

---

## 性能优化建议

### 1. 使用GPU加速

```yaml
device: "cuda"  # 而不是 "cpu"
```

### 2. 调整批大小

```yaml
# 根据GPU内存调整
bodyBatch: 8192  # GPU内存充足时增大
bdryBatch: 8192
```

### 3. 减少重采样频率

```yaml
sampleStep: 500  # 从200增加到500
```

### 4. 使用混合精度训练

```python
# 在训练器中启用
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    loss = trainer._compute_loss(data_body, data_boundary)
```

---

## 示例代码

### 完整训练示例

```python
#!/usr/bin/env python
"""完整的训练示例"""

import torch
from core.trainer.trainer_matlab import MatlabTrainer
from core.models.mlp import EnhancedRitzNet
from core.pdes.poisson import Poisson2D
from config.config_loader import Config

def main():
    # 1. 加载配置
    config = Config('config/base_config.yaml')
    params = config.config

    # 2. 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 3. 创建PDE和模型
    pde = Poisson2D(radius=params['radius'])
    model = EnhancedRitzNet(
        d=params['d'],
        m=params['width'],
        depth=params['depth'],
        pde=pde
    ).to(device)

    # 4. 创建训练器
    use_matlab = params['backend']['type'] == 'matlab'
    trainer = MatlabTrainer(
        model=model,
        device=device,
        params=params,
        use_matlab=use_matlab
    )

    # 5. 训练
    print("开始训练...")
    steps, l2_errors, h1_errors = trainer.train()

    # 6. 保存模型
    torch.save(model.state_dict(), 'output/model_final.pt')
    print("训练完成!")

    # 7. 打印最终误差
    print(f"最终L2误差: {l2_errors[-1]:.6e}")
    print(f"最终H1误差: {h1_errors[-1]:.6e}")

if __name__ == '__main__':
    main()
```

### 自定义PDE示例

```python
"""自定义PDE示例"""

import torch
from core.pdes.base_pde import BasePDE

class CustomPDE(BasePDE):
    """自定义PDE: -Δu = sin(πx)sin(πy)"""

    def __init__(self, radius: float = 1.0):
        self.radius = radius

    def source_term(self, data: torch.Tensor) -> torch.Tensor:
        """源项"""
        x, y = data[:, 0:1], data[:, 1:2]
        return torch.sin(torch.pi * x) * torch.sin(torch.pi * y)

    def exact_solution(self, data: torch.Tensor) -> torch.Tensor:
        """解析解(如果已知)"""
        x, y = data[:, 0:1], data[:, 1:2]
        return torch.sin(torch.pi * x) * torch.sin(torch.pi * y) / (2 * torch.pi**2)

    def boundary_condition(self, data: torch.Tensor) -> torch.Tensor:
        """边界条件"""
        return torch.zeros(data.shape[0], 1)
```

---

## 更多资源

- **项目文档**: `MATLAB_MIGRATION_FINAL_SUMMARY.md`
- **设计文档**: `.spec-workflow/specs/matlab-migration/design.md`
- **API文档**: 查看各模块的docstring
- **测试示例**: `tests/` 目录

---

## 联系和支持

如有问题或建议,请:
1. 查看本使用指南
2. 查看常见问题部分
3. 查看故障排除部分
4. 提交Issue到项目仓库

---

*最后更新: 2025-12-18*
*版本: 1.0*
