# 绘图功能使用说明

本项目为两个算例提供了完整的可视化功能：
1. **Poisson方程**（deepritz_standalone.py）
2. **热传导问题**（heat.py）

---

## 一、Poisson方程可视化

### 方法1：运行主文件（推荐）

```bash
python deepritz_standalone.py
```

**自动生成的图表**：
- `training_history.png` - 训练历史（损失、L2误差、H1误差）
- `solution_comparison.png` - 解对比（网络解、精确解、误差分布、误差直方图）

### 方法2：使用独立绘图脚本

```bash
python plot_deepritz.py
```

**额外生成的图表**：
- `deepritz_comprehensive.png` - 综合结果图（6个子图）
  - 训练损失曲线
  - 相对L2误差曲线
  - 相对H1误差曲线
  - 网络解云图
  - 精确解云图
  - 误差分布云图

- `deepritz_slices.png` - 切片对比图
  - 沿x轴切片（y=0）
  - 沿y轴切片（x=0）

- `deepritz_comparison.png` - 解对比图（同主文件）

### 图表说明

#### 1. 训练历史图 (training_history.png)
```
┌─────────────────┬─────────────────┬─────────────────┐
│  训练损失曲线    │  相对L2误差曲线  │  相对H1误差曲线  │
│  (对数坐标)     │  (对数坐标)     │  (对数坐标)     │
└─────────────────┴─────────────────┴─────────────────┘
```

**用途**：
- 评估训练收敛情况
- 查看误差下降趋势
- 确定是否需要更多训练步数

#### 2. 解对比图 (solution_comparison.png)
```
┌─────────────────┬─────────────────┐
│  网络解 u_θ     │  精确解 u_exact │
├─────────────────┼─────────────────┤
│  绝对误差分布    │  误差直方图     │
└─────────────────┴─────────────────┘
```

**用途**：
- 直观对比网络解与精确解
- 查看误差的空间分布
- 分析误差的统计特性

#### 3. 综合结果图 (deepritz_comprehensive.png)
```
┌─────────────────┬─────────────────┐
│  训练损失       │  相对L2误差     │
├─────────────────┼─────────────────┤
│  相对H1误差     │  网络解云图     │
├─────────────────┼─────────────────┤
│  精确解云图     │  误差分布云图   │
└─────────────────┴─────────────────┘
```

**用途**：
- 一张图展示所有关键信息
- 适合论文和报告使用
- 包含详细的误差统计

#### 4. 切片对比图 (deepritz_slices.png)
```
┌─────────────────┬─────────────────┐
│  沿x轴切片      │  沿y轴切片      │
│  (y=0)         │  (x=0)         │
└─────────────────┴─────────────────┘
```

**用途**：
- 精确对比网络解与精确解
- 查看特定方向的解分布
- 验证边界条件满足情况

---

## 二、热传导问题可视化

### 运行方式

```bash
python heat.py
```

**自动生成的图表**：
- `heat_results.png` - 完整结果图（4个子图）
- `heat_profiles.png` - 温度剖面图（2个子图）

### 图表说明

#### 1. 完整结果图 (heat_results.png)
```
┌─────────────────┬─────────────────┐
│  温度分布云图    │  训练损失曲线   │
│  (含边界条件)   │  (对数坐标)     │
├─────────────────┼─────────────────┤
│  温度等值线图    │  温度分布直方图 │
│  (含数值标注)   │  (含统计信息)   │
└─────────────────┴─────────────────┘
```

**特点**：
- **温度分布云图**：使用inferno色图，清晰显示温度梯度
- **边界条件标注**：
  - 左边界：T=20°C (Dirichlet)
  - 上边界：q=-100 W/m² (Neumann)
  - 底边和斜边：绝热 (q=0)
- **损失曲线**：显示训练收敛过程
- **等值线图**：带数值标注，便于读取温度值
- **统计信息**：最小值、最大值、平均值、标准差

#### 2. 温度剖面图 (heat_profiles.png)
```
┌─────────────────┬─────────────────┐
│  沿x轴剖面      │  沿y轴剖面      │
│  (y=0.5)       │  (x=0.5)       │
└─────────────────┴─────────────────┘
```

**用途**：
- 查看特定路径的温度变化
- 分析热传导规律
- 验证边界条件影响

---

## 三、图表定制

### 修改图表参数

#### Poisson方程（deepritz_standalone.py）

在`main()`函数中修改配置：

```python
config = {
    'train_steps': 5000,    # 训练步数
    'eval_step': 100,       # 评估间隔
    'width': 100,           # 网络宽度
    'depth': 3,             # 网络深度
    # ... 其他参数
}
```

#### 热传导问题（heat.py）

在`train()`函数中修改参数：

```python
n_steps = 5000          # 训练步数
batch_domain = 5000     # 域内采样点数
batch_bound = 1000      # 边界采样点数
beta = 500.0           # Dirichlet惩罚系数
```

### 修改图表样式

#### 颜色映射

```python
# Poisson方程
cmap='viridis'  # 可选: 'plasma', 'inferno', 'magma', 'cividis'

# 热传导问题
cmap='inferno'  # 可选: 'hot', 'coolwarm', 'RdYlBu'
```

#### 分辨率

```python
# 修改采样点数
x = np.linspace(-radius, radius, 100)  # 改为200提高分辨率
y = np.linspace(-radius, radius, 100)  # 改为200提高分辨率
```

#### 保存格式

```python
# 修改DPI和格式
plt.savefig('result.png', dpi=300, bbox_inches='tight')  # PNG格式
plt.savefig('result.pdf', bbox_inches='tight')           # PDF格式（矢量图）
```

---

## 四、常见问题

### Q1: 图表显示不完整？

**解决方案**：
```python
plt.tight_layout()  # 自动调整子图间距
```

### Q2: 中文显示乱码？

**解决方案**：
```python
# 在文件开头添加
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False     # 用来正常显示负号
```

### Q3: 内存不足？

**解决方案**：
```python
# 减少采样点数
x = np.linspace(-radius, radius, 50)  # 从100减少到50
y = np.linspace(-radius, radius, 50)
```

### Q4: 训练太慢？

**解决方案**：
```python
# 减少训练步数
config['train_steps'] = 2000  # 从5000减少到2000

# 或使用GPU
config['device'] = 'cuda'  # 需要CUDA支持
```

### Q5: 如何只生成特定图表？

**Poisson方程**：
```python
# 只生成训练历史图
Visualizer.plot_training_history(
    loss_history, l2_errors, h1_errors,
    eval_step=config['eval_step'],
    save_path='training_only.png'
)

# 只生成解对比图
Visualizer.plot_solution_comparison(
    model, pde, device,
    radius=config['radius'],
    save_path='solution_only.png'
)
```

**热传导问题**：
```python
# 只生成温度分布
plot_results(trained_model, loss_history)

# 只生成温度剖面
plot_temperature_profile(trained_model)
```

---

## 五、图表用途建议

### 论文使用

**推荐图表**：
- `deepritz_comprehensive.png` - 综合展示所有结果
- `heat_results.png` - 完整的热传导结果
- 使用PDF格式保存（矢量图，无损缩放）

### 报告使用

**推荐图表**：
- `training_history.png` - 展示训练过程
- `solution_comparison.png` - 对比网络解与精确解
- `heat_profiles.png` - 展示温度分布规律

### 调试使用

**推荐图表**：
- 损失曲线 - 判断是否收敛
- 误差曲线 - 评估精度
- 切片图 - 检查边界条件

---

## 六、性能优化建议

### 快速预览模式

```python
# 减少训练步数和采样点
config['train_steps'] = 1000
config['body_batch'] = 1024
config['boundary_batch'] = 1024

# 降低图表分辨率
n_samples = 50  # 从100降到50
```

### 高质量模式

```python
# 增加训练步数
config['train_steps'] = 10000

# 提高图表分辨率
n_samples = 200  # 从100提高到200

# 使用高DPI保存
plt.savefig('result.png', dpi=600)
```

---

## 七、输出文件清单

### Poisson方程
```
deepritz_standalone.py 运行后：
├── training_history.png      (训练历史)
└── solution_comparison.png   (解对比)

plot_deepritz.py 运行后：
├── deepritz_comprehensive.png  (综合结果)
├── deepritz_slices.png        (切片对比)
└── deepritz_comparison.png    (解对比)
```

### 热传导问题
```
heat.py 运行后：
├── heat_results.png    (完整结果)
└── heat_profiles.png   (温度剖面)
```

---

## 八、技术支持

如有问题，请检查：
1. Python版本 >= 3.7
2. PyTorch已正确安装
3. Matplotlib版本 >= 3.0
4. NumPy版本 >= 1.18

**依赖安装**：
```bash
pip install torch numpy matplotlib
```

---

**最后更新**: 2026-01-02
**版本**: 1.0
