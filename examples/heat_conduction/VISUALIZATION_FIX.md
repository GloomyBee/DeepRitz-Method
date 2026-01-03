# 可视化脚本修复说明

## 问题
1. `torch.load` 的 FutureWarning
2. `loss_history` 为空列表导致 IndexError

## 修复

### 1. torch.load 警告
**问题：** PyTorch警告使用`weights_only=False`
**修复：** 明确指定参数
```python
checkpoint = torch.load(model_path, map_location=device, weights_only=False)
```

### 2. 空 loss_history 处理
**问题：** 访问`loss_history[-1]`时列表为空
**修复：** 添加检查
```python
if loss_history:
    print(f"  最终损失: {loss_history[-1]:.6f}")
else:
    print(f"  警告: 未找到损失历史记录")
```

### 3. 绘图函数处理
**问题：** plot_results中也访问空列表
**修复：** 添加条件判断
```python
if loss_history:
    ax2.plot(loss_history, linewidth=1.5, color='blue')
    # ...
else:
    ax2.text(0.5, 0.5, 'Loss history not available', ...)
```

## 为什么 loss_history 为空？

训练脚本保存checkpoint时没有包含loss_history：
```python
checkpoint = {
    'model_state_dict': model.state_dict(),
    'config': params,
    'residual': residual,
    'train_time': train_time
    # 缺少 'loss_history': loss_history
}
```

## 解决方案

可视化脚本会尝试从文件加载：
```python
loss_file = os.path.join(os.path.dirname(model_path), 'heat_loss_history.txt')
if os.path.exists(loss_file):
    with open(loss_file, 'r') as f:
        lines = f.readlines()[1:]
        loss_history = [float(line.split()[1]) for line in lines if line.strip()]
```

如果文件存在，会从`heat_loss_history.txt`加载损失历史。

## 现在可以运行

```bash
python visualize_heat.py
```

即使没有loss_history，也能正常显示其他5个子图。
