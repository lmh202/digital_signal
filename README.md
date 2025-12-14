# 槽楔模型测试数据分析

## 项目概述

本项目用于分析不同松紧度下的槽楔敲击测试数据，通过信号处理和机器学习方法实现松紧度的自动分类。

## 文件结构

```
.
├── feature_extraction.py      # 特征提取脚本
├── classification_model.py    # 分类模型脚本
├── acquisitionData-*.txt      # 原始数据文件（10个不同松紧度）
├── features/                  # 特征提取结果
│   ├── endpoints/            # 端点检测效果图
│   ├── endpoints_zoom/       # 端点检测放大图
│   ├── frequency/            # 时频分析图
│   ├── features.npy          # 特征矩阵
│   ├── labels.npy            # 标签
│   └── *.json               # 元数据
├── results/                   # 分类结果
│   ├── loss_curves.png       # 训练损失曲线
│   ├── confusion_matrix.png  # 混淆矩阵
│   ├── feature_importance.png # 特征重要性
│   └── *.json               # 分类结果详情
└── models/                    # 保存的模型
```

## 运行方法

### 1. 环境配置

```powershell
# 已配置虚拟环境，激活后安装依赖
D:/Downloads/digital_signal/Attempt/.venv/Scripts/python.exe -m pip install numpy matplotlib scipy scikit-learn torch tqdm
```

### 2. 特征提取

```powershell
D:/Downloads/digital_signal/Attempt/.venv/Scripts/python.exe feature_extraction.py
```

### 3. 模型训练与分类

```powershell
D:/Downloads/digital_signal/Attempt/.venv/Scripts/python.exe classification_model.py
```

## 技术方案

### 端点检测（峰值对齐方法）

使用**包络+峰值检测**替代传统的能量阈值分段方法，解决了峰值漏检和对齐不准的问题：

#### 核心步骤：
1. **带通滤波**：去除直流和高频噪声（100-15000 Hz）
2. **整流**：取绝对值
3. **低通平滑**：移动平均得到包络
4. **峰值检测**：使用 `scipy.signal.find_peaks` 在包络上检测峰值
   - `prominence`：峰值显著性阈值（相对于最大值 30%）
   - `distance`：相邻峰值最小间隔（0.5秒）
5. **信号截取**：以峰值为中心，前后各取固定长度（前10ms + 后80ms = 90ms）

#### 关键参数（可调）：
```python
BANDPASS_LOW = 100              # 带通滤波器低频截止 (Hz)
BANDPASS_HIGH = 15000           # 带通滤波器高频截止 (Hz)
ENVELOPE_WINDOW_SIZE = 256      # 包络平滑窗口大小（采样点）
PEAK_PROMINENCE = 0.3           # 峰值显著性阈值（相对值）
PEAK_MIN_DISTANCE = 0.5         # 相邻峰值最小间隔（秒）
SEGMENT_BEFORE_PEAK = 0.01      # 峰值前截取长度（秒）
SEGMENT_AFTER_PEAK = 0.08       # 峰值后截取长度（秒）
SIGNAL_DURATION = 0.07          # 用于特征提取的固定长度（秒）
```

### 特征提取（16维）

#### 频域特征（13维）：
1. **主频率**：最大幅值对应的频率
2. **频谱质心**：频谱的"重心"位置
3. **频谱带宽**：频率分布的离散程度
4. **5个频段能量比**：0-2k, 2-5k, 5-10k, 10-15k, 15-25.6k Hz
5. **前5个峰值频率**：按幅值排序的频谱峰

#### 时域特征（3维）：
1. **RMS**：均方根值
2. **峰值幅度**：信号最大绝对值
3. **过零率**：信号符号变化频率

### 分类模型

- **模型架构**：3层全连接神经网络（128-64-32）
- **框架**：PyTorch（带 BatchNorm + Dropout）
- **优化器**：Adam（学习率 0.001）
- **训练**：200 epochs，batch size 32

## 结果对比

| 指标 | 旧方法（能量阈值） | 新方法（峰值对齐） | 改进 |
|------|-------------------|-------------------|------|
| 检测到的信号数 | 452 | 382 | 更精准 |
| 测试集准确率 | 61.54% | **75.32%** | **+13.78%** |
| 类别 400-1600 F1 | 0.89-0.93 | **1.00** | 完美分类 |
| 频谱质心与松紧度相关性 | 不显著 | **显著（p=0.034）** | 更强相关 |

### 性能提升原因：
1. ✅ **峰值对齐**：确保每个样本的主能量在固定位置，减少时域抖动
2. ✅ **减少误检**：prominence 阈值过滤掉弱峰和噪声
3. ✅ **一致性**：所有样本以相同的相位（峰值）开始，特征更稳定
4. ✅ **补零处理**：保留边界样本，避免数据丢失

## 可视化输出

### 端点检测效果图
- 上图：原始信号 + 峰值标记（红线）
- 下图：包络曲线 + 检测到的峰值（红点）

### 端点检测放大图
- 显示单个敲击的峰值对齐效果
- 包含：信号波形、包络、起点、终点、峰值中心

### 时频图
- 上图：时域波形（70ms 固定长度）
- 下图：频域幅频特性（FFT结果）

### 分类结果图
- **损失曲线**：训练/测试损失和准确率随epoch变化
- **混淆矩阵**：各类别分类详情
- **特征重要性**：随机森林评估的特征贡献度
- **松紧度-频率关系**：线性回归分析

## 参数调优建议

如果检测效果不理想，可按以下顺序调整参数：

1. **调整峰值显著性**（`PEAK_PROMINENCE`）：
   - 值太小 → 检测到太多伪峰
   - 值太大 → 漏掉真实峰值
   - 建议范围：0.2 - 0.5

2. **调整峰值间隔**（`PEAK_MIN_DISTANCE`）：
   - 根据敲击频率设置（通常 0.3 - 1.0 秒）

3. **调整带通范围**（`BANDPASS_LOW`, `BANDPASS_HIGH`）：
   - 根据敲击信号的主频范围设置

4. **调整截取长度**（`SEGMENT_BEFORE_PEAK`, `SEGMENT_AFTER_PEAK`）：
   - 根据信号衰减速度调整

## 依赖库

```
numpy>=1.20.0
matplotlib>=3.3.0
scipy>=1.6.0
scikit-learn>=0.24.0
torch>=1.8.0
tqdm>=4.60.0
```

## 作者

生成时间：2025年12月14日

## 许可

本项目仅供学习和研究使用。
