"""
槽楔模型测试数据分析 - 分类模型脚本
功能：
1. 加载提取的特征
2. 搭建神经网络分类模型
3. 训练模型并计算分类精度
4. 绘制损失曲线
5. 分析松紧度（压力值）和频率的变化关系
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# 尝试导入深度学习框架
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    USE_TORCH = True
    print("使用 PyTorch 作为深度学习框架")
except ImportError:
    USE_TORCH = False
    from sklearn.neural_network import MLPClassifier
    print("PyTorch 未安装，使用 sklearn MLPClassifier")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 参数设置
FEATURE_DIR = "features"
MODEL_DIR = "models"
RESULT_DIR = "results"

# 训练参数
EPOCHS = 200
BATCH_SIZE = 32
LEARNING_RATE = 0.001
HIDDEN_SIZES = [128, 64, 32]
TEST_SIZE = 0.2
RANDOM_STATE = 42


def create_output_dirs():
    """创建输出目录"""
    for d in [MODEL_DIR, RESULT_DIR]:
        os.makedirs(d, exist_ok=True)
    print(f"已创建输出目录: {MODEL_DIR}, {RESULT_DIR}")


def load_features():
    """加载特征和标签"""
    print("\n加载特征数据...")
    
    X = np.load(os.path.join(FEATURE_DIR, "features.npy"))
    y = np.load(os.path.join(FEATURE_DIR, "labels.npy"))
    
    with open(os.path.join(FEATURE_DIR, "feature_names.json"), 'r', encoding='utf-8') as f:
        feature_names = json.load(f)
    
    print(f"特征矩阵形状: {X.shape}")
    print(f"标签数量: {len(y)}")
    print(f"类别: {np.unique(y)}")
    
    return X, y, feature_names


def preprocess_data(X, y, test_size=TEST_SIZE):
    """数据预处理"""
    print("\n数据预处理...")
    
    # 标签编码
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=test_size, random_state=RANDOM_STATE, stratify=y_encoded
    )
    
    print(f"训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")
    print(f"类别映射: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
    
    return X_train, X_test, y_train, y_test, scaler, label_encoder


class NeuralNetwork(nn.Module):
    """PyTorch 神经网络模型"""
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(NeuralNetwork, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def train_pytorch_model(X_train, X_test, y_train, y_test, num_classes):
    """使用 PyTorch 训练模型"""
    print("\n使用 PyTorch 训练神经网络...")
    
    # 转换为张量
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 创建模型
    input_size = X_train.shape[1]
    model = NeuralNetwork(input_size, HIDDEN_SIZES, num_classes)
    print(f"模型结构:\n{model}")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    # 训练历史
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    # 训练循环
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        scheduler.step()
        
        # 计算训练指标
        train_loss = epoch_loss / len(train_loader)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # 计算测试指标
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            test_loss = criterion(test_outputs, y_test_tensor).item()
            _, test_predicted = torch.max(test_outputs, 1)
            test_acc = (test_predicted == y_test_tensor).sum().item() / len(y_test_tensor)
        
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                  f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    
    # 最终评估
    model.eval()
    with torch.no_grad():
        final_outputs = model(X_test_tensor)
        _, final_predictions = torch.max(final_outputs, 1)
        final_predictions = final_predictions.numpy()
    
    history = {
        'train_loss': train_losses,
        'test_loss': test_losses,
        'train_accuracy': train_accuracies,
        'test_accuracy': test_accuracies
    }
    
    # 保存模型
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "neural_network.pth"))
    
    return model, final_predictions, history


def train_sklearn_model(X_train, X_test, y_train, y_test, num_classes):
    """使用 sklearn 训练模型"""
    print("\n使用 sklearn MLPClassifier 训练神经网络...")
    
    model = MLPClassifier(
        hidden_layer_sizes=tuple(HIDDEN_SIZES),
        activation='relu',
        solver='adam',
        alpha=1e-4,
        batch_size=BATCH_SIZE,
        learning_rate_init=LEARNING_RATE,
        max_iter=EPOCHS,
        random_state=RANDOM_STATE,
        early_stopping=False,
        verbose=True
    )
    
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    # sklearn 的 loss_curve_ 只有训练损失
    history = {
        'train_loss': model.loss_curve_,
        'test_loss': [],  # sklearn 不提供测试损失
        'train_accuracy': [],
        'test_accuracy': []
    }
    
    return model, predictions, history


def plot_loss_curves(history, save_path):
    """绘制损失曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 损失曲线
    axes[0].plot(history['train_loss'], 'b-', label='训练损失', linewidth=2)
    if len(history['test_loss']) > 0:
        axes[0].plot(history['test_loss'], 'r-', label='测试损失', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('损失')
    axes[0].set_title('训练和测试损失曲线')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 准确率曲线
    if len(history['train_accuracy']) > 0:
        axes[1].plot(history['train_accuracy'], 'b-', label='训练准确率', linewidth=2)
        axes[1].plot(history['test_accuracy'], 'r-', label='测试准确率', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('准确率')
        axes[1].set_title('训练和测试准确率曲线')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, '准确率数据不可用', ha='center', va='center', fontsize=14)
        axes[1].set_title('准确率曲线')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"损失曲线已保存: {save_path}")


def plot_confusion_matrix(y_true, y_pred, labels, save_path):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # 设置刻度
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=labels,
           yticklabels=labels,
           title='混淆矩阵',
           ylabel='真实标签',
           xlabel='预测标签')
    
    # 旋转x轴标签
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # 在格子中添加数值
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"混淆矩阵已保存: {save_path}")


def analyze_feature_importance(X, y, feature_names, save_path):
    """分析特征重要性"""
    from sklearn.ensemble import RandomForestClassifier
    
    print("\n分析特征重要性...")
    
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    rf.fit(X, y)
    
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # 绘制特征重要性图
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(feature_names)))
    bars = ax.bar(range(len(feature_names)), importances[indices], color=colors)
    
    ax.set_xlabel('特征')
    ax.set_ylabel('重要性')
    ax.set_title('特征重要性分析')
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"特征重要性图已保存: {save_path}")
    
    print("\n特征重要性排名:")
    for i, idx in enumerate(indices[:10]):
        print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")


def analyze_tightness_frequency_detailed():
    """详细分析松紧度与频率的关系"""
    print("\n" + "="*60)
    print("松紧度与频率关系详细分析")
    print("="*60)
    
    # 加载关系数据
    with open(os.path.join(FEATURE_DIR, "tightness_frequency_relationship.json"), 'r', encoding='utf-8') as f:
        relationship = json.load(f)
    
    tightness = relationship['tightness']
    main_frequencies = relationship['main_frequencies']
    spectral_centroids = relationship['spectral_centroids']
    
    # 计算相关系数
    from scipy import stats
    
    corr_main, p_main = stats.pearsonr(tightness, main_frequencies)
    corr_centroid, p_centroid = stats.pearsonr(tightness, spectral_centroids)
    
    print(f"\n松紧度与主频率的相关性:")
    print(f"  Pearson相关系数: {corr_main:.4f}")
    print(f"  p值: {p_main:.6f}")
    print(f"  结论: {'显著相关' if p_main < 0.05 else '不显著相关'}")
    
    print(f"\n松紧度与频谱质心的相关性:")
    print(f"  Pearson相关系数: {corr_centroid:.4f}")
    print(f"  p值: {p_centroid:.6f}")
    print(f"  结论: {'显著相关' if p_centroid < 0.05 else '不显著相关'}")
    
    # 线性回归分析
    slope_main, intercept_main, r_main, _, _ = stats.linregress(tightness, main_frequencies)
    slope_centroid, intercept_centroid, r_centroid, _, _ = stats.linregress(tightness, spectral_centroids)
    
    # 绘制详细关系图
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 主频率散点图和回归线
    axes[0, 0].scatter(tightness, main_frequencies, s=100, c='blue', alpha=0.7, edgecolors='black')
    x_line = np.array([min(tightness), max(tightness)])
    axes[0, 0].plot(x_line, slope_main * x_line + intercept_main, 'r--', linewidth=2, 
                    label=f'线性回归: y = {slope_main:.4f}x + {intercept_main:.2f}')
    axes[0, 0].set_xlabel('松紧度（压力值）')
    axes[0, 0].set_ylabel('平均主频率 (Hz)')
    axes[0, 0].set_title(f'松紧度与主频率关系 (R² = {r_main**2:.4f})')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 频谱质心散点图和回归线
    axes[0, 1].scatter(tightness, spectral_centroids, s=100, c='red', alpha=0.7, edgecolors='black')
    axes[0, 1].plot(x_line, slope_centroid * x_line + intercept_centroid, 'b--', linewidth=2,
                    label=f'线性回归: y = {slope_centroid:.4f}x + {intercept_centroid:.2f}')
    axes[0, 1].set_xlabel('松紧度（压力值）')
    axes[0, 1].set_ylabel('平均频谱质心 (Hz)')
    axes[0, 1].set_title(f'松紧度与频谱质心关系 (R² = {r_centroid**2:.4f})')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 主频率变化趋势
    axes[1, 0].plot(tightness, main_frequencies, 'bo-', markersize=10, linewidth=2)
    for i, (t, f) in enumerate(zip(tightness, main_frequencies)):
        axes[1, 0].annotate(f'{f:.0f}Hz', (t, f), textcoords="offset points", 
                           xytext=(0, 10), ha='center', fontsize=9)
    axes[1, 0].set_xlabel('松紧度（压力值）')
    axes[1, 0].set_ylabel('平均主频率 (Hz)')
    axes[1, 0].set_title('主频率随松紧度的变化趋势')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 频谱质心变化趋势
    axes[1, 1].plot(tightness, spectral_centroids, 'ro-', markersize=10, linewidth=2)
    for i, (t, c) in enumerate(zip(tightness, spectral_centroids)):
        axes[1, 1].annotate(f'{c:.0f}Hz', (t, c), textcoords="offset points",
                           xytext=(0, 10), ha='center', fontsize=9)
    axes[1, 1].set_xlabel('松紧度（压力值）')
    axes[1, 1].set_ylabel('平均频谱质心 (Hz)')
    axes[1, 1].set_title('频谱质心随松紧度的变化趋势')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(RESULT_DIR, "tightness_frequency_detailed.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n详细关系图已保存: {save_path}")
    
    # 保存分析结果
    analysis_results = {
        'main_frequency': {
            'correlation': corr_main,
            'p_value': p_main,
            'slope': slope_main,
            'intercept': intercept_main,
            'r_squared': r_main**2
        },
        'spectral_centroid': {
            'correlation': corr_centroid,
            'p_value': p_centroid,
            'slope': slope_centroid,
            'intercept': intercept_centroid,
            'r_squared': r_centroid**2
        }
    }
    
    with open(os.path.join(RESULT_DIR, "frequency_analysis.json"), 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, ensure_ascii=False, indent=2)
    
    return analysis_results


def main():
    """主函数"""
    print("="*60)
    print("槽楔模型测试数据分析 - 分类模型")
    print("="*60)
    
    # 创建输出目录
    create_output_dirs()
    
    # 加载特征
    X, y, feature_names = load_features()
    
    # 数据预处理
    X_train, X_test, y_train, y_test, scaler, label_encoder = preprocess_data(X, y)
    num_classes = len(label_encoder.classes_)
    
    # 训练模型
    if USE_TORCH:
        model, predictions, history = train_pytorch_model(X_train, X_test, y_train, y_test, num_classes)
    else:
        model, predictions, history = train_sklearn_model(X_train, X_test, y_train, y_test, num_classes)
    
    # 计算分类精度
    accuracy = accuracy_score(y_test, predictions)
    print(f"\n{'='*60}")
    print(f"分类结果")
    print(f"{'='*60}")
    print(f"测试集准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # 详细分类报告
    print("\n分类报告:")
    class_labels = [str(c) for c in label_encoder.classes_]
    print(classification_report(y_test, predictions, target_names=class_labels))
    
    # 绘制损失曲线
    plot_loss_curves(history, os.path.join(RESULT_DIR, "loss_curves.png"))
    
    # 绘制混淆矩阵
    plot_confusion_matrix(y_test, predictions, class_labels, 
                         os.path.join(RESULT_DIR, "confusion_matrix.png"))
    
    # 分析特征重要性
    analyze_feature_importance(X, y, feature_names, 
                              os.path.join(RESULT_DIR, "feature_importance.png"))
    
    # 详细分析松紧度与频率关系
    analyze_tightness_frequency_detailed()
    
    # 保存分类结果
    results = {
        'accuracy': float(accuracy),
        'num_classes': num_classes,
        'class_labels': class_labels,
        'train_samples': len(y_train),
        'test_samples': len(y_test),
        'feature_dim': X.shape[1],
        'model_architecture': {
            'hidden_sizes': HIDDEN_SIZES,
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE
        }
    }
    
    with open(os.path.join(RESULT_DIR, "classification_results.json"), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print("分析完成")
    print(f"{'='*60}")
    print(f"结果已保存到: {RESULT_DIR}/")
    print("  - loss_curves.png: 损失曲线")
    print("  - confusion_matrix.png: 混淆矩阵")
    print("  - feature_importance.png: 特征重要性")
    print("  - tightness_frequency_detailed.png: 松紧度与频率关系")
    print("  - classification_results.json: 分类结果汇总")
    print("  - frequency_analysis.json: 频率分析结果")


if __name__ == "__main__":
    main()
