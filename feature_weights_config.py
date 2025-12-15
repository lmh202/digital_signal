"""
特征权重配置文件
根据特征重要性图调整各个特征的权重

使用说明：
1. 修改 FEATURE_WEIGHTS 数组中的权重值
2. 设置 USE_FEATURE_WEIGHTS = True 启用权重，= False 禁用权重
3. 运行 classification_model.py 即可应用新权重

权重设置建议：
- 根据特征重要性图，重要性高的特征给予更大的权重
- 权重值的和不要求等于1.0，程序会自动归一化
- 建议权重范围：0.01 - 0.20 之间

特征索引对应关系：
索引 | 特征名称                  | 默认权重 | 特征重要性（参考）
-----|--------------------------|---------|------------------
 0   | main_frequency           | 0.02    | 0.0184 (低)
 1   | spectral_centroid        | 0.08    | 0.0545 (中)
 2   | spectral_bandwidth       | 0.11    | 0.0568 (中)
 3   | band_energy_0_2k         | 0.08    | 0.0844 (中高)
 4   | band_energy_2k_5k        | 0.08    | 0.0774 (中高)
 5   | band_energy_5k_10k       | 0.06    | 0.1113 (高)
 6   | band_energy_10k_15k      | 0.05    | 0.0489 (中低)
 7   | band_energy_15k_25.6k    | 0.04    | 0.0485 (中低)
 8   | peak_freq_1              | 0.16    | 未在前10（低）
 9   | peak_freq_2              | 0.12    | 0.1155 (很高)
10   | peak_freq_3              | 0.11    | 0.1575 (最高)
11   | peak_freq_4              | 0.04    | 0.0400 (低)
12   | peak_freq_5              | 0.03    | 0.0292 (低)
13   | rms                      | 0.09    | 0.0918 (高)
14   | peak_amplitude           | 0.06    | 0.0548 (中)
15   | zero_crossing_rate       | 0.08    | 0.0292 (低)
"""

import numpy as np

# ==================== 可调参数 ====================

# 是否启用特征权重
USE_FEATURE_WEIGHTS = True

# 特征权重数组（16个特征）
# 策略1：根据实测特征重要性调整（推荐）
FEATURE_WEIGHTS = np.array([
    0.00,   # 0: main_frequency (重要性: 0.0184)
    0.05,   # 1: spectral_centroid (重要性: 0.0545)
    0.06,   # 2: spectral_bandwidth (重要性: 0.0568)
    0.08,   # 3: band_energy_0_2k (重要性: 0.0844)
    0.08,   # 4: band_energy_2k_5k (重要性: 0.0774)
    0.11,   # 5: band_energy_5k_10k (重要性: 0.1113) - 较高
    0.05,   # 6: band_energy_10k_15k (重要性: 0.0489)
    0.05,   # 7: band_energy_15k_25.6k (重要性: 0.0485)
    0.00,   # 8: peak_freq_1 (未在前10)
    0.12,   # 9: peak_freq_2 (重要性: 0.1155) - 很高
    0.16,   # 10: peak_freq_3 (重要性: 0.1575) - 最高！
    0.04,   # 11: peak_freq_4 (重要性: 0.0400)
    0.00,   # 12: peak_freq_5 (重要性: 0.0292)
    0.09,   # 13: rms (重要性: 0.0918) - 高
    0.05,   # 14: peak_amplitude (重要性: 0.0548)
    0.03,   # 15: zero_crossing_rate (重要性: 0.0292)
])  # 总和 = 1.05

# ==================== 预设配置方案 ====================

# 策略2：均匀权重（不加权）
UNIFORM_WEIGHTS = np.ones(16)

# 策略3：强调频谱峰值特征
PEAK_FOCUSED_WEIGHTS = np.array([
    0.02, 0.04, 0.05,  # 基础频域特征
    0.05, 0.05, 0.08, 0.04, 0.04,  # 能量带
    0.10, 0.18, 0.20, 0.06, 0.04,  # 峰值频率（强调2、3）
    0.07, 0.04, 0.02   # 时域特征
])

# 策略4：平衡频域和时域
BALANCED_WEIGHTS = np.array([
    0.03, 0.08, 0.08,  # 基础频域
    0.07, 0.07, 0.10, 0.05, 0.05,  # 能量带
    0.05, 0.12, 0.15, 0.04, 0.03,  # 峰值频率
    0.10, 0.08, 0.05   # 时域特征（增强RMS和峰值幅度）
])

# ==================== 快速切换配置 ====================
# 取消下面某一行的注释以使用对应的权重策略

# FEATURE_WEIGHTS = UNIFORM_WEIGHTS  # 不加权
# FEATURE_WEIGHTS = PEAK_FOCUSED_WEIGHTS  # 强调峰值
# FEATURE_WEIGHTS = BALANCED_WEIGHTS  # 平衡策略

# ==================== 权重验证 ====================
def validate_weights(weights, tolerance=0.001):
    """验证权重配置的合理性"""
    if len(weights) != 16:
        raise ValueError(f"权重数组长度必须为16，当前为{len(weights)}")
    
    if np.any(weights < 0):
        raise ValueError("权重值不能为负数")
    
    if np.sum(weights) < tolerance:
        raise ValueError("权重总和不能为0")
    
    print("✅ 权重配置验证通过")
    print(f"   权重总和: {np.sum(weights):.4f}")
    print(f"   权重范围: [{np.min(weights):.4f}, {np.max(weights):.4f}]")
    return True

# 自动验证当前配置
if __name__ == "__main__":
    print("="*60)
    print("特征权重配置验证")
    print("="*60)
    print(f"\n启用状态: {'✅ 已启用' if USE_FEATURE_WEIGHTS else '❌ 未启用'}")
    
    if USE_FEATURE_WEIGHTS:
        validate_weights(FEATURE_WEIGHTS)
        
        print("\n当前权重配置:")
        feature_names = [
            'main_frequency', 'spectral_centroid', 'spectral_bandwidth',
            'band_energy_0_2k', 'band_energy_2k_5k', 'band_energy_5k_10k',
            'band_energy_10k_15k', 'band_energy_15k_25.6k',
            'peak_freq_1', 'peak_freq_2', 'peak_freq_3', 'peak_freq_4', 'peak_freq_5',
            'rms', 'peak_amplitude', 'zero_crossing_rate'
        ]
        
        # 按权重从大到小排序
        sorted_indices = np.argsort(FEATURE_WEIGHTS)[::-1]
        
        print(f"\n{'排名':<6} {'索引':<6} {'特征名':<30} {'权重':<10}")
        print("-" * 60)
        for rank, idx in enumerate(sorted_indices, 1):
            print(f"{rank:<6} {idx:<6} {feature_names[idx]:<30} {FEATURE_WEIGHTS[idx]:<10.4f}")
        
        print("\n" + "="*60)
        print("💡 提示：修改 FEATURE_WEIGHTS 数组后，运行 classification_model.py 即可应用新权重")
