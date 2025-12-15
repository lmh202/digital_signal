"""
孤立尖峰持续时间分析脚本（原始信号版本 - 无预处理）
功能：直接在原始信号上检测异常大的峰值，统计连续大的离散点个数，计算持续时间
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os
import glob
import json

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 参数设置 ====================
SAMPLE_RATE = 51200  # 采样率 Hz
DATA_DIR = "."  # 数据目录
OUTPUT_DIR = "peak_duration_analysis_raw"  # 输出目录

# 峰值检测参数（直接在原始信号上）
PEAK_PERCENTILE = 99.5  # 峰值阈值百分位数（检测前0.5%的大值）
PEAK_MIN_DISTANCE = 2560  # 相邻峰值最小间隔（采样点数，约50ms）

# 持续时间计算参数
DURATION_THRESHOLD_RATIO = 0.3  # 持续时间边界阈值（相对于峰值的比例）


def create_output_dir():
    """创建输出目录"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"已创建输出目录: {OUTPUT_DIR}")


def load_data(filepath):
    """加载数据文件"""
    data = np.loadtxt(filepath)
    return data


def analyze_peak_duration_raw(signal_data, sample_rate=SAMPLE_RATE):
    """
    直接在原始信号上分析峰值持续时间（无预处理）
    
    算法：
    1. 计算信号绝对值
    2. 找到超过阈值的峰值点
    3. 对每个峰，向前后扩展，统计连续大于某阈值的点数
    4. 计算持续时间 = 采样点数 / 采样率
    
    返回：
        peak_info: 列表，每个元素是字典 {
            'peak_idx': 峰值位置索引,
            'peak_amplitude': 峰值幅值（原始信号值）,
            'start_idx': 起始位置,
            'end_idx': 结束位置,
            'duration_samples': 持续采样点数,
            'duration_ms': 持续时间(毫秒)
        }
    """
    # 计算信号绝对值
    abs_signal = np.abs(signal_data)
    
    # 计算峰值检测阈值（使用百分位数）
    peak_threshold = np.percentile(abs_signal, PEAK_PERCENTILE)
    
    print(f"  原始信号幅值范围: [{np.min(signal_data):.4f}, {np.max(signal_data):.4f}]")
    print(f"  峰值检测阈值 ({PEAK_PERCENTILE}百分位): {peak_threshold:.4f}")
    
    # 检测峰值（在绝对值信号上）
    peaks, properties = find_peaks(
        abs_signal,
        height=peak_threshold,
        distance=PEAK_MIN_DISTANCE
    )
    
    if len(peaks) == 0:
        return []
    
    peak_info = []
    
    # 对每个峰值计算持续时间
    for i, peak_idx in enumerate(peaks):
        peak_value = abs_signal[peak_idx]
        original_peak_value = signal_data[peak_idx]  # 保留原始信号值（带符号）
        
        # 持续时间边界阈值（相对于峰值）
        threshold = peak_value * DURATION_THRESHOLD_RATIO
        
        # === 向前搜索起点：找到第一个低于阈值的点 ===
        start_idx = peak_idx
        for idx in range(peak_idx - 1, -1, -1):
            if abs_signal[idx] >= threshold:
                start_idx = idx
            else:
                break
        
        # === 向后搜索终点：找到第一个低于阈值的点 ===
        end_idx = peak_idx
        for idx in range(peak_idx + 1, len(abs_signal)):
            if abs_signal[idx] >= threshold:
                end_idx = idx
            else:
                break
        
        # 计算持续时间
        duration_samples = end_idx - start_idx + 1
        duration_ms = duration_samples / sample_rate * 1000
        
        peak_info.append({
            'peak_idx': int(peak_idx),
            'peak_amplitude': float(original_peak_value),  # 原始信号值
            'peak_abs_amplitude': float(peak_value),  # 绝对值
            'start_idx': int(start_idx),
            'end_idx': int(end_idx),
            'duration_samples': int(duration_samples),
            'duration_ms': float(duration_ms)
        })
    
    return peak_info


def plot_peak_duration_distribution(all_results):
    """绘制所有文件的峰值持续时间分布图"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 准备数据
    tightness_values = sorted(all_results.keys())
    
    # === 子图1: 持续时间箱线图 ===
    durations_by_file = []
    labels = []
    for tight in tightness_values:
        durations = [p['duration_ms'] for p in all_results[tight]['peaks']]
        durations_by_file.append(durations)
        labels.append(f"{tight}")
    
    bp = axes[0].boxplot(durations_by_file, tick_labels=labels, patch_artist=True)
    
    # 设置颜色
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    
    axes[0].axhline(y=5, color='red', linestyle='--', linewidth=1.5, label='5ms')
    axes[0].axhline(y=10, color='orange', linestyle='--', linewidth=1.5, label='10ms (毛刺阈值)')
    axes[0].axhline(y=20, color='green', linestyle='--', linewidth=1.5, label='20ms (真实敲击阈值)')
    axes[0].set_xlabel('松紧度')
    axes[0].set_ylabel('持续时间 (ms)')
    axes[0].set_title('各文件峰值持续时间分布（箱线图）- 原始信号')
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].legend()
    
    # === 子图2: 持续时间统计柱状图 ===
    width = 0.2
    x = np.arange(len(tightness_values))
    
    very_short_counts = []  # <5ms
    short_counts = []  # 5-10ms
    medium_counts = []  # 10-20ms
    long_counts = []  # ≥20ms
    
    for tight in tightness_values:
        durations = [p['duration_ms'] for p in all_results[tight]['peaks']]
        very_short_counts.append(sum(1 for d in durations if d < 5))
        short_counts.append(sum(1 for d in durations if 5 <= d < 10))
        medium_counts.append(sum(1 for d in durations if 10 <= d < 20))
        long_counts.append(sum(1 for d in durations if d >= 20))
    
    axes[1].bar(x - 1.5*width, very_short_counts, width, label='<5ms (极短)', color='red', alpha=0.7)
    axes[1].bar(x - 0.5*width, short_counts, width, label='5-10ms (短)', color='orange', alpha=0.7)
    axes[1].bar(x + 0.5*width, medium_counts, width, label='10-20ms (中)', color='yellow', alpha=0.7)
    axes[1].bar(x + 1.5*width, long_counts, width, label='≥20ms (长)', color='green', alpha=0.7)
    
    axes[1].set_xlabel('松紧度')
    axes[1].set_ylabel('峰值数量')
    axes[1].set_title('各文件峰值持续时间分类统计 - 原始信号')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "peak_duration_distribution_raw.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n持续时间分布图已保存: {save_path}")


def plot_individual_file_histogram(tightness, peak_info, filename):
    """绘制单个文件的持续时间直方图"""
    durations = [p['duration_ms'] for p in peak_info]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制直方图
    n, bins, patches = ax.hist(durations, bins=30, edgecolor='black', alpha=0.7)
    
    # 标记阈值线
    ax.axvline(x=5, color='red', linestyle='--', linewidth=2, label='5ms')
    ax.axvline(x=10, color='orange', linestyle='--', linewidth=2, label='10ms (毛刺阈值)')
    ax.axvline(x=20, color='green', linestyle='--', linewidth=2, label='20ms (真实敲击阈值)')
    
    ax.set_xlabel('持续时间 (ms)')
    ax.set_ylabel('峰值数量')
    ax.set_title(f'松紧度 {tightness} - 峰值持续时间分布（原始信号）')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加统计信息文本
    stats_text = f"总峰值数: {len(durations)}\n"
    stats_text += f"最小: {np.min(durations):.2f} ms\n"
    stats_text += f"最大: {np.max(durations):.2f} ms\n"
    stats_text += f"平均: {np.mean(durations):.2f} ms\n"
    stats_text += f"中位数: {np.median(durations):.2f} ms"
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f"{filename}_duration_histogram_raw.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """主函数"""
    print("="*70)
    print("孤立尖峰持续时间分析 - 原始信号（无预处理）")
    print("="*70)
    print(f"采样率: {SAMPLE_RATE} Hz")
    print(f"峰值检测: {PEAK_PERCENTILE}百分位数（前{100-PEAK_PERCENTILE:.1f}%的大值）")
    print(f"峰值最小间隔: {PEAK_MIN_DISTANCE}采样点 ({PEAK_MIN_DISTANCE/SAMPLE_RATE*1000:.1f}ms)")
    print(f"持续时间阈值比例: {DURATION_THRESHOLD_RATIO} (相对于峰值)")
    print("="*70)
    
    # 创建输出目录
    create_output_dir()
    
    # 获取所有数据文件
    data_files = sorted(glob.glob(os.path.join(DATA_DIR, "acquisitionData-*.txt")))
    print(f"\n找到 {len(data_files)} 个数据文件\n")
    
    if len(data_files) == 0:
        print("错误: 未找到数据文件!")
        return
    
    # 处理所有文件
    all_results = {}
    
    for filepath in data_files:
        basename = os.path.basename(filepath)
        filename = os.path.splitext(basename)[0]
        tightness = int(filename.split('-')[1])
        
        print(f"{'='*70}")
        print(f"处理文件: {basename} (松紧度: {tightness})")
        print(f"{'='*70}")
        
        # 加载数据
        signal_data = load_data(filepath)
        print(f"数据长度: {len(signal_data)} 采样点, 时长: {len(signal_data)/SAMPLE_RATE:.2f} 秒")
        
        # 分析峰值持续时间（原始信号）
        peak_info = analyze_peak_duration_raw(signal_data)
        
        if len(peak_info) == 0:
            print("⚠️  未检测到峰值\n")
            continue
        
        # 统计信息
        durations_ms = [p['duration_ms'] for p in peak_info]
        durations_samples = [p['duration_samples'] for p in peak_info]
        
        print(f"\n📊 检测到 {len(peak_info)} 个峰值")
        print(f"\n持续采样点数统计:")
        print(f"  最小: {np.min(durations_samples)} 采样点")
        print(f"  最大: {np.max(durations_samples)} 采样点")
        print(f"  平均: {np.mean(durations_samples):.1f} 采样点")
        print(f"  中位数: {np.median(durations_samples):.1f} 采样点")
        
        print(f"\n持续时间统计 (毫秒):")
        print(f"  最小: {np.min(durations_ms):.2f} ms")
        print(f"  最大: {np.max(durations_ms):.2f} ms")
        print(f"  平均: {np.mean(durations_ms):.2f} ms")
        print(f"  中位数: {np.median(durations_ms):.2f} ms")
        
        # 分类统计
        very_short = sum(1 for d in durations_ms if d < 5)
        short_count = sum(1 for d in durations_ms if 5 <= d < 10)
        medium_count = sum(1 for d in durations_ms if 10 <= d < 20)
        long_count = sum(1 for d in durations_ms if d >= 20)
        
        print(f"\n持续时间分类:")
        print(f"  < 5ms (极短):       {very_short} 个 ({very_short/len(peak_info)*100:.1f}%)")
        print(f"  5-10ms (短):        {short_count} 个 ({short_count/len(peak_info)*100:.1f}%)")
        print(f"  10-20ms (灰色地带): {medium_count} 个 ({medium_count/len(peak_info)*100:.1f}%)")
        print(f"  ≥ 20ms (真实敲击):  {long_count} 个 ({long_count/len(peak_info)*100:.1f}%)")
        
        # 显示前10个峰值的详细信息
        print(f"\n前10个峰值详细信息:")
        print(f"{'序号':<6} {'峰值位置':<12} {'幅值':<12} {'|幅值|':<12} {'起始':<10} {'结束':<10} {'采样点数':<12} {'持续时间(ms)':<15}")
        print("-" * 100)
        for i, p in enumerate(peak_info[:10]):
            print(f"{i+1:<6} {p['peak_idx']:<12} {p['peak_amplitude']:<12.4f} "
                  f"{p['peak_abs_amplitude']:<12.4f} {p['start_idx']:<10} {p['end_idx']:<10} "
                  f"{p['duration_samples']:<12} {p['duration_ms']:<15.2f}")
        
        if len(peak_info) > 10:
            print(f"... (还有 {len(peak_info)-10} 个峰值)")
        
        print()
        
        # 保存结果
        all_results[tightness] = {
            'filename': basename,
            'num_peaks': len(peak_info),
            'peaks': peak_info,
            'statistics': {
                'duration_samples_min': int(np.min(durations_samples)),
                'duration_samples_max': int(np.max(durations_samples)),
                'duration_samples_mean': float(np.mean(durations_samples)),
                'duration_samples_median': float(np.median(durations_samples)),
                'duration_ms_min': float(np.min(durations_ms)),
                'duration_ms_max': float(np.max(durations_ms)),
                'duration_ms_mean': float(np.mean(durations_ms)),
                'duration_ms_median': float(np.median(durations_ms)),
                'very_short_count': int(very_short),
                'short_count': int(short_count),
                'medium_count': int(medium_count),
                'long_count': int(long_count)
            }
        }
        
        # 绘制单个文件的直方图
        plot_individual_file_histogram(tightness, peak_info, filename)
    
    # 保存所有结果到JSON
    json_path = os.path.join(OUTPUT_DIR, "peak_duration_results_raw.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n所有结果已保存到: {json_path}")
    
    # 绘制汇总图
    if len(all_results) > 0:
        plot_peak_duration_distribution(all_results)
    
    # 打印汇总表格
    print("\n" + "="*70)
    print("汇总统计表")
    print("="*70)
    print(f"{'松紧度':<10} {'峰值数':<10} {'平均(ms)':<12} {'<5ms':<8} {'5-10ms':<10} {'10-20ms':<10} {'≥20ms':<10}")
    print("-" * 70)
    
    for tight in sorted(all_results.keys()):
        stats = all_results[tight]['statistics']
        print(f"{tight:<10} {all_results[tight]['num_peaks']:<10} "
              f"{stats['duration_ms_mean']:<12.2f} "
              f"{stats['very_short_count']:<8} {stats['short_count']:<10} "
              f"{stats['medium_count']:<10} {stats['long_count']:<10}")
    
    print("\n" + "="*70)
    print("分析完成!")
    print("="*70)
    print(f"结果已保存到目录: {OUTPUT_DIR}/")
    print("  - peak_duration_results_raw.json: 详细数据")
    print("  - peak_duration_distribution_raw.png: 汇总分布图")
    print("  - *_duration_histogram_raw.png: 各文件直方图")


if __name__ == "__main__":
    main()
