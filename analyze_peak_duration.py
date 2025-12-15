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
PEAK_THRESHOLD_RATIO = 0.3  # 峰值阈值：信号绝对值的百分位数（如0.3表示取前30%的大值）
PEAK_MIN_DISTANCE = 256  # 相邻峰值最小间隔（采样点数）

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


def compute_envelope(signal_data, sample_rate=SAMPLE_RATE, 
                     low_freq=BANDPASS_LOW, high_freq=BANDPASS_HIGH):
    """
    计算信号包络
    步骤：
    1. 带通滤波（去除直流和高频噪声）
    2. 整流（取绝对值）
    3. 低通滤波平滑（移动平均）
    """
    # 带通滤波
    nyquist = sample_rate / 2
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    filtered_signal = signal.filtfilt(b, a, signal_data)
    
    # 整流
    rectified = np.abs(filtered_signal)
    
    # 移动平均平滑
    window_size = 512
    window = np.ones(window_size) / window_size
    envelope = np.convolve(rectified, window, mode='same')
    
    return envelope


def analyze_peak_duration(signal_data, sample_rate=SAMPLE_RATE):
    """
    分析每个孤立峰的持续时间
    
    返回：
        peak_info: 列表，每个元素是字典 {
            'peak_idx': 峰值位置索引,
            'peak_amplitude': 峰值幅值,
            'start_idx': 起始位置,
            'end_idx': 结束位置,
            'duration_samples': 持续采样点数,
            'duration_ms': 持续时间(毫秒)
        }
    """
    # 计算包络
    envelope = compute_envelope(signal_data, sample_rate=sample_rate)
    
    # 检测峰值
    min_distance_samples = int(PEAK_MIN_DISTANCE * sample_rate)
    peaks, properties = find_peaks(
        envelope,
        height=MIN_PEAK_AMPLITUDE,
        distance=min_distance_samples
    )
    
    if len(peaks) == 0:
        return []
    
    peak_info = []
    
    # 对每个峰值计算持续时间
    for i, peak_idx in enumerate(peaks):
        peak_value = envelope[peak_idx]
        threshold = peak_value * DURATION_THRESHOLD_RATIO
        
        # === 向前搜索起点 ===
        start_idx = peak_idx
        for idx in range(peak_idx - 1, -1, -1):
            if envelope[idx] >= threshold:
                start_idx = idx
            else:
                break
        
        # === 向后搜索终点 ===
        end_idx = peak_idx
        for idx in range(peak_idx + 1, len(envelope)):
            if envelope[idx] >= threshold:
                end_idx = idx
            else:
                break
        
        # 计算持续时间
        duration_samples = end_idx - start_idx + 1
        duration_ms = duration_samples / sample_rate * 1000
        
        peak_info.append({
            'peak_idx': int(peak_idx),
            'peak_amplitude': float(peak_value),
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
    
    bp = axes[0].boxplot(durations_by_file, labels=labels, patch_artist=True)
    
    # 设置颜色
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    
    axes[0].axhline(y=10, color='orange', linestyle='--', linewidth=1.5, label='10ms (毛刺阈值)')
    axes[0].axhline(y=20, color='green', linestyle='--', linewidth=1.5, label='20ms (真实敲击阈值)')
    axes[0].set_xlabel('松紧度')
    axes[0].set_ylabel('持续时间 (ms)')
    axes[0].set_title('各文件峰值持续时间分布（箱线图）')
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].legend()
    
    # === 子图2: 持续时间统计柱状图 ===
    categories = ['<10ms\n(毛刺)', '10-20ms\n(灰色地带)', '≥20ms\n(真实敲击)']
    width = 0.15
    x = np.arange(len(tightness_values))
    
    short_counts = []
    medium_counts = []
    long_counts = []
    
    for tight in tightness_values:
        durations = [p['duration_ms'] for p in all_results[tight]['peaks']]
        short_counts.append(sum(1 for d in durations if d < 10))
        medium_counts.append(sum(1 for d in durations if 10 <= d < 20))
        long_counts.append(sum(1 for d in durations if d >= 20))
    
    axes[1].bar(x - width, short_counts, width, label='<10ms (毛刺)', color='red', alpha=0.7)
    axes[1].bar(x, medium_counts, width, label='10-20ms (灰色地带)', color='orange', alpha=0.7)
    axes[1].bar(x + width, long_counts, width, label='≥20ms (真实敲击)', color='green', alpha=0.7)
    
    axes[1].set_xlabel('松紧度')
    axes[1].set_ylabel('峰值数量')
    axes[1].set_title('各文件峰值持续时间分类统计')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "peak_duration_distribution.png")
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
    ax.axvline(x=10, color='orange', linestyle='--', linewidth=2, label='10ms (毛刺阈值)')
    ax.axvline(x=20, color='green', linestyle='--', linewidth=2, label='20ms (真实敲击阈值)')
    
    ax.set_xlabel('持续时间 (ms)')
    ax.set_ylabel('峰值数量')
    ax.set_title(f'松紧度 {tightness} - 峰值持续时间分布直方图')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加统计信息文本
    stats_text = f"总峰值数: {len(durations)}\n"
    stats_text += f"最小: {np.min(durations):.1f} ms\n"
    stats_text += f"最大: {np.max(durations):.1f} ms\n"
    stats_text += f"平均: {np.mean(durations):.1f} ms\n"
    stats_text += f"中位数: {np.median(durations):.1f} ms"
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f"{filename}_duration_histogram.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """主函数"""
    print("="*70)
    print("孤立尖峰持续时间分析")
    print("="*70)
    print(f"采样率: {SAMPLE_RATE} Hz")
    print(f"最小峰值幅值: {MIN_PEAK_AMPLITUDE}")
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
        
        # 分析峰值持续时间
        peak_info = analyze_peak_duration(signal_data)
        
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
        short_count = sum(1 for d in durations_ms if d < 10)
        medium_count = sum(1 for d in durations_ms if 10 <= d < 20)
        long_count = sum(1 for d in durations_ms if d >= 20)
        
        print(f"\n持续时间分类:")
        print(f"  < 10ms (疑似毛刺):  {short_count} 个 ({short_count/len(peak_info)*100:.1f}%)")
        print(f"  10-20ms (灰色地带): {medium_count} 个 ({medium_count/len(peak_info)*100:.1f}%)")
        print(f"  ≥ 20ms (真实敲击):  {long_count} 个 ({long_count/len(peak_info)*100:.1f}%)")
        
        # 显示前10个峰值的详细信息
        print(f"\n前10个峰值详细信息:")
        print(f"{'序号':<6} {'峰值位置':<12} {'幅值':<10} {'起始':<10} {'结束':<10} {'采样点数':<12} {'持续时间(ms)':<15}")
        print("-" * 85)
        for i, p in enumerate(peak_info[:10]):
            print(f"{i+1:<6} {p['peak_idx']:<12} {p['peak_amplitude']:<10.4f} "
                  f"{p['start_idx']:<10} {p['end_idx']:<10} {p['duration_samples']:<12} "
                  f"{p['duration_ms']:<15.2f}")
        
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
                'short_count': int(short_count),
                'medium_count': int(medium_count),
                'long_count': int(long_count)
            }
        }
        
        # 绘制单个文件的直方图
        plot_individual_file_histogram(tightness, peak_info, filename)
    
    # 保存所有结果到JSON
    json_path = os.path.join(OUTPUT_DIR, "peak_duration_results.json")
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
    print(f"{'松紧度':<10} {'峰值数':<10} {'平均持续(ms)':<15} {'<10ms':<10} {'10-20ms':<10} {'≥20ms':<10}")
    print("-" * 70)
    
    for tight in sorted(all_results.keys()):
        stats = all_results[tight]['statistics']
        print(f"{tight:<10} {all_results[tight]['num_peaks']:<10} "
              f"{stats['duration_ms_mean']:<15.2f} "
              f"{stats['short_count']:<10} {stats['medium_count']:<10} {stats['long_count']:<10}")
    
    print("\n" + "="*70)
    print("分析完成!")
    print("="*70)
    print(f"结果已保存到目录: {OUTPUT_DIR}/")
    print("  - peak_duration_results.json: 详细数据")
    print("  - peak_duration_distribution.png: 汇总分布图")
    print("  - *_duration_histogram.png: 各文件直方图")


if __name__ == "__main__":
    main()
