"""
槽楔模型测试数据分析 - 特征提取脚本
功能：
1. 读取不同松紧度下的采样数据
2. 时域上分离每一次敲击的信号（端点检测）
3. 选定合适的时域信号长度
4. 对选定的信号进行频域变换
5. 比较并分析不同松紧度信号的特征
6. 提取特征并保存
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
import os
import glob
import json
from tqdm import tqdm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 参数设置
SAMPLE_RATE = 51200  # 采样率 Hz
DATA_DIR = "."  # 数据目录
OUTPUT_DIR = "features"  # 输出目录

# 端点检测参数（基于包络峰值检测）
BANDPASS_LOW = 100  # 带通滤波器低频截止 (Hz)
BANDPASS_HIGH = 15000  # 带通滤波器高频截止 (Hz)
ENVELOPE_WINDOW_SIZE = 256  # 包络平滑窗口大小（采样点）
PEAK_PROMINENCE = 0.3  # 峰值显著性阈值（相对于信号最大值的比例）
PEAK_MIN_DISTANCE = 0.5  # 相邻峰值最小间隔（秒）
SEGMENT_BEFORE_PEAK = 0.01  # 峰值前截取长度（秒）
SEGMENT_AFTER_PEAK = 0.08  # 峰值后截取长度（秒）

# 信号截取参数
SIGNAL_DURATION = 0.07  # 截取的信号长度（秒）- 根据附图4选择约70ms
SIGNAL_SAMPLES = int(SIGNAL_DURATION * SAMPLE_RATE)  # 对应的采样点数


def create_output_dirs():
    """创建输出目录结构"""
    dirs = [
        OUTPUT_DIR,
        os.path.join(OUTPUT_DIR, "endpoints"),  # 端点检测图
        os.path.join(OUTPUT_DIR, "endpoints_zoom"),  # 端点检测放大图
        os.path.join(OUTPUT_DIR, "frequency"),  # 频域图
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print(f"已创建输出目录: {dirs}")


def load_data(filepath):
    """加载数据文件"""
    print(f"正在加载: {filepath}")
    data = np.loadtxt(filepath)
    print(f"  数据长度: {len(data)} 采样点, 时长: {len(data)/SAMPLE_RATE:.2f} 秒")
    return data


def compute_envelope(signal_data, window_size=ENVELOPE_WINDOW_SIZE, sample_rate=SAMPLE_RATE, 
                     low_freq=BANDPASS_LOW, high_freq=BANDPASS_HIGH):
    """
    计算信号包络
    步骤：
    1. 带通滤波（可选，去除直流和高频噪声）
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
    
    # 移动平均平滑（低通滤波）
    window = np.ones(window_size) / window_size
    envelope = np.convolve(rectified, window, mode='same')
    
    return envelope


def detect_endpoints(signal_data, sample_rate=SAMPLE_RATE):
    """
    端点检测：使用包络峰值检测敲击信号
    返回：敲击段的起始和结束位置列表 [(start1, end1), (start2, end2), ...]
    """
    from scipy.signal import find_peaks
    
    # 1. 计算包络
    envelope = compute_envelope(signal_data, sample_rate=sample_rate)
    
    # 2. 在包络上检测峰值
    max_amplitude = np.max(envelope)
    prominence_threshold = max_amplitude * PEAK_PROMINENCE
    min_distance_samples = int(PEAK_MIN_DISTANCE * sample_rate)
    
    peaks, properties = find_peaks(
        envelope,
        prominence=prominence_threshold,
        distance=min_distance_samples
    )
    
    # 3. 对每个峰值，定义信号段（峰值前后一定范围）
    segments = []
    before_samples = int(SEGMENT_BEFORE_PEAK * sample_rate)
    after_samples = int(SEGMENT_AFTER_PEAK * sample_rate)
    
    for peak in peaks:
        start = max(0, peak - before_samples)
        end = min(len(signal_data), peak + after_samples)
        segments.append((start, end))
    
    return segments


def plot_endpoint_detection(signal_data, segments, filename, sample_rate=SAMPLE_RATE):
    """绘制端点检测效果图（包含包络和峰值标记）"""
    time = np.arange(len(signal_data)) / sample_rate
    
    # 计算包络用于显示
    envelope = compute_envelope(signal_data, sample_rate=sample_rate)
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    
    # 上图：原始信号 + 端点标记
    axes[0].plot(time, signal_data, 'b-', linewidth=0.5, alpha=0.7, label='原始信号')
    for i, (start, end) in enumerate(segments):
        start_time = start / sample_rate
        end_time = end / sample_rate
        peak_time = start_time + SEGMENT_BEFORE_PEAK  # 峰值位置
        axes[0].axvline(x=peak_time, color='r', linestyle='-', linewidth=1.5, alpha=0.8)
        axes[0].axvline(x=start_time, color='g', linestyle='--', linewidth=1, alpha=0.5)
        axes[0].axvline(x=end_time, color='g', linestyle='--', linewidth=1, alpha=0.5)
    
    axes[0].set_ylabel('幅值')
    axes[0].set_title('原始信号与检测到的敲击峰值（红线）')
    axes[0].set_ylim([-1, 1])
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # 下图：包络 + 峰值标记
    axes[1].plot(time, envelope, 'b-', linewidth=1, label='包络')
    for i, (start, end) in enumerate(segments):
        peak_time = start / sample_rate + SEGMENT_BEFORE_PEAK
        peak_idx = int(peak_time * sample_rate)
        if 0 <= peak_idx < len(envelope):
            axes[1].plot(peak_time, envelope[peak_idx], 'ro', markersize=8)
    
    axes[1].set_xlabel('时间/s')
    axes[1].set_ylabel('包络幅值')
    axes[1].set_title('信号包络与检测到的峰值（红点）')
    axes[1].set_xlim([0, time[-1]])
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(OUTPUT_DIR, "endpoints", f"{filename}_endpoints.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  端点检测图已保存: {save_path}")
    
    return save_path


def plot_endpoint_zoom(signal_data, segments, filename, segment_idx=0, sample_rate=SAMPLE_RATE):
    """绘制端点检测放大图（显示某一个敲击信号，包含包络和峰值对齐）"""
    if len(segments) == 0:
        return
    
    if segment_idx >= len(segments):
        segment_idx = len(segments) // 2  # 选择中间的一个段
    
    start, end = segments[segment_idx]
    peak_sample = start + int(SEGMENT_BEFORE_PEAK * sample_rate)
    
    # 扩展显示范围
    extend_samples = int(0.3 * sample_rate)
    plot_start = max(0, start - extend_samples)
    plot_end = min(len(signal_data), end + extend_samples)
    
    time = np.arange(plot_start, plot_end) / sample_rate
    signal_segment = signal_data[plot_start:plot_end]
    
    # 计算该段的包络
    envelope_full = compute_envelope(signal_data, sample_rate=sample_rate)
    envelope_segment = envelope_full[plot_start:plot_end]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 绘制信号和包络
    ax.plot(time, signal_segment, 'b-', linewidth=0.8, alpha=0.7, label='信号')
    ax.plot(time, envelope_segment, 'orange', linewidth=2, label='包络')
    
    # 绘制端点线和峰值
    peak_time = peak_sample / sample_rate
    ax.axvline(x=start/sample_rate, color='g', linestyle='--', linewidth=1.5, label='起点', alpha=0.7)
    ax.axvline(x=end/sample_rate, color='g', linestyle='--', linewidth=1.5, label='终点', alpha=0.7)
    ax.axvline(x=peak_time, color='r', linestyle='-', linewidth=2, label='峰值中心', alpha=0.9)
    
    # 标记峰值点
    if plot_start <= peak_sample < plot_end:
        peak_idx_local = peak_sample - plot_start
        ax.plot(peak_time, envelope_segment[peak_idx_local], 'ro', markersize=12, zorder=5)
    
    ax.set_xlabel('时间/s')
    ax.set_ylabel('幅值')
    ax.set_title(f'端点检测放大图（第 {segment_idx+1} 个敲击信号）')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 保存图片
    save_path = os.path.join(OUTPUT_DIR, "endpoints_zoom", f"{filename}_zoom.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  端点放大图已保存: {save_path}")
    
    return save_path


def extract_signal_segments(signal_data, segments, target_length=SIGNAL_SAMPLES):
    """
    从检测到的敲击段中提取固定长度的信号
    以峰值为参考点，向前取 SEGMENT_BEFORE_PEAK，向后取剩余部分至 target_length
    选择信号长度的原因：
    1. 根据频域分析，敲击信号的主要能量集中在峰值前后约70ms
    2. 固定长度便于后续的频域分析和特征提取
    3. 以峰值对齐确保每个样本捕获的是冲击主能量部分
    """
    extracted = []
    
    for start, end in segments:
        # 峰值位置在 start + SEGMENT_BEFORE_PEAK
        peak_sample = start + int(SEGMENT_BEFORE_PEAK * SAMPLE_RATE)
        
        # 以峰值为中心，截取固定长度
        # 为保持一致性，从start开始截取target_length
        segment_start = start
        segment_end = segment_start + target_length
        
        # 确保不越界
        if segment_end <= len(signal_data):
            segment = signal_data[segment_start:segment_end]
            extracted.append(segment)
        elif segment_start < len(signal_data):
            # 如果剩余长度不足，补零
            segment = signal_data[segment_start:]
            segment = np.pad(segment, (0, target_length - len(segment)), 'constant')
            extracted.append(segment)
    
    return np.array(extracted)


def compute_frequency_spectrum(signal_segment, sample_rate=SAMPLE_RATE):
    """计算信号的频谱"""
    n = len(signal_segment)
    
    # 应用汉宁窗减少频谱泄漏
    window = np.hanning(n)
    windowed_signal = signal_segment * window
    
    # FFT
    spectrum = fft(windowed_signal)
    frequencies = fftfreq(n, 1/sample_rate)
    
    # 取正频率部分
    positive_freq_idx = frequencies >= 0
    frequencies = frequencies[positive_freq_idx]
    magnitude = np.abs(spectrum[positive_freq_idx]) * 2 / n
    
    return frequencies, magnitude


def plot_time_frequency(signal_segment, filename, segment_idx=0, sample_rate=SAMPLE_RATE):
    """绘制时域图和频域图"""
    time = np.arange(len(signal_segment)) / sample_rate
    frequencies, magnitude = compute_frequency_spectrum(signal_segment, sample_rate)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # 时域图
    axes[0].plot(time, signal_segment, 'b-', linewidth=0.8)
    axes[0].set_xlabel('时间/s')
    axes[0].set_ylabel('幅度')
    axes[0].set_title('时域图')
    axes[0].grid(True, alpha=0.3)
    
    # 频域图
    axes[1].plot(frequencies, magnitude, 'b-', linewidth=0.8)
    axes[1].set_xlabel('频率/Hz')
    axes[1].set_ylabel('幅度')
    axes[1].set_title('频域图')
    axes[1].set_xlim([0, sample_rate/2])  # 到奈奎斯特频率
    axes[1].grid(True, alpha=0.3)
    
    # 设置x轴刻度使用科学计数法
    axes[1].ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(OUTPUT_DIR, "frequency", f"{filename}_freq_{segment_idx}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  时频图已保存: {save_path}")
    
    return save_path


def extract_features(signal_segments, sample_rate=SAMPLE_RATE):
    """
    提取频域特征
    特征包括：
    1. 主频率（最大幅值对应的频率）
    2. 频谱质心
    3. 频谱带宽
    4. 多个频段的能量比
    5. 频谱峰值
    """
    features_list = []
    
    for segment in signal_segments:
        frequencies, magnitude = compute_frequency_spectrum(segment, sample_rate)
        
        # 1. 主频率
        main_freq_idx = np.argmax(magnitude)
        main_frequency = frequencies[main_freq_idx]
        
        # 2. 频谱质心
        if np.sum(magnitude) > 0:
            spectral_centroid = np.sum(frequencies * magnitude) / np.sum(magnitude)
        else:
            spectral_centroid = 0
        
        # 3. 频谱带宽
        if np.sum(magnitude) > 0:
            spectral_bandwidth = np.sqrt(np.sum(((frequencies - spectral_centroid) ** 2) * magnitude) / np.sum(magnitude))
        else:
            spectral_bandwidth = 0
        
        # 4. 频段能量比
        total_energy = np.sum(magnitude ** 2)
        
        # 定义频段 (Hz)
        bands = [(0, 2000), (2000, 5000), (5000, 10000), (10000, 15000), (15000, 25600)]
        band_energies = []
        
        for low, high in bands:
            band_mask = (frequencies >= low) & (frequencies < high)
            band_energy = np.sum(magnitude[band_mask] ** 2)
            band_ratio = band_energy / total_energy if total_energy > 0 else 0
            band_energies.append(band_ratio)
        
        # 5. 频谱峰值（前5个峰值的频率）
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(magnitude, height=np.max(magnitude) * 0.1, distance=50)
        
        # 按幅值排序取前5个
        if len(peaks) > 0:
            peak_magnitudes = magnitude[peaks]
            sorted_indices = np.argsort(peak_magnitudes)[::-1][:5]
            top_peaks = peaks[sorted_indices]
            peak_frequencies = frequencies[top_peaks].tolist()
            # 补齐到5个
            while len(peak_frequencies) < 5:
                peak_frequencies.append(0)
        else:
            peak_frequencies = [0, 0, 0, 0, 0]
        
        # 6. 时域特征
        rms = np.sqrt(np.mean(segment ** 2))
        peak_amplitude = np.max(np.abs(segment))
        zero_crossing_rate = np.sum(np.diff(np.sign(segment)) != 0) / len(segment)
        
        # 组合特征
        feature = {
            'main_frequency': main_frequency,
            'spectral_centroid': spectral_centroid,
            'spectral_bandwidth': spectral_bandwidth,
            'band_energies': band_energies,
            'peak_frequencies': peak_frequencies,
            'rms': rms,
            'peak_amplitude': peak_amplitude,
            'zero_crossing_rate': zero_crossing_rate
        }
        
        features_list.append(feature)
    
    return features_list


def features_to_array(features_list):
    """将特征列表转换为numpy数组"""
    feature_vectors = []
    
    for f in features_list:
        vector = [
            f['main_frequency'],
            f['spectral_centroid'],
            f['spectral_bandwidth'],
            *f['band_energies'],
            *f['peak_frequencies'],
            f['rms'],
            f['peak_amplitude'],
            f['zero_crossing_rate']
        ]
        feature_vectors.append(vector)
    
    return np.array(feature_vectors)


def process_file(filepath):
    """处理单个数据文件"""
    # 获取文件名和松紧度值
    basename = os.path.basename(filepath)
    filename = os.path.splitext(basename)[0]
    
    # 从文件名提取松紧度值（如 acquisitionData-400 -> 400）
    tightness = int(filename.split('-')[1])
    
    print(f"\n{'='*60}")
    print(f"处理文件: {basename}, 松紧度: {tightness}")
    print(f"{'='*60}")
    
    # 1. 加载数据
    signal_data = load_data(filepath)
    
    # 2. 端点检测
    print("正在进行端点检测...")
    segments = detect_endpoints(signal_data)
    print(f"  检测到 {len(segments)} 个敲击信号")
    
    # 3. 绘制端点检测效果图
    plot_endpoint_detection(signal_data, segments, filename)
    
    # 4. 绘制端点检测放大图
    if len(segments) > 0:
        plot_endpoint_zoom(signal_data, segments, filename, segment_idx=len(segments)//2)
    
    # 5. 提取固定长度的信号段
    print(f"正在提取信号段 (每段长度: {SIGNAL_DURATION*1000:.0f}ms, {SIGNAL_SAMPLES} 采样点)...")
    signal_segments = extract_signal_segments(signal_data, segments)
    print(f"  成功提取 {len(signal_segments)} 个信号段")
    
    # 6. 绘制时频图（第一个信号段）
    if len(signal_segments) > 0:
        plot_time_frequency(signal_segments[0], filename, segment_idx=0)
    
    # 7. 提取特征
    print("正在提取特征...")
    features = extract_features(signal_segments)
    feature_array = features_to_array(features)
    print(f"  特征维度: {feature_array.shape}")
    
    return {
        'tightness': tightness,
        'n_segments': len(signal_segments),
        'features': feature_array,
        'segments': signal_segments
    }


def analyze_tightness_frequency_relationship(all_results):
    """分析松紧度与频率的关系"""
    print("\n" + "="*60)
    print("松紧度与频率关系分析")
    print("="*60)
    
    tightness_values = []
    main_frequencies = []
    spectral_centroids = []
    
    for result in all_results:
        tightness = result['tightness']
        features = result['features']
        
        if len(features) > 0:
            avg_main_freq = np.mean(features[:, 0])  # 主频率
            avg_centroid = np.mean(features[:, 1])   # 频谱质心
            
            tightness_values.append(tightness)
            main_frequencies.append(avg_main_freq)
            spectral_centroids.append(avg_centroid)
            
            print(f"松紧度 {tightness}: 平均主频率 = {avg_main_freq:.2f} Hz, 平均频谱质心 = {avg_centroid:.2f} Hz")
    
    # 绘制关系图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(tightness_values, main_frequencies, 'bo-', markersize=8, linewidth=2)
    axes[0].set_xlabel('松紧度（压力值）')
    axes[0].set_ylabel('平均主频率 (Hz)')
    axes[0].set_title('松紧度与主频率的关系')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(tightness_values, spectral_centroids, 'ro-', markersize=8, linewidth=2)
    axes[1].set_xlabel('松紧度（压力值）')
    axes[1].set_ylabel('平均频谱质心 (Hz)')
    axes[1].set_title('松紧度与频谱质心的关系')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, "tightness_frequency_relationship.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n关系图已保存: {save_path}")
    
    return {
        'tightness': tightness_values,
        'main_frequencies': main_frequencies,
        'spectral_centroids': spectral_centroids
    }


def main():
    """主函数"""
    print("="*60)
    print("槽楔模型测试数据分析 - 特征提取")
    print("="*60)
    
    # 创建输出目录
    create_output_dirs()
    
    # 获取所有数据文件
    data_files = sorted(glob.glob(os.path.join(DATA_DIR, "acquisitionData-*.txt")))
    print(f"\n找到 {len(data_files)} 个数据文件")
    
    if len(data_files) == 0:
        print("错误: 未找到数据文件!")
        return
    
    # 处理所有文件
    all_results = []
    all_features = []
    all_labels = []
    
    for filepath in data_files:
        result = process_file(filepath)
        all_results.append(result)
        
        # 收集特征和标签
        if len(result['features']) > 0:
            all_features.append(result['features'])
            all_labels.extend([result['tightness']] * len(result['features']))
    
    # 合并所有特征
    X = np.vstack(all_features)
    y = np.array(all_labels)
    
    print(f"\n{'='*60}")
    print("特征提取完成")
    print(f"{'='*60}")
    print(f"总样本数: {len(y)}")
    print(f"特征维度: {X.shape[1]}")
    print(f"类别数: {len(np.unique(y))}")
    print(f"类别分布: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    # 分析松紧度与频率的关系
    relationship = analyze_tightness_frequency_relationship(all_results)
    
    # 保存特征和标签
    np.save(os.path.join(OUTPUT_DIR, "features.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "labels.npy"), y)
    
    # 保存特征名称
    feature_names = [
        'main_frequency', 'spectral_centroid', 'spectral_bandwidth',
        'band_energy_0_2k', 'band_energy_2k_5k', 'band_energy_5k_10k',
        'band_energy_10k_15k', 'band_energy_15k_25.6k',
        'peak_freq_1', 'peak_freq_2', 'peak_freq_3', 'peak_freq_4', 'peak_freq_5',
        'rms', 'peak_amplitude', 'zero_crossing_rate'
    ]
    
    with open(os.path.join(OUTPUT_DIR, "feature_names.json"), 'w', encoding='utf-8') as f:
        json.dump(feature_names, f, ensure_ascii=False, indent=2)
    
    # 保存关系数据
    with open(os.path.join(OUTPUT_DIR, "tightness_frequency_relationship.json"), 'w', encoding='utf-8') as f:
        json.dump(relationship, f, ensure_ascii=False, indent=2)
    
    print(f"\n特征已保存到: {OUTPUT_DIR}/")
    print("  - features.npy: 特征矩阵")
    print("  - labels.npy: 标签")
    print("  - feature_names.json: 特征名称")
    print("  - tightness_frequency_relationship.json: 松紧度与频率关系")
    
    print("\n" + "="*60)
    print("信号长度选择说明")
    print("="*60)
    print(f"选择 {SIGNAL_DURATION*1000:.0f}ms ({SIGNAL_SAMPLES} 采样点) 作为信号长度的原因:")
    print("1. 敲击信号的主要能量集中在冲击后的前70ms内")
    print("2. 该长度能完整捕获冲击响应和初始衰减过程")
    print("3. 固定长度便于FFT分析和特征提取的一致性")
    print("4. 避免包含过多静音部分导致的噪声干扰")


if __name__ == "__main__":
    main()
