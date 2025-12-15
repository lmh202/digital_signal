"""
槽楔模型测试数据分析 - 特征提取脚本
功能：# 端点检测参数（基于阈值回落+最大事件时长）
BANDPASS_LOW = 100  # 带通滤波器低频截止 (Hz)
BANDPASS_HIGH = 15000  # 带通滤波器高频截止 (Hz)
ENVELOPE_WINDOW_SIZE = 128  # 包络平滑窗口大小（采样点）
PEAK_PROMINENCE = 0.3  # 峰值显著性阈值（相对于信号最大值的比例）
PEAK_MIN_DISTANCE = 0.1  # 相邻峰值最小间隔（秒）
PEAK_DECAY_THRESHOLD = 0.1  # 峰值衰减阈值（相对于该峰峰值的比例）- 主要切分依据
MAX_EVENT_DURATION = 0.05  # 最大事件时长（秒）- 每次敲击振铃最长50ms
MIN_PEAK_SPACING = 0.02  # 最小峰间距（秒）- 当两峰小于此值时触发防跨峰机制松紧度下的采样数据
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

# 端点检测参数（基于双门限阈值法 - 改进版）
BANDPASS_LOW = 100  # 带通滤波器低频截止 (Hz)
BANDPASS_HIGH = 15000  # 带通滤波器高频截止 (Hz)

# 双门限法参数（改进：直接在包络上检测峰值）
MIN_PEAK_AMPLITUDE = 0.18  # 候选峰的最小包络幅值（绝对值）- 关键参数
PEAK_HIGH_THRESHOLD_RATIO = 0.60  # 高门限：相对于峰值的比例（用于边界扩展）
PEAK_LOW_THRESHOLD_RATIO = 0.30  # 低门限：相对于峰值的比例（用于边界扩展）
PEAK_MIN_DISTANCE = 0.05  # 相邻峰值最小间隔（秒）
MAX_EVENT_DURATION = 0.08  # 最大事件时长（秒）
MIN_EVENT_DURATION = 0.010  # 最小事件时长（秒）- 用于过滤毛刺噪声（< 10ms认为是噪声）
MIN_SEGMENT_SAMPLES = 256  # 最小段长度（采样点），避免切得太短

# 信号截取参数
SIGNAL_DURATION = 0.02  # 截取的信号长度（秒）
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


def compute_envelope(signal_data, sample_rate=SAMPLE_RATE, 
                     low_freq=BANDPASS_LOW, high_freq=BANDPASS_HIGH):
    """
    计算信号包络（用于可视化）
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
    
    # 移动平均平滑（使用固定窗口512）
    window_size = 512
    window = np.ones(window_size) / window_size
    envelope = np.convolve(rectified, window, mode='same')
    
    return envelope


def detect_endpoints(signal_data, sample_rate=SAMPLE_RATE):
    """
    端点检测：双门限阈值法（改进版 - 直接在包络上检测峰值）
    
    算法流程：
        1. 计算信号包络
        2. 使用find_peaks在包络上直接检测峰值，要求峰值幅值 > MIN_PEAK_AMPLITUDE
        3. 对每个峰值使用双门限扩展边界：
           - 高门限 = 峰值 × PEAK_HIGH_THRESHOLD_RATIO（如0.6）
           - 低门限 = 峰值 × PEAK_LOW_THRESHOLD_RATIO（如0.3）
        4. 从峰值向前后扩展到低于低门限的位置
    
    参数:
        signal_data: 原始信号数据 (1D numpy array)
        sample_rate: 采样率 (Hz)
    
    返回:
        segments: 列表 [(start1, end1, peak1), (start2, end2, peak2), ...]
                 start/end 是样本索引，peak 是峰值位置索引
    
    外露参数（在文件顶部定义）:
        MIN_PEAK_AMPLITUDE: 候选峰的最小包络幅值
        PEAK_HIGH_THRESHOLD_RATIO: 高门限比例
        PEAK_LOW_THRESHOLD_RATIO: 低门限比例
        PEAK_MIN_DISTANCE: 相邻峰值最小间隔
        MAX_EVENT_DURATION: 最大事件时长
    """
    from scipy.signal import find_peaks
    
    # 步骤1: 计算包络
    envelope = compute_envelope(signal_data, sample_rate=sample_rate)
    
    # 步骤2: 在包络上检测峰值（必须满足最小幅值要求）
    min_distance_samples = int(PEAK_MIN_DISTANCE * sample_rate)
    
    peaks, properties = find_peaks(
        envelope,
        height=MIN_PEAK_AMPLITUDE,  # 关键：峰值必须大于0.18
        distance=min_distance_samples
    )
    
    if len(peaks) == 0:
        print("  ⚠️  警告：未检测到任何峰值！")
        return []
    
    # 打印检测到的峰值幅值（用于调试）
    peak_amplitudes = envelope[peaks]
    print(f"  📊 检测到 {len(peaks)} 个峰值")
    print(f"  📈 峰值幅值范围: 最小={np.min(peak_amplitudes):.4f}, 最大={np.max(peak_amplitudes):.4f}, 平均={np.mean(peak_amplitudes):.4f}")
    print(f"  🔍 所有峰值幅值: {[f'{a:.4f}' for a in peak_amplitudes[:10]]}{'...' if len(peak_amplitudes) > 10 else ''}")
    
    # 步骤3: 对每个峰值使用双门限确定起点和终点
    max_event_samples = int(MAX_EVENT_DURATION * sample_rate)
    min_event_samples = int(MIN_EVENT_DURATION * sample_rate)
    segments = []
    rejected_segments = []  # 记录被拒绝的段（用于统计）
    
    for i, peak_idx in enumerate(peaks):
        peak_value = envelope[peak_idx]
        
        # 计算该峰的高低门限
        high_threshold = peak_value * PEAK_HIGH_THRESHOLD_RATIO
        low_threshold = peak_value * PEAK_LOW_THRESHOLD_RATIO
        
        # === 确定起点：从峰向前找到低于低门限的位置 ===
        search_start = max(0, peak_idx - max_event_samples)
        if i > 0:
            search_start = max(search_start, peaks[i-1])  # 不早于上一个峰
        
        start_idx = search_start
        for idx in range(peak_idx - 1, search_start - 1, -1):
            if envelope[idx] >= low_threshold:
                start_idx = idx
            else:
                break
        
        # === 确定终点：从峰向后找到低于低门限的位置 ===
        search_end = min(len(envelope) - 1, peak_idx + max_event_samples)
        if i < len(peaks) - 1:
            search_end = min(search_end, peaks[i+1])  # 不晚于下一个峰
        
        end_idx = search_end
        for idx in range(peak_idx + 1, search_end + 1):
            if envelope[idx] >= low_threshold:
                end_idx = idx
            else:
                break
        
        # === 合法性检查 ===
        if end_idx <= start_idx:
            end_idx = min(start_idx + max_event_samples // 2, len(envelope) - 1)
        
        # 确保不越界
        start_idx = max(0, start_idx)
        end_idx = min(len(envelope) - 1, end_idx)
        
        # 计算持续时间（毫秒）
        duration_samples = end_idx - start_idx
        duration_ms = duration_samples / sample_rate * 1000
        
        # 检查最小长度
        if duration_samples < MIN_SEGMENT_SAMPLES:
            rejected_segments.append({
                'peak_idx': peak_idx,
                'peak_amplitude': peak_value,
                'duration_ms': duration_ms,
                'reason': '段长度不足'
            })
            continue
        
        # 【关键验证】检查最小持续时间（过滤毛刺噪声）
        if duration_samples < min_event_samples:
            rejected_segments.append({
                'peak_idx': peak_idx,
                'peak_amplitude': peak_value,
                'duration_ms': duration_ms,
                'reason': f'持续时间过短(<{MIN_EVENT_DURATION*1000:.0f}ms,疑似毛刺)'
            })
            continue
        
        segments.append((start_idx, end_idx, peak_idx))
    
    # 打印详细统计信息
    print(f"  ✅ 成功提取 {len(segments)} 个有效段")
    if len(rejected_segments) > 0:
        print(f"  ⚠️  拒绝 {len(rejected_segments)} 个可疑段:")
        for seg in rejected_segments[:5]:  # 只显示前5个
            print(f"     - 峰值幅值={seg['peak_amplitude']:.4f}, 持续时间={seg['duration_ms']:.1f}ms, 原因: {seg['reason']}")
        if len(rejected_segments) > 5:
            print(f"     ... (还有 {len(rejected_segments)-5} 个被拒绝)")
    
    # 打印有效段的持续时间统计
    if len(segments) > 0:
        durations_ms = [(end - start) / sample_rate * 1000 for start, end, _ in segments]
        print(f"  📏 有效段持续时间: 最小={np.min(durations_ms):.1f}ms, 最大={np.max(durations_ms):.1f}ms, 平均={np.mean(durations_ms):.1f}ms")
        
        # 统计持续时间分布
        short_count = sum(1 for d in durations_ms if d < 20)
        medium_count = sum(1 for d in durations_ms if 20 <= d < 50)
        long_count = sum(1 for d in durations_ms if d >= 50)
        print(f"  📊 持续时间分布: <20ms({short_count}个), 20-50ms({medium_count}个), ≥50ms({long_count}个)")
    
    print()  # 空行
    
    return segments


def plot_endpoint_detection(signal_data, segments, filename, sample_rate=SAMPLE_RATE):
    """绘制端点检测效果图（包含包络、峰值和谷底切分标记）"""
    time = np.arange(len(signal_data)) / sample_rate
    
    # 计算包络用于显示
    envelope = compute_envelope(signal_data, sample_rate=sample_rate)
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    
    # 上图：原始信号 + 端点标记
    axes[0].plot(time, signal_data, 'b-', linewidth=0.5, alpha=0.7, label='原始信号')
    for i, (start, end, peak) in enumerate(segments):
        start_time = start / sample_rate
        end_time = end / sample_rate
        peak_time = peak / sample_rate
        axes[0].axvline(x=peak_time, color='r', linestyle='-', linewidth=1.0, alpha=0.8)
        axes[0].axvline(x=start_time, color='g', linestyle='--', linewidth=0.5, alpha=0.5)
        axes[0].axvline(x=end_time, color='g', linestyle='--', linewidth=0.5, alpha=0.5)
    
    axes[0].set_ylabel('幅值')
    axes[0].set_title('原始信号与检测到的敲击（红线=峰值，绿虚线=段边界）')
    axes[0].set_ylim([-1, 1])
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # 下图：包络 + 峰值和切分点标记
    axes[1].plot(time, envelope, 'b-', linewidth=1, label='包络')
    for i, (start, end, peak) in enumerate(segments):
        peak_time = peak / sample_rate
        start_time = start / sample_rate
        end_time = end / sample_rate
        
        # 标记峰值
        if 0 <= peak < len(envelope):
            axes[1].plot(peak_time, envelope[peak], 'ro', markersize=8, zorder=5)
        
        # 标记谷底切分点
        if 0 <= start < len(envelope):
            axes[1].plot(start_time, envelope[start], 'gs', markersize=6, alpha=0.7)
        if 0 <= end < len(envelope):
            axes[1].plot(end_time, envelope[end], 'gs', markersize=6, alpha=0.7)
    
    axes[1].set_xlabel('时间/s')
    axes[1].set_ylabel('包络幅值')
    axes[1].set_title('信号包络（红圆=峰值，绿方=段边界）')
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
    """绘制端点检测放大图（显示某一个敲击信号，包含包络、峰值和谷底）"""
    if len(segments) == 0:
        return
    
    if segment_idx >= len(segments):
        segment_idx = len(segments) // 2  # 选择中间的一个段
    
    start, end, peak = segments[segment_idx]
    
    # 目标：在放大图中同时显示上一个峰、当前峰和下一个峰的截取段
    # 使用与 extract_signal_segments 相同的对齐规则来计算理想截取窗口
    before_peak = SIGNAL_SAMPLES // 4
    after_peak = SIGNAL_SAMPLES - before_peak

    # 当前峰的理想截取范围
    ideal_start = peak - before_peak
    ideal_end = peak + after_peak

    # 上一个峰的理想范围（若存在）
    if segment_idx > 0:
        prev_peak = segments[segment_idx - 1][2]
        prev_ideal_start = prev_peak - before_peak
        prev_ideal_end = prev_peak + after_peak
    else:
        prev_ideal_start = ideal_start
        prev_ideal_end = ideal_start

    # 下一个峰的理想范围（若存在）
    if segment_idx < len(segments) - 1:
        next_peak = segments[segment_idx + 1][2]
        next_ideal_start = next_peak - before_peak
        next_ideal_end = next_peak + after_peak
    else:
        next_ideal_start = ideal_end
        next_ideal_end = ideal_end

    # 计算整体显示范围：从上一个理想起点到下一个理想终点，并留少量边距
    margin = int(0.01 * sample_rate)  # 10ms 边距
    plot_start = max(0, min(prev_ideal_start, ideal_start, next_ideal_start) - margin)
    plot_end = min(len(signal_data), max(prev_ideal_end, ideal_end, next_ideal_end) + margin)

    time = np.arange(plot_start, plot_end) / sample_rate
    signal_segment = signal_data[plot_start:plot_end]

    fig, ax = plt.subplots(figsize=(12, 6))

    # 仅绘制原始信号（无需包络），并用虚线标注截取窗口边界
    ax.plot(time, signal_segment, 'b-', linewidth=0.8, alpha=0.9, label='信号')

    # 绘制上/当前/下的截取窗口边界（虚线）并用图例标注
    def mark_window_lines(s_idx, e_idx, color, label=None, linestyle='--'):
        s_time = s_idx / sample_rate
        e_time = e_idx / sample_rate
        ax.axvline(x=s_time, color=color, linestyle=linestyle, linewidth=1.0, alpha=0.9)
        ax.axvline(x=e_time, color=color, linestyle=linestyle, linewidth=1.0, alpha=0.9)
        if label:
            # 在图例中用一个小水平线示意（通过 plot 一个不可见点并给 label）
            ax.plot([], [], color=color, linestyle=linestyle, linewidth=1.0, label=label)

    # 上一个（蓝色），当前（绿色），下一个（蓝色）
    mark_window_lines(max(0, prev_ideal_start), min(len(signal_data)-1, prev_ideal_end), 'C0', label='上一个峰截取段')
    mark_window_lines(max(0, ideal_start), min(len(signal_data)-1, ideal_end), 'C2', label='当前峰截取段')
    mark_window_lines(max(0, next_ideal_start), min(len(signal_data)-1, next_ideal_end), 'C0', label='下一个峰截取段')

    ax.set_xlabel('时间/s')
    ax.set_ylabel('幅值')
    ax.set_title(f'端点检测放大图（第 {segment_idx+1} 个敲击，显示上/当前/下峰截取段）')
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
    策略：
    1. 以峰值为对齐点
    2. 向峰值前后扩展至 target_length
    3. 如果不够长度，用零填充（padding）
    4. 不允许跨入下一个峰的区域
    """
    extracted = []
    
    for i, (start, end, peak) in enumerate(segments):
        segment_length = end - start
        
        # 计算以峰值为中心的理想截取范围
        # 让峰值位于固定位置（例如 1/4 处，留更多空间给衰减）
        before_peak = target_length // 2
        after_peak = target_length - before_peak
        
        ideal_start = peak - before_peak
        ideal_end = peak + after_peak
        
        # 调整起点：不能早于当前段的起点
        actual_start = max(start, ideal_start)
        
        # 调整终点：不能晚于当前段的终点（不跨入下一峰）
        actual_end = min(end, ideal_end)
        
        # 确保不越界
        actual_start = max(0, actual_start)
        actual_end = min(len(signal_data), actual_end)
        
        # 提取信号
        segment = signal_data[actual_start:actual_end]
        
        # 如果长度不足，补零
        if len(segment) < target_length:
            # 计算需要在前后补多少零
            # 优先保证峰值在正确位置
            peak_offset_in_segment = peak - actual_start
            target_peak_position = before_peak
            
            if peak_offset_in_segment < target_peak_position:
                # 需要在前面补零
                pad_before = target_peak_position - peak_offset_in_segment
                pad_after = target_length - len(segment) - pad_before
            else:
                # 正常情况
                pad_before = 0
                pad_after = target_length - len(segment)
            
            # 确保补零数量非负
            pad_before = max(0, pad_before)
            pad_after = max(0, target_length - len(segment) - pad_before)
            
            segment = np.pad(segment, (pad_before, pad_after), 'constant')
        
        # 如果长度超过目标，从峰值对齐的角度截取
        if len(segment) > target_length:
            peak_in_segment = peak - actual_start
            cut_start = max(0, peak_in_segment - before_peak)
            segment = segment[cut_start:cut_start + target_length]
        
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
