#!/usr/bin/env python
"""
纯Python实现合成CMP道集生成脚本
替代Shell脚本中的susynlv工具链

直接运行即可生成数据：
    python generate_synthetic_cmp.py
"""

import numpy as np
import h5py
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from scipy import signal


# ============================================================================
# 数据结构定义
# ============================================================================

@dataclass
class CMPGather:
    """CMP道集数据结构"""
    data: np.ndarray          # shape: (n_traces, n_samples)
    offsets: np.ndarray       # shape: (n_traces,)
    angles: np.ndarray        # shape: (n_traces,)
    time_axis: np.ndarray     # shape: (n_samples,)
    dt: float                 # 采样间隔（秒）
    sample_rate: float        # 采样率（Hz）
    
    @property
    def n_traces(self):
        return self.data.shape[0]
    
    @property
    def n_samples(self):
        return self.data.shape[1]
    
    def get_amplitudes_at_time(self, t: float) -> np.ndarray:
        """获取指定时间的振幅值"""
        # 找到最接近的时间索引
        idx = np.argmin(np.abs(self.time_axis - t))
        return self.data[:, idx]
    
    def get_trace(self, trace_idx: int) -> np.ndarray:
        """获取指定道的完整数据"""
        return self.data[trace_idx, :]


@dataclass
class Reflector:
    """反射层定义"""
    depth: float        # 深度（ft）
    amplitude:  float    # 反射系数
    x_start: float      # 横向起始位置
    x_end: float        # 横向结束位置


# ============================================================================
# 核心算法函数
# ============================================================================

def ricker_wavelet(freq: float, dt: float, length: float = 0.128) -> np.ndarray:
    """
    生成Ricker子波
    
    Args:
        freq: 主频（Hz）
        dt:  采样间隔（秒）
        length: 子波长度（秒）
    
    Returns:
        归一化的Ricker子波
    """
    t = np.arange(-length/2, length/2, dt)
    y = (1.0 - 2.0*(np.pi*freq*t)**2) * np.exp(-(np.pi*freq*t)**2)
    return y / np.max(np.abs(y))


def compute_velocity(z: float, v00: float, dvdz: float) -> float:
    """
    计算指定深度的速度（线性梯度模型）
    
    Args: 
        z: 深度
        v00: 表面速度
        dvdz: 速度梯度
    
    Returns:
        该深度的速度
    """
    return v00 + dvdz * z


def nmo_traveltime(offset: float, t0: float, v_nmo: float) -> float:
    """
    NMO双曲线走时公式
    
    Args: 
        offset: 偏移距
        t0: 零偏移距双程走时
        v_nmo:  NMO速度
    
    Returns:
        该偏移距的双程走时
    """
    return np.sqrt(t0**2 + (offset/v_nmo)**2)


def generate_trace(
    shot_x: float,
    receiver_x: float,
    reflectors: List[Reflector],
    wavelet: np.ndarray,
    nt: int,
    dt:  float,
    v00: float,
    dvdz: float
) -> np.ndarray:
    """
    生成单道地震记录
    
    Args: 
        shot_x: 炮点位置
        receiver_x: 检波点位置
        reflectors: 反射层列表
        wavelet: 子波
        nt: 时间样本数
        dt: 采样间隔
        v00: 表面速度
        dvdz: 速度梯度
    
    Returns: 
        时间道数据
    """
    trace = np.zeros(nt)
    offset = abs(receiver_x - shot_x)
    midpoint_x = (shot_x + receiver_x) / 2
    
    wavelet_len = len(wavelet)
    wavelet_center = wavelet_len // 2
    
    for reflector in reflectors:
        # 检查反射层是否覆盖该位置
        if not (reflector.x_start <= midpoint_x <= reflector. x_end):
            continue
        
        # 计算平均速度
        depth = reflector.depth
        v_avg = compute_velocity(depth/2, v00, dvdz)
        
        # 零偏移距双程走时
        t0 = 2.0 * depth / v_avg
        
        # NMO走时
        t_nmo = nmo_traveltime(offset, t0, v_avg)
        
        # 转换为样本索引
        sample_idx = int(t_nmo / dt)
        
        # 插入子波
        start_idx = sample_idx - wavelet_center
        end_idx = start_idx + wavelet_len
        
        if 0 <= start_idx < nt and end_idx <= nt:
            trace[start_idx:end_idx] += reflector.amplitude * wavelet
    
    return trace


def generate_shot_gather(
    shot_x: float,
    offsets: np.ndarray,
    reflectors: List[Reflector],
    wavelet: np.ndarray,
    nt: int,
    dt: float,
    v00: float,
    dvdz: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成单炮记录
    
    Returns:
        shot_data: shape (n_offsets, nt)
        receiver_positions: shape (n_offsets,)
    """
    n_offsets = len(offsets)
    shot_data = np.zeros((n_offsets, nt))
    receiver_positions = shot_x + offsets
    
    for i, receiver_x in enumerate(receiver_positions):
        shot_data[i, :] = generate_trace(
            shot_x, receiver_x, reflectors, wavelet, 
            nt, dt, v00, dvdz
        )
    
    return shot_data, receiver_positions


def sort_to_cmp(
    all_shots: List[np.ndarray],
    all_receivers: List[np.ndarray],
    shot_positions: np. ndarray
) -> Dict[int, Tuple[np.ndarray, np.ndarray, float]]:
    """
    将炮集重排为CMP道集
    
    Returns:
        {cdp_bin:  (data, offsets, cdp_x)}
    """
    traces_info = []
    
    for shot_idx, shot_x in enumerate(shot_positions):
        shot_data = all_shots[shot_idx]
        receiver_pos = all_receivers[shot_idx]
        
        for trace_idx, receiver_x in enumerate(receiver_pos):
            cdp_x = (shot_x + receiver_x) / 2
            offset = abs(receiver_x - shot_x)
            
            traces_info.append({
                'cdp_x': cdp_x,
                'offset': offset,
                'data': shot_data[trace_idx, :]
            })
    
    # 按CMP位置分组（量化到10ft单位）
    cmp_groups = {}
    for info in traces_info:
        cdp_bin = int(round(info['cdp_x'] / 10.0))
        
        if cdp_bin not in cmp_groups:
            cmp_groups[cdp_bin] = []
        
        cmp_groups[cdp_bin].append(info)
    
    # 对每个CMP按偏移距排序
    cmp_gathers = {}
    for cdp_bin, trace_list in cmp_groups.items():
        trace_list.sort(key=lambda x: x['offset'])
        
        data = np.array([t['data'] for t in trace_list])
        offsets = np.array([t['offset'] for t in trace_list])
        cdp_x = np.mean([t['cdp_x'] for t in trace_list])
        
        cmp_gathers[cdp_bin] = (data, offsets, cdp_x)
    
    return cmp_gathers


def add_bandlimited_noise(
    data: np.ndarray,
    snr_db: float,
    f1: float,
    f2: float,
    f3: float,
    f4: float,
    dt: float
) -> np.ndarray:
    """
    添加带限白噪声（对应suaddnoise）
    
    Args:
        data: 输入数据
        snr_db: 信噪比（dB）
        f1, f2, f3, f4: 梯形滤波器的4个角频率
        dt: 采样间隔
    
    Returns:
        加噪后的数据
    """
    # 计算信号功率
    signal_power = np.mean(data**2)
    
    # 计算噪声功率
    noise_power = signal_power / (10**(snr_db/10))
    
    # 生成白噪声
    noise = np.random.normal(0, np.sqrt(noise_power), data.shape)
    
    # 带通滤波
    fs = 1.0 / dt
    nyquist = fs / 2.0
    
    # 使用Butterworth滤波器
    sos = signal.butter(4, [f2/nyquist, f3/nyquist], btype='band', output='sos')
    
    if data.ndim == 1:
        noise_filtered = signal.sosfilt(sos, noise)
    else:
        noise_filtered = np.zeros_like(noise)
        for i in range(data.shape[0]):
            noise_filtered[i, :] = signal.sosfilt(sos, noise[i, :])
    
    return data + noise_filtered


def create_cmp_gather(
    data: np.ndarray,
    offsets: np.ndarray,
    cdp_x: float,
    time_axis: np.ndarray,
    dt: float,
    v00: float,
    dvdz: float
) -> CMPGather:
    """
    创建CMPGather对象
    """
    # 估算深度用于计算角度
    t_mid = time_axis[len(time_axis)//2]
    v_mid = compute_velocity(v00 * t_mid / 2, v00, dvdz)
    depth_approx = v_mid * t_mid / 2
    
    # 计算入射角
    angles = np.rad2deg(np.arctan(offsets / (2 * depth_approx)))
    
    return CMPGather(
        data=data,
        offsets=offsets,
        angles=angles,
        time_axis=time_axis,
        dt=dt,
        sample_rate=1.0/dt
    )


# ============================================================================
# 保存和可视化
# ============================================================================

def save_to_hdf5(cmp_gathers: Dict[int, CMPGather], filepath: str):
    """保存到HDF5文件"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(filepath, 'w') as f:
        for cdp_id, gather in cmp_gathers.items():
            grp = f.create_group(f'cdp_{cdp_id}')
            grp.create_dataset('data', data=gather.data, compression='gzip')
            grp.create_dataset('offsets', data=gather.offsets)
            grp.create_dataset('angles', data=gather.angles)
            grp.create_dataset('time_axis', data=gather.time_axis)
            grp.attrs['dt'] = gather.dt
            grp.attrs['sample_rate'] = gather.sample_rate
            grp.attrs['n_traces'] = gather.n_traces
            grp.attrs['n_samples'] = gather.n_samples
    
    print(f"✓ Saved {len(cmp_gathers)} CMP gathers to {filepath}")


def load_from_hdf5(filepath: str, cdp_id: int) -> CMPGather:
    """从HDF5加载"""
    with h5py.File(filepath, 'r') as f:
        grp = f[f'cdp_{cdp_id}']
        return CMPGather(
            data=grp['data'][:],
            offsets=grp['offsets'][:],
            angles=grp['angles'][:],
            time_axis=grp['time_axis'][: ],
            dt=grp. attrs['dt'],
            sample_rate=grp.attrs['sample_rate']
        )


def visualize_gather(gather: CMPGather, cdp_id: int, save_path: str = None):
    """可视化CMP道集"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    extent = [gather.offsets[0], gather.offsets[-1],
              gather.time_axis[-1], gather.time_axis[0]]
    
    vmax = np.percentile(np.abs(gather.data), 95)
    
    im = ax.imshow(
        gather.data. T,
        aspect='auto',
        cmap='seismic',
        extent=extent,
        vmin=-vmax,
        vmax=vmax,
        interpolation='bilinear'
    )
    
    ax.set_xlabel('Offset (ft)', fontsize=12)
    ax.set_ylabel('Time (s)', fontsize=12)
    ax.set_title(f'Synthetic CMP {cdp_id} ({gather.n_traces} traces)', 
                 fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Amplitude')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved figure:  {save_path}")
    else:
        plt.show()


# ============================================================================
# 主程序
# ============================================================================

def main():
    """
    主函数：生成合成CMP道集
    对应原Shell脚本的参数
    """
    
    print("\n" + "="*70)
    print("Generating Synthetic CMP Gathers (Python Implementation)")
    print("="*70 + "\n")
    
    # ========== 参数设置（对应Shell脚本）==========
    
    # 反射层定义（对应REF1-REF4）
    reflectors = [
        Reflector(depth=1000.0, amplitude=0.0909091, x_start=-4000.0, x_end=12000.0),
        Reflector(depth=2200.0, amplitude=0.1428570, x_start=-4000.0, x_end=12000.0),
        Reflector(depth=3500.0, amplitude=0.1111110, x_start=-4000.0, x_end=12000.0),
        Reflector(depth=5000.0, amplitude=0.2000000, x_start=-4000.0, x_end=12000.0),
    ]
    
    # 速度模型
    v00 = 5000.0      # 表面速度 (ft/s)
    dvdz = 2.0        # 速度梯度 (1/s)
    
    # 子波参数
    fpeak = 25.0      # 主频 (Hz)
    
    # 时间采样
    nt = 501          # 样本数
    dt = 0.004        # 采样间隔 (s)
    
    # 偏移距参数
    nxo = 64          # 偏移距道数
    fxo = 100.0       # 首道偏移距 (ft)
    dxo = 100.0       # 偏移距间隔 (ft)
    
    # 炮点参数
    nxs = 12          # 炮点数
    fxs = 1400.0      # 首炮位置 (ft)
    dxs = -100.0      # 炮点间隔 (ft)
    
    # 噪声参数（对应suaddnoise）
    sn = 50.0         # 信噪比 (dB)
    f1, f2, f3, f4 = 4.0, 8.0, 20.0, 25.0  # 频带
    
    # 输出文件
    output_file = 'data/synthetic/modeldata_cmp.h5'
    
    # ========== 打印配置 ==========
    
    print("Configuration:")
    print(f"  Reflectors: {len(reflectors)} layers")
    for i, ref in enumerate(reflectors, 1):
        print(f"    Layer {i}: depth={ref.depth:.0f} ft, amplitude={ref.amplitude:.4f}")
    
    print(f"\n  Velocity model:")
    print(f"    v0 = {v00:.0f} ft/s")
    print(f"    dv/dz = {dvdz:.1f} (1/s)")
    
    print(f"\n  Acquisition:")
    print(f"    Offsets: {nxo} traces, {fxo:.0f}-{fxo+(nxo-1)*dxo:.0f} ft")
    print(f"    Shots: {nxs}, starting at {fxs:.0f} ft, interval {dxs:.0f} ft")
    
    print(f"\n  Time sampling:")
    print(f"    {nt} samples × {dt*1000:.1f} ms = {nt*dt:.2f} s")
    
    print(f"\n  Wavelet:")
    print(f"    Peak frequency = {fpeak:.0f} Hz")
    
    print(f"\n  Noise:")
    print(f"    SNR = {sn:.0f} dB, band = [{f1}, {f2}, {f3}, {f4}] Hz")
    
    # ========== Step 1: 生成子波 ==========
    
    print(f"\n{'='*70}")
    print("Step 1: Generating Ricker wavelet")
    print("="*70)
    
    wavelet = ricker_wavelet(fpeak, dt)
    print(f"✓ Wavelet:  {len(wavelet)} samples ({len(wavelet)*dt*1000:.1f} ms)")
    
    # ========== Step 2: 生成炮集 ==========
    
    print(f"\n{'='*70}")
    print("Step 2: Generating shot gathers")
    print("="*70)
    
    # 偏移距数组
    offsets = np. arange(nxo) * dxo + fxo
    
    # 炮点位置
    shot_positions = np.arange(nxs) * dxs + fxs
    
    all_shot_data = []
    all_receiver_pos = []
    
    for i, shot_x in enumerate(shot_positions):
        print(f"  Generating shot {i+1}/{nxs} at x={shot_x:.0f} ft.. .", end='')
        
        shot_data, receiver_pos = generate_shot_gather(
            shot_x, offsets, reflectors, wavelet, nt, dt, v00, dvdz
        )
        
        all_shot_data.append(shot_data)
        all_receiver_pos.append(receiver_pos)
        
        print(" ✓")
    
    # ========== Step 3: 排序到CMP ==========
    
    print(f"\n{'='*70}")
    print("Step 3: Sorting to CMP gathers")
    print("="*70)
    
    cmp_gathers_raw = sort_to_cmp(all_shot_data, all_receiver_pos, shot_positions)
    print(f"✓ Created {len(cmp_gathers_raw)} CMP bins")
    
    # 统计fold
    folds = [len(traces) for _, (traces, _, _) in cmp_gathers_raw.items()]
    print(f"  Fold:  min={min(folds)}, max={max(folds)}, avg={np.mean(folds):.1f}")
    
    # ========== Step 4: 添加噪声 ==========
    
    print(f"\n{'='*70}")
    print("Step 4: Adding noise")
    print("="*70)
    
    time_axis = np.arange(nt) * dt
    cmp_gathers = {}
    
    for cdp_id, (data, offset_arr, cdp_x) in cmp_gathers_raw. items():
        # 添加噪声
        noisy_data = add_bandlimited_noise(data, sn, f1, f2, f3, f4, dt)
        
        # 创建CMPGather对象
        gather = create_cmp_gather(noisy_data, offset_arr, cdp_x, time_axis, dt, v00, dvdz)
        cmp_gathers[cdp_id] = gather
    
    print(f"✓ Added noise to {len(cmp_gathers)} gathers")
    
    # ========== Step 5: 保存 ==========
    
    print(f"\n{'='*70}")
    print("Step 5: Saving data")
    print("="*70)
    
    save_to_hdf5(cmp_gathers, output_file)
    
    # ========== Step 6: 可视化几个CMP ==========
    
    print(f"\n{'='*70}")
    print("Step 6: Generating visualizations")
    print("="*70)
    
    fig_dir = Path('data/results/synthetic_gathers')
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # 可视化3个CMP
    cdp_ids = sorted(cmp_gathers.keys())
    sample_cdps = [cdp_ids[0], cdp_ids[len(cdp_ids)//2], cdp_ids[-1]]
    
    for cdp_id in sample_cdps: 
        gather = cmp_gathers[cdp_id]
        fig_path = fig_dir / f'cdp_{cdp_id}_synthetic.png'
        visualize_gather(gather, cdp_id, save_path=str(fig_path))
    
    # ========== 完成 ==========
    
    print(f"\n{'='*70}")
    print("✓ Generation Complete!")
    print("="*70 + "\n")
    
    print("Summary:")
    print(f"  Output file: {output_file}")
    print(f"  CMP gathers: {len(cmp_gathers)}")
    print(f"  Total traces: {sum(g.n_traces for g in cmp_gathers.values())}")
    print(f"  CDP range: {min(cmp_gathers.keys())} - {max(cmp_gathers.keys())}")
    
    print(f"\n{'='*70}")
    print("Usage Example:")
    print("="*70 + "\n")
    
    mid_cdp = sorted(cmp_gathers.keys())[len(cmp_gathers)//2]
    
    print("# 加载CMP道集")
    print(f"from generate_synthetic_cmp import load_from_hdf5")
    print(f"cmp = load_from_hdf5('{output_file}', cdp_id={mid_cdp})")
    print(f"")
    print(f"# 查看信息")
    print(f"print(f'Traces: {{cmp.n_traces}}')")
    print(f"print(f'Time range: {{cmp.time_axis[0]:.3f}} - {{cmp.time_axis[-1]:.3f}} s')")
    print(f"print(f'Offset range: {{cmp.offsets[0]:.0f}} - {{cmp.offsets[-1]:.0f}} ft')")
    print(f"")


if __name__ == '__main__':
    main()