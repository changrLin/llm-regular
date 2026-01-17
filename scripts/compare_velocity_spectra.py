import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '.')
from scripts.cmp import load_from_hdf5
from src.neurosymbolic_avo.core.nmo import apply_nmo_correction


def compute_ab_semblance(cmp, time_windows, velocities):
    """计算传统的AB相似度速度谱"""
    n_time = len(time_windows)
    n_velocity = len(velocities)
    semblance = np.zeros((n_time, n_velocity))
    
    for t_idx, t in enumerate(time_windows):
        for v_idx, v in enumerate(velocities):
            # NMO校正
            y_nmo = apply_nmo_correction(cmp, t, v)
            
            # AB相似度公式
            sum_y = np.sum(y_nmo)
            sum_y2 = np.sum(y_nmo**2)
            
            if sum_y2 > 1e-10:
                semblance[t_idx, v_idx] = (sum_y**2) / (cmp.n_traces * sum_y2)
            else:
                semblance[t_idx, v_idx] = 0.0
    
    return semblance


def compare_velocity_spectra():
    """对比传统AB相似度和神经符号方法"""
    print("=== 对比速度谱方法 ===")
    
    # 加载CMP道集
    file_path = "data/synthetic/modeldata_cmp.h5"
    cdp_id = 250
    
    print(f"加载CMP {cdp_id}...")
    cmp = load_from_hdf5(file_path, cdp_id)
    cmp.offsets = cmp.offsets * 0.3048  # 英尺转米
    
    print(f"CMP信息: {cmp.n_traces}道, {cmp.n_samples}采样点")
    
    # 设置参数
    time_windows = np.arange(0.2, 1.5, 0.01)
    velocities = np.linspace(1500, 4500, 60)
    
    print(f"时间窗口: {len(time_windows)}个")
    print(f"速度范围: {len(velocities)}个")
    
    # 计算传统AB相似度
    print("\n计算传统AB相似度...")
    ab_semblance = compute_ab_semblance(cmp, time_windows, velocities)
    
    # 计算神经符号方法
    print("计算神经符号方法...")
    from src.neurosymbolic_avo.pipeline import process_cmp_optimized
    from src.neurosymbolic_avo.agent.llm_agent import SeismicAgent
    
    # 使用LLM代理进行内核设计
    agent = SeismicAgent()
    neuro_semblance = process_cmp_optimized(
        cmp,
        agent=agent,
        time_windows=time_windows,
        velocities=velocities,
        config={'n_sparse_samples': 10}
    )
    
    # 绘制对比图
    print("\n绘制对比图...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 颜色映射
    colors = ['darkblue', 'blue', 'cyan', 'yellow', 'orange', 'red', 'darkred']
    cmap = plt.cm.colors.LinearSegmentedColormap.from_list('seismic_hot', colors, N=256)
    
    extent = [velocities[0], velocities[-1], time_windows[-1], time_windows[0]]
    
    # 传统AB相似度
    im1 = axes[0].imshow(ab_semblance, aspect='auto', cmap=cmap, extent=extent, vmin=0, vmax=1)
    axes[0].set_xlabel('Velocity (m/s)')
    axes[0].set_ylabel('Time (s)')
    axes[0].set_title('Traditional AB Semblance', fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(im1, ax=axes[0], label='Semblance')
    
    # 神经符号方法
    im2 = axes[1].imshow(neuro_semblance, aspect='auto', cmap=cmap, extent=extent, vmin=0, vmax=1)
    axes[1].set_xlabel('Velocity (m/s)')
    axes[1].set_ylabel('Time (s)')
    axes[1].set_title('NeuroSymbolic Semblance', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    plt.colorbar(im2, ax=axes[1], label='Semblance')
    
    # 差异图
    diff = neuro_semblance - ab_semblance
    im3 = axes[2].imshow(diff, aspect='auto', cmap='RdBu_r', extent=extent, vmin=-0.5, vmax=0.5)
    axes[2].set_xlabel('Velocity (m/s)')
    axes[2].set_ylabel('Time (s)')
    axes[2].set_title('Difference (Neuro - AB)', fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    plt.colorbar(im3, ax=axes[2], label='Difference')
    
    plt.suptitle(f'Velocity Spectrum Comparison - CMP {cdp_id}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('velocity_spectrum_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 分析结果
    print("\n=== 分析结果 ===")
    
    # 传统方法结果
    ab_max_idx = np.unravel_index(np.argmax(ab_semblance), ab_semblance.shape)
    ab_max_time = time_windows[ab_max_idx[0]]
    ab_max_velocity = velocities[ab_max_idx[1]]
    ab_max_semblance = ab_semblance[ab_max_idx]
    
    # 神经符号方法结果
    neuro_max_idx = np.unravel_index(np.argmax(neuro_semblance), neuro_semblance.shape)
    neuro_max_time = time_windows[neuro_max_idx[0]]
    neuro_max_velocity = velocities[neuro_max_idx[1]]
    neuro_max_semblance = neuro_semblance[neuro_max_idx]
    
    print(f"传统AB相似度:")
    print(f"  最高相似度: {ab_max_semblance:.4f}")
    print(f"  对应时间: {ab_max_time:.3f}s")
    print(f"  对应速度: {ab_max_velocity:.0f} m/s")
    
    print(f"\n神经符号方法:")
    print(f"  最高相似度: {neuro_max_semblance:.4f}")
    print(f"  对应时间: {neuro_max_time:.3f}s")
    print(f"  对应速度: {neuro_max_velocity:.0f} m/s")
    
    print(f"\n差异分析:")
    print(f"  相似度提升: {neuro_max_semblance - ab_max_semblance:.4f}")
    print(f"  速度差异: {neuro_max_velocity - ab_max_velocity:.0f} m/s")
    
    # 统计假红区减少
    threshold = 0.7
    ab_high_semblance = np.sum(ab_semblance > threshold)
    neuro_high_semblance = np.sum(neuro_semblance > threshold)
    
    print(f"\n假红区分析 (相似度 > {threshold}):")
    print(f"  传统方法高相似度点数: {ab_high_semblance}")
    print(f"  神经符号方法高相似度点数: {neuro_high_semblance}")
    print(f"  假红区减少: {ab_high_semblance - neuro_high_semblance} 点")
    
    return ab_semblance, neuro_semblance


def plot_individual_spectra():
    """单独绘制每个速度谱"""
    file_path = "data/synthetic/modeldata_cmp.h5"
    cdp_id = 250
    
    cmp = load_from_hdf5(file_path, cdp_id)
    cmp.offsets = cmp.offsets * 0.3048
    
    time_windows = np.arange(0.2, 1.5, 0.01)
    velocities = np.linspace(1500, 4500, 60)
    
    # 传统AB相似度
    ab_semblance = compute_ab_semblance(cmp, time_windows, velocities)
    
    # 绘制传统AB相似度
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['darkblue', 'blue', 'cyan', 'yellow', 'orange', 'red', 'darkred']
    cmap = plt.cm.colors.LinearSegmentedColormap.from_list('seismic_hot', colors, N=256)
    
    extent = [velocities[0], velocities[-1], time_windows[-1], time_windows[0]]
    
    im = ax.imshow(ab_semblance, aspect='auto', cmap=cmap, extent=extent, vmin=0, vmax=1)
    ax.set_xlabel('Velocity (m/s)', fontsize=12)
    ax.set_ylabel('Time (s)', fontsize=12)
    ax.set_title('Traditional AB Semblance Velocity Spectrum', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.colorbar(im, ax=ax, label='Semblance')
    plt.tight_layout()
    plt.savefig('traditional_velocity_spectrum.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ 传统速度谱已保存到: traditional_velocity_spectrum.png")


if __name__ == "__main__":
    # 绘制传统AB相似度速度谱
    plot_individual_spectra()
    
    # 对比两种方法
    print("\n" + "="*50)
    ab_semblance, neuro_semblance = compare_velocity_spectra()
    
    print("\n✅ 对比图已保存到: velocity_spectrum_comparison.png")
    print("🎉 速度谱对比完成!")