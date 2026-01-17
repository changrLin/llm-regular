import numpy as np
import argparse
import sys
sys.path.insert(0, '.')

from scripts.cmp import load_from_hdf5
from src.neurosymbolic_avo.pipeline import process_cmp_optimized
from src.neurosymbolic_avo.visualization.velocity_spectrum import plot_velocity_spectrum
from src.neurosymbolic_avo.agent.llm_agent import SeismicAgent
from src.neurosymbolic_avo.core.nmo import apply_nmo_correction


def compute_ab_semblance(cmp, time_windows, velocities):
    """计算传统的AB相似度速度谱"""
    n_time = len(time_windows)
    n_velocity = len(velocities)
    semblance = np.zeros((n_time, n_velocity))
    
    for t_idx, t in enumerate(time_windows):
        for v_idx, v in enumerate(velocities):
            y_nmo = apply_nmo_correction(cmp, t, v)
            
            sum_y = np.sum(y_nmo)
            sum_y2 = np.sum(y_nmo**2)
            
            if sum_y2 > 1e-10:
                semblance[t_idx, v_idx] = (sum_y**2) / (cmp.n_traces * sum_y2)
            else:
                semblance[t_idx, v_idx] = 0.0
    
    return semblance


def main():
    parser = argparse.ArgumentParser(description="Compare LLM vs Traditional Velocity Analysis")
    parser.add_argument("--cmp", type=int, default=250, help="CMP number to process")
    parser.add_argument("--time-min", type=float, default=0.2, help="Minimum time (s)")
    parser.add_argument("--time-max", type=float, default=1.5, help="Maximum time (s)")
    parser.add_argument("--time-step", type=float, default=0.01, help="Time step (s)")
    parser.add_argument("--vel-min", type=float, default=1500, help="Minimum velocity (m/s)")
    parser.add_argument("--vel-max", type=float, default=4500, help="Maximum velocity (m/s)")
    parser.add_argument("--vel-n", type=int, default=60, help="Number of velocities")
    parser.add_argument("--api-key", type=str, help="OpenAI API key")
    
    args = parser.parse_args()
    
    print("=== LLM vs Traditional Velocity Analysis ===")
    
    # 加载CMP道集
    file_path = "data/synthetic/modeldata_cmp.h5"
    print(f"Loading CMP {args.cmp}...")
    
    cmp = load_from_hdf5(file_path, args.cmp)
    cmp.offsets = cmp.offsets * 0.3048  # 英尺转米
    
    print(f"CMP info: {cmp.n_traces} traces, {cmp.n_samples} samples")
    
    # 设置参数
    time_windows = np.arange(args.time_min, args.time_max, args.time_step)
    velocities = np.linspace(args.vel_min, args.vel_max, args.vel_n)
    
    print(f"Time windows: {len(time_windows)}")
    print(f"Velocities: {len(velocities)}")
    
    # 1. 计算传统AB相似度
    print("\n1. Computing traditional AB semblance...")
    ab_semblance = compute_ab_semblance(cmp, time_windows, velocities)
    
    # 2. 计算基于规则的方法
    print("2. Computing rule-based method...")
    rule_semblance = process_cmp_optimized(
        cmp,
        agent=None,  # 使用基于规则的内核设计
        time_windows=time_windows,
        velocities=velocities,
        config={'n_sparse_samples': 10}
    )
    
    # 3. 计算LLM方法
    print("3. Computing LLM-based method...")
    
    # 设置API密钥
    if args.api_key:
        import os
        os.environ['OPENAI_API_KEY'] = args.api_key
        print("   Using provided API key")
    else:
        print("   Warning: No API key provided, LLM calls will fail")
    
    try:
        agent = SeismicAgent()
        llm_semblance = process_cmp_optimized(
            cmp,
            agent=agent,  # 使用LLM内核设计
            time_windows=time_windows,
            velocities=velocities,
            config={'n_sparse_samples': 10}
        )
        llm_success = True
    except Exception as e:
        print(f"   LLM method failed: {e}")
        print("   Falling back to rule-based method for comparison")
        llm_semblance = rule_semblance.copy()
        llm_success = False
    
    # 绘制对比图
    print("\n4. Generating comparison plots...")
    
    colors = ['darkblue', 'blue', 'cyan', 'yellow', 'orange', 'red', 'darkred']
    cmap = plt.cm.colors.LinearSegmentedColormap.from_list('seismic_hot', colors, N=256)
    
    extent = [velocities[0], velocities[-1], time_windows[-1], time_windows[0]]
    
    # 三图对比
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 第一行：三种方法
    im1 = axes[0, 0].imshow(ab_semblance, aspect='auto', cmap=cmap, extent=extent, vmin=0, vmax=1)
    axes[0, 0].set_xlabel('Velocity (m/s)')
    axes[0, 0].set_ylabel('Time (s)')
    axes[0, 0].set_title('Traditional AB Semblance', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    plt.colorbar(im1, ax=axes[0, 0], label='Semblance')
    
    im2 = axes[0, 1].imshow(rule_semblance, aspect='auto', cmap=cmap, extent=extent, vmin=0, vmax=1)
    axes[0, 1].set_xlabel('Velocity (m/s)')
    axes[0, 1].set_ylabel('Time (s)')
    axes[0, 1].set_title('Rule-Based Method', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    plt.colorbar(im2, ax=axes[0, 1], label='Semblance')
    
    im3 = axes[0, 2].imshow(llm_semblance, aspect='auto', cmap=cmap, extent=extent, vmin=0, vmax=1)
    axes[0, 2].set_xlabel('Velocity (m/s)')
    axes[0, 2].set_ylabel('Time (s)')
    title = 'LLM-Based Method' if llm_success else 'LLM-Based Method (Fallback)'
    axes[0, 2].set_title(title, fontweight='bold')
    axes[0, 2].grid(True, alpha=0.3)
    plt.colorbar(im3, ax=axes[0, 2], label='Semblance')
    
    # 第二行：差异对比
    diff_rule_ab = rule_semblance - ab_semblance
    diff_llm_ab = llm_semblance - ab_semblance
    diff_llm_rule = llm_semblance - rule_semblance
    
    im4 = axes[1, 0].imshow(diff_rule_ab, aspect='auto', cmap='RdBu_r', extent=extent, vmin=-0.5, vmax=0.5)
    axes[1, 0].set_xlabel('Velocity (m/s)')
    axes[1, 0].set_ylabel('Time (s)')
    axes[1, 0].set_title('Rule - AB', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    plt.colorbar(im4, ax=axes[1, 0], label='Difference')
    
    im5 = axes[1, 1].imshow(diff_llm_ab, aspect='auto', cmap='RdBu_r', extent=extent, vmin=-0.5, vmax=0.5)
    axes[1, 1].set_xlabel('Velocity (m/s)')
    axes[1, 1].set_ylabel('Time (s)')
    axes[1, 1].set_title('LLM - AB', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    plt.colorbar(im5, ax=axes[1, 1], label='Difference')
    
    im6 = axes[1, 2].imshow(diff_llm_rule, aspect='auto', cmap='RdBu_r', extent=extent, vmin=-0.5, vmax=0.5)
    axes[1, 2].set_xlabel('Velocity (m/s)')
    axes[1, 2].set_ylabel('Time (s)')
    axes[1, 2].set_title('LLM - Rule', fontweight='bold')
    axes[1, 2].grid(True, alpha=0.3)
    plt.colorbar(im6, ax=axes[1, 2], label='Difference')
    
    plt.suptitle(f'Velocity Spectrum Comparison - CMP {args.cmp}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('llm_vs_traditional_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 分析结果
    print("\n5. Analyzing results...")
    
    def analyze_spectrum(semblance, name):
        max_idx = np.unravel_index(np.argmax(semblance), semblance.shape)
        max_time = time_windows[max_idx[0]]
        max_velocity = velocities[max_idx[1]]
        max_semblance = semblance[max_idx]
        
        # 统计高相似度点数
        threshold = 0.7
        high_points = np.sum(semblance > threshold)
        
        return {
            'max_semblance': max_semblance,
            'max_time': max_time,
            'max_velocity': max_velocity,
            'high_points': high_points
        }
    
    ab_results = analyze_spectrum(ab_semblance, "AB")
    rule_results = analyze_spectrum(rule_semblance, "Rule")
    llm_results = analyze_spectrum(llm_semblance, "LLM")
    
    print(f"\n=== Results Summary ===")
    print(f"Method           | Max Semblance | Time (s) | Velocity (m/s) | High Points")
    print(f"-" * 70)
    print(f"AB Semblance     | {ab_results['max_semblance']:.4f}        | {ab_results['max_time']:.3f}    | {ab_results['max_velocity']:.0f}         | {ab_results['high_points']}")
    print(f"Rule-Based       | {rule_results['max_semblance']:.4f}        | {rule_results['max_time']:.3f}    | {rule_results['max_velocity']:.0f}         | {rule_results['high_points']}")
    print(f"LLM-Based        | {llm_results['max_semblance']:.4f}        | {llm_results['max_time']:.3f}    | {llm_results['max_velocity']:.0f}         | {llm_results['high_points']}")
    
    print(f"\n=== Improvements ===")
    print(f"LLM vs AB:  Semblance +{llm_results['max_semblance'] - ab_results['max_semblance']:.4f}, "
          f"High Points -{ab_results['high_points'] - llm_results['high_points']}")
    print(f"LLM vs Rule: Semblance +{llm_results['max_semblance'] - rule_results['max_semblance']:.4f}, "
          f"High Points -{rule_results['high_points'] - llm_results['high_points']}")
    
    print(f"\n✅ Comparison completed!")
    print(f"✅ Results saved to: llm_vs_traditional_comparison.png")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    main()