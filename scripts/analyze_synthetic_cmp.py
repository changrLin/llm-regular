import numpy as np
import sys
sys.path.insert(0, '.')
from scripts.cmp import load_from_hdf5
from src.neurosymbolic_avo.pipeline import process_cmp_optimized
from src.neurosymbolic_avo.visualization.velocity_spectrum import plot_velocity_spectrum


def analyze_synthetic_cmp():
    print("=== 分析合成CMP道集 ===")
    
    # 加载CMP道集
    file_path = "data/synthetic/modeldata_cmp.h5"
    
    # 选择中间的一个CMP进行测试
    cdp_id = 250
    print(f"\n加载CMP {cdp_id}...")
    
    try:
        cmp = load_from_hdf5(file_path, cdp_id)
        
        print(f"✅ 成功加载CMP {cdp_id}")
        print(f"   道数: {cmp.n_traces}")
        print(f"   采样点数: {cmp.n_samples}")
        print(f"   时间范围: {cmp.time_axis[0]:.3f}s - {cmp.time_axis[-1]:.3f}s")
        print(f"   偏移距范围: {cmp.offsets[0]:.0f}ft - {cmp.offsets[-1]:.0f}ft")
        print(f"   数据范围: {cmp.data.min():.4f} to {cmp.data.max():.4f}")
        
        # 转换为米单位（我们的系统使用米）
        cmp.offsets = cmp.offsets * 0.3048  # 英尺转米
        
        # 设置时间窗口和速度范围
        time_windows = np.arange(0.2, 1.5, 0.01)  # 0.2s到1.5s
        velocities = np.linspace(1500, 4500, 60)  # 1500到4500 m/s
        
        print(f"\n=== 开始速度谱分析 ===")
        print(f"时间窗口数: {len(time_windows)}")
        print(f"速度点数: {len(velocities)}")
        
        # 使用神经符号方法处理
        velocity_spectrum = process_cmp_optimized(
            cmp,
            agent=None,  # 使用基于规则的内核设计
            time_windows=time_windows,
            velocities=velocities,
            config={'n_sparse_samples': 10}
        )
        
        # 绘制速度谱
        plot_velocity_spectrum(
            velocity_spectrum,
            time_windows,
            velocities,
            title=f"Velocity Spectrum - CMP {cdp_id} (Synthetic)",
            save_path=f"velocity_spectrum_cdp_{cdp_id}.png"
        )
        
        print(f"\n✅ 速度谱分析完成!")
        print(f"✅ 结果保存到: velocity_spectrum_cdp_{cdp_id}.png")
        
        # 分析速度谱结果
        max_semblance_idx = np.unravel_index(np.argmax(velocity_spectrum), velocity_spectrum.shape)
        max_time = time_windows[max_semblance_idx[0]]
        max_velocity = velocities[max_semblance_idx[1]]
        max_semblance = velocity_spectrum[max_semblance_idx]
        
        print(f"\n=== 速度谱分析结果 ===")
        print(f"最高相似度: {max_semblance:.4f}")
        print(f"对应时间: {max_time:.3f}s")
        print(f"对应速度: {max_velocity:.0f} m/s")
        
        # 根据已知速度模型验证
        # 表面速度: 5000 ft/s = 1524 m/s
        # 速度梯度: 2.0/s
        expected_v0 = 5000 * 0.3048  # 英尺转米
        print(f"预期表面速度: {expected_v0:.0f} m/s")
        
        return cmp, velocity_spectrum
        
    except Exception as e:
        print(f"❌ 加载CMP失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def analyze_multiple_cmps():
    """分析多个CMP道集"""
    print("\n=== 分析多个CMP道集 ===")
    
    file_path = "data/synthetic/modeldata_cmp.h5"
    cdp_ids = [35, 250, 460]  # 首、中、尾三个CMP
    
    results = {}
    
    for cdp_id in cdp_ids:
        print(f"\n分析CMP {cdp_id}...")
        
        try:
            cmp = load_from_hdf5(file_path, cdp_id)
            cmp.offsets = cmp.offsets * 0.3048  # 英尺转米
            
            time_windows = np.arange(0.2, 1.5, 0.01)
            velocities = np.linspace(1500, 4500, 60)
            
            velocity_spectrum = process_cmp_optimized(
                cmp,
                agent=None,
                time_windows=time_windows,
                velocities=velocities,
                config={'n_sparse_samples': 10}
            )
            
            plot_velocity_spectrum(
                velocity_spectrum,
                time_windows,
                velocities,
                title=f"Velocity Spectrum - CMP {cdp_id}",
                save_path=f"velocity_spectrum_cdp_{cdp_id}.png"
            )
            
            max_semblance_idx = np.unravel_index(np.argmax(velocity_spectrum), velocity_spectrum.shape)
            max_time = time_windows[max_semblance_idx[0]]
            max_velocity = velocities[max_semblance_idx[1]]
            max_semblance = velocity_spectrum[max_semblance_idx]
            
            results[cdp_id] = {
                'velocity_spectrum': velocity_spectrum,
                'max_velocity': max_velocity,
                'max_time': max_time,
                'max_semblance': max_semblance
            }
            
            print(f"  ✅ 完成: 速度={max_velocity:.0f} m/s, 时间={max_time:.3f}s")
            
        except Exception as e:
            print(f"  ❌ 失败: {e}")
    
    # 汇总结果
    print(f"\n=== 汇总结果 ===")
    for cdp_id, result in results.items():
        print(f"CMP {cdp_id}: 速度={result['max_velocity']:.0f} m/s, "
              f"时间={result['max_time']:.3f}s, "
              f"相似度={result['max_semblance']:.4f}")


if __name__ == "__main__":
    # 分析单个CMP
    cmp, velocity_spectrum = analyze_synthetic_cmp()
    
    # 分析多个CMP
    if cmp is not None:
        analyze_multiple_cmps()
    
    print("\n🎉 所有分析完成!")
