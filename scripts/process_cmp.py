import argparse
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neurosymbolic_avo import (
    load_cmp_from_segy,
    generate_synthetic_cmp,
    process_cmp_optimized,
    plot_velocity_spectrum,
    SeismicAgent
)

# 导入HDF5加载函数
try:
    from cmp import load_from_hdf5
except ImportError:
    # 如果直接运行脚本，可能需要添加路径
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from cmp import load_from_hdf5


def main():
    parser = argparse.ArgumentParser(
        description="NeuroSymbolic AVO Velocity Analysis"
    )

    parser.add_argument(
        "--input",
        type=str,
        help="Input SEG-Y file path"
    )

    parser.add_argument(
        "--cmp",
        type=int,
        help="CMP number to process"
    )

    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data instead of SEG-Y file"
    )

    parser.add_argument(
        "--avo-type",
        type=str,
        default="II",
        choices=["I", "II", "III"],
        help="AVO type for synthetic data"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="velocity_spectrum.png",
        help="Output figure path"
    )

    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use LLM agent for kernel design"
    )

    parser.add_argument(
        "--time-start",
        type=float,
        default=0.2,
        help="Start time (seconds)"
    )

    parser.add_argument(
        "--time-end",
        type=float,
        default=1.5,
        help="End time (seconds)"
    )

    parser.add_argument(
        "--time-step",
        type=float,
        default=0.01,
        help="Time step (seconds)"
    )

    parser.add_argument(
        "--vel-min",
        type=float,
        default=1500,
        help="Minimum velocity (m/s)"
    )

    parser.add_argument(
        "--vel-max",
        type=float,
        default=4500,
        help="Maximum velocity (m/s)"
    )

    parser.add_argument(
        "--vel-n",
        type=int,
        default=50,
        help="Number of velocity samples"
    )

    args = parser.parse_args()

    if args.synthetic:
        print(f"Generating synthetic CMP with AVO type {args.avo_type}...")
        cmp = generate_synthetic_cmp(avo_type=args.avo_type)
    elif args.input and args.cmp:
        print(f"Loading CMP {args.cmp} from {args.input}...")
        
        # 检查文件格式
        if args.input.endswith('.h5') or args.input.endswith('.hdf5'):
            # HDF5格式
            cmp = load_from_hdf5(args.input, args.cmp)
            # 转换为米单位（我们的系统使用米）
            cmp.offsets = cmp.offsets * 0.3048  # 英尺转米
        else:
            # SEG-Y格式
            cmp = load_cmp_from_segy(args.input, args.cmp)
    else:
        parser.print_help()
        sys.exit(1)

    print(f"CMP loaded: {cmp.n_traces} traces × {cmp.n_samples} samples")
    print(f"Time range: {cmp.time_axis[0]:.3f}s - {cmp.time_axis[-1]:.3f}s")
    print(f"Offset range: {cmp.offsets[0]:.0f}m - {cmp.offsets[-1]:.0f}m")

    time_windows = np.arange(args.time_start, args.time_end, args.time_step)
    velocities = np.linspace(args.vel_min, args.vel_max, args.vel_n)

    agent = None
    if args.use_llm:
        print("Initializing LLM agent...")
        agent = SeismicAgent()
    else:
        print("Using rule-based kernel design...")

    print(f"Processing {len(time_windows)} time windows × {len(velocities)} velocities...")

    velocity_spectrum = process_cmp_optimized(
        cmp,
        agent=agent,
        time_windows=time_windows,
        velocities=velocities
    )

    print(f"Saving result to {args.output}...")
    plot_velocity_spectrum(
        velocity_spectrum,
        time_windows,
        velocities,
        title=f"NeuroSymbolic Velocity Spectrum - CMP {args.cmp if args.cmp else 'Synthetic'}",
        save_path=args.output
    )

    # 保存和可视化LLM推理过程
    if args.use_llm and agent is not None:
        # 保存推理日志
        log_file = f"llm_inference_log_cmp_{args.cmp if args.cmp else 'synthetic'}.json"
        agent.save_inference_log(log_file)
        print(f"LLM推理日志已保存到: {log_file}")
        
        # 生成推理过程可视化
        viz_file = f"llm_inference_visualization_cmp_{args.cmp if args.cmp else 'synthetic'}.png"
        agent.visualize_inference_process(viz_file)
        print(f"LLM推理过程可视化已保存到: {viz_file}")

    print("Done!")


if __name__ == "__main__":
    main()
