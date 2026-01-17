#!/usr/bin/env python
"""
LLM推理过程可视化脚本

使用方法：
    python visualize_llm_inference.py --log-file inference_log.json --output inference_visualization.png
"""

import argparse
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neurosymbolic_avo.agent.llm_agent import SeismicAgent
from neurosymbolic_avo.core.data_structures import SeismicFeatures


def main():
    parser = argparse.ArgumentParser(description="LLM推理过程可视化")
    parser.add_argument("--log-file", type=str, help="推理日志文件路径")
    parser.add_argument("--output", type=str, default="llm_inference_visualization.png", 
                       help="输出图像文件路径")
    parser.add_argument("--realtime", action="store_true", 
                       help="实时运行并记录推理过程")
    parser.add_argument("--cmp", type=int, default=250, help="CMP编号")
    
    args = parser.parse_args()
    
    agent = SeismicAgent()
    
    if args.realtime:
        # 实时运行推理并记录
        print("=== 实时LLM推理过程可视化 ===")
        
        # 加载CMP数据
        from scripts.cmp import load_from_hdf5
        
        file_path = "data/synthetic/modeldata_cmp.h5"
        print(f"加载CMP {args.cmp}...")
        
        cmp = load_from_hdf5(file_path, args.cmp)
        cmp.offsets = cmp.offsets * 0.3048  # 英尺转米
        
        print(f"CMP信息: {cmp.n_traces}道, {cmp.n_samples}采样点")
        
        # 设置时间窗口
        time_windows = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3]  # 选择几个关键时间点
        
        print(f"将在 {len(time_windows)} 个时间点进行LLM推理...")
        
        # 特征提取器
        from neurosymbolic_avo.core.features import FeatureExtractor
        extractor = FeatureExtractor(cmp)
        
        for i, t in enumerate(time_windows):
            print(f"\n推理 {i+1}/{len(time_windows)}: 时间 {t:.2f}s")
            
            # 提取特征
            y = cmp.get_amplitudes_at_time(t)
            features = extractor.extract(y, cmp.angles)
            
            # LLM推理
            blueprint = agent.design_kernel(features)
            
            print(f"  核结构: {blueprint.base_kernel_type}")
            print(f"  决策理由: {blueprint.reasoning}")
        
        # 保存推理日志
        log_file = f"inference_log_cmp_{args.cmp}.json"
        agent.save_inference_log(log_file)
        print(f"\n推理日志已保存到: {log_file}")
        
        # 可视化
        print("\n生成可视化图表...")
        agent.visualize_inference_process(args.output)
        
    elif args.log_file:
        # 从日志文件加载并可视化
        print(f"=== 从日志文件加载推理过程 ===")
        print(f"加载日志文件: {args.log_file}")
        
        agent.load_inference_log(args.log_file)
        
        print(f"加载完成: {len(agent.inference_log)} 次推理记录")
        
        # 可视化
        agent.visualize_inference_process(args.output)
        
    else:
        parser.print_help()
        print("\n示例用法:")
        print("  实时运行: python visualize_llm_inference.py --realtime --cmp 250")
        print("  加载日志: python visualize_llm_inference.py --log-file inference_log.json")


def create_detailed_visualization():
    """创建详细的推理过程可视化"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 示例数据（用于演示）
    inference_data = {
        'time_points': [0.3, 0.5, 0.7, 0.9, 1.1, 1.3],
        'features': {
            'zcr': [0.25, 0.32, 0.41, 0.28, 0.35, 0.22],
            'curvature': [0.05, 0.12, 0.08, 0.15, 0.09, 0.06],
            'avo_slope': [-0.02, -0.08, -0.15, -0.05, -0.12, -0.03]
        },
        'kernel_types': ['RBF', 'RBF+Linear', 'RBF', 'RBF+Linear', 'RBF', 'RBF'],
        'reasoning_keywords': ['smooth', 'AVO', 'curvature', 'outlier', 'AVO', 'smooth']
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 特征演化
    ax1 = axes[0, 0]
    ax1.plot(inference_data['time_points'], inference_data['features']['zcr'], 
             marker='o', label='ZCR', linewidth=2)
    ax1.plot(inference_data['time_points'], inference_data['features']['curvature'], 
             marker='s', label='Curvature', linewidth=2)
    ax1.plot(inference_data['time_points'], inference_data['features']['avo_slope'], 
             marker='^', label='AVO Slope', linewidth=2)
    ax1.set_xlabel('时间 (s)')
    ax1.set_ylabel('特征值')
    ax1.set_title('地震特征随时间的演化')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 核结构选择
    ax2 = axes[0, 1]
    kernel_counts = {}
    for kt in inference_data['kernel_types']:
        kernel_counts[kt] = kernel_counts.get(kt, 0) + 1
    
    bars = ax2.bar(kernel_counts.keys(), kernel_counts.values(), 
                   color=['skyblue', 'lightcoral'])
    ax2.set_xlabel('核结构类型')
    ax2.set_ylabel('选择次数')
    ax2.set_title('LLM核结构选择分布')
    ax2.grid(True, alpha=0.3)
    
    # 3. 决策理由分析
    ax3 = axes[1, 0]
    keyword_counts = {}
    for kw in inference_data['reasoning_keywords']:
        keyword_counts[kw] = keyword_counts.get(kw, 0) + 1
    
    ax3.bar(keyword_counts.keys(), keyword_counts.values(), 
            color=['lightgreen', 'orange', 'purple'])
    ax3.set_xlabel('决策关键词')
    ax3.set_ylabel('出现次数')
    ax3.set_title('LLM决策理由分析')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # 4. 推理过程示意图
    ax4 = axes[1, 1]
    ax4.text(0.5, 0.5, 'LLM推理过程示意图\n\n1. 特征提取 → 2. LLM分析 → 3. 核设计', 
             ha='center', va='center', fontsize=14, transform=ax4.transAxes)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_title('LLM推理流程')
    ax4.axis('off')
    
    plt.suptitle('LLM推理过程可视化演示', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('llm_inference_demo.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()