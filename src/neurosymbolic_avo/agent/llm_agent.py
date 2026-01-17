import os
import json
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv
from ..core.data_structures import SeismicFeatures
from .blueprint import KernelBlueprint, RBFConstraint, LinearConstraint, PeriodicConstraint, NoiseStrategy

load_dotenv()


class SeismicAgent:
    def __init__(self, model: str = "gpt-4o", temperature: float = 0.1):
        api_key = os.getenv("LLM_API_KEY")
        base_url = os.getenv("LLM_BASE_URL")
        
        if base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=api_key)
        
        self.model = model
        self.temperature = temperature
        self.system_prompt = self._build_system_prompt()
        
        # 推理过程记录
        self.inference_log = []
        self.features_history = []
        self.blueprints_history = []

    def _build_system_prompt(self) -> str:
        return """你是一位地震数据处理专家，专长是AVO分析和速度谱优化。

## 任务
根据地震振幅序列的物理特征，设计高斯过程的核函数结构和超参数约束。

## 输入特征说明

### 频率特性
- **zero_crossing_rate** (ZCR): 范围[0,1]
  - <0.2: 信号平滑，变化缓慢
  - 0.2-0.5: 中等变化
  - >0.6: 剧烈振荡，高频成分多

### 趋势特性
- **curvature**: 二阶导数均值
  - <0.01: 几乎线性/平直（正确速度）
  - 0.01-0.05: 轻微弯曲
  - >0.1: 显著弯曲（残余时差RMO）

- **linear_trend_slope**: AVO梯度B
  - >0: 振幅增大（罕见）
  - 约0: 无AVO效应
  - <0: 振幅衰减（常见）

- **trend_r_squared**: 线性拟合优度
  - >0.9: 高度线性
  - 0.6-0.9: 中等线性
  - <0.6: 非线性

### AVO语义
- **avo_type**: "I", "II", "III", "Unknown"
  - I类: 常规气藏，截距主导
  - II类: 页岩气，梯度主导，需Linear核
  - III类: 亮点

### 异常检测
- **max_z_score**: 最大Z-score
  - <2: 无明显异常
  - 2-3: 轻微异常
  - >3: 存在野值，需屏蔽

- **outlier_indices**: 异常道的索引列表

## 决策规则

### 1. 核结构选择
```
IF avo_type == "I" AND curvature < 0.05:
    → base_kernel_type = "RBF"

IF avo_type == "II":
    → base_kernel_type = "RBF+Linear"
    # 理由：II类存在线性反转，需要Linear核捕捉

IF periodicity_score > 0.7:
    → base_kernel_type = "RBF+Periodic"
    # 理由：检测到多次波干扰
```

### 2. 长度尺度（length_scale）约束
```
基本原则：ZCR越高 → length_scale越小（短相关）

IF zero_crossing_rate < 0.2:
    → length_scale_bounds = [20, 50]  # 平滑，长相关

IF 0.2 <= zero_crossing_rate < 0.5:
    → length_scale_bounds = [10, 25]  # 中等

IF zero_crossing_rate >= 0.5:
    → length_scale_bounds = [5, 15]   # 粗糙，短相关

修正因子：
IF curvature > 0.08:  # 存在显著弯曲
    → 将上界缩小20%
    # 理由：缩小相关范围以拒绝弯曲拟合
```

### 3. 方差（variance）约束
```
基于能量包络均值：
IF energy_envelope_mean < 0.1:
    → variance_bounds = [0.01, 0.5]
ELSE IF energy_envelope_mean < 0.5:
    → variance_bounds = [0.1, 2.0]
ELSE:
    → variance_bounds = [0.5, 5.0]
```

### 4. 异常处理
```
IF max_z_score > 3.0:
    → mask_outliers = True
    → outlier_indices = [检测到的异常道索引]
ELSE:
    → mask_outliers = False
    → outlier_indices = []
```

## 输出格式

必须严格按照以下JSON格式输出，不要添加任何额外文本：

```json
{
  "base_kernel_type": "RBF" | "RBF+Linear" | "RBF+Periodic",
  "rbf_config": {
    "length_scale_initial": float,
    "length_scale_bounds": [float, float],
    "variance_bounds": [float, float]
  },
  "linear_config": {
    "variance_bounds": [float, float]
  },
  "periodic_config": {
    "period_initial": float,
    "period_bounds": [float, float],
    "length_scale_bounds": [float, float]
  },
  "noise_config": {
    "noise_level_bounds": [float, float],
    "mask_outliers": boolean,
    "outlier_indices": [int, int, ...]
  },
  "reasoning": "简短解释你的决策过程"
}
```

注意：
- linear_config和periodic_config根据base_kernel_type选择性包含
- 所有数值必须是合理的物理值
- reasoning字段要简洁明了
"""

    def design_kernel(self, features: SeismicFeatures) -> KernelBlueprint:
        # 记录特征
        self.features_history.append(features)
        
        features_dict = {
            "zero_crossing_rate": features.zero_crossing_rate,
            "curvature": features.curvature,
            "linear_trend_slope": features.linear_trend_slope,
            "trend_r_squared": features.trend_r_squared,
            "avo_type": features.avo_type,
            "max_z_score": features.max_z_score,
            "outlier_indices": features.outlier_indices,
            "periodicity_score": features.periodicity_score,
            "energy_envelope_mean": features.energy_envelope_mean,
            "dominant_frequency": features.dominant_frequency,
            "bandwidth": features.bandwidth,
            "dynamic_range_db": features.dynamic_range_db,
            "phase_reversals": features.phase_reversals,
            "intercept": features.intercept,
            "gradient": features.gradient,
            "intercept_gradient_ratio": features.intercept_gradient_ratio
        }

        user_prompt = f"""请根据以下地震特征设计高斯过程核函数：

特征数据：
{json.dumps(features_dict, indent=2)}

请严格按照JSON格式输出核函数设计方案。"""

        # 记录推理开始
        inference_entry = {
            "timestamp": time.time(),
            "features": features_dict,
            "prompt": user_prompt,
            "response": None,
            "blueprint": None
        }

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.temperature,
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)
        blueprint = KernelBlueprint.from_dict(result)
        
        # 记录推理结果
        inference_entry["response"] = result
        inference_entry["blueprint"] = blueprint.to_dict()
        self.inference_log.append(inference_entry)
        self.blueprints_history.append(blueprint)
        
        return blueprint

    def design_kernel_with_fallback(
        self,
        features: SeismicFeatures,
        max_retries: int = 3
    ) -> KernelBlueprint:
        for attempt in range(max_retries):
            try:
                blueprint = self.design_kernel(features)
                blueprint.validate()
                return blueprint
            except Exception as e:
                if attempt == max_retries - 1:
                    return self._get_fallback_blueprint(features)
                continue
        return self._get_fallback_blueprint(features)

    def _get_fallback_blueprint(self, features: SeismicFeatures) -> KernelBlueprint:
        if features.avo_type == "II":
            base_kernel_type = "RBF+Linear"
            length_scale_bounds = [10, 25]
        elif features.avo_type == "III":
            base_kernel_type = "RBF"
            length_scale_bounds = [5, 15]
        else:
            base_kernel_type = "RBF"
            length_scale_bounds = [20, 50]

        if features.zero_crossing_rate >= 0.5:
            length_scale_bounds = [5, 15]
        elif features.zero_crossing_rate >= 0.2:
            length_scale_bounds = [10, 25]

        if features.curvature > 0.08:
            length_scale_bounds[1] *= 0.8

        rbf_config = RBFConstraint(
            length_scale_initial=sum(length_scale_bounds) / 2,
            length_scale_bounds=tuple(length_scale_bounds),
            variance_bounds=(0.1, 2.0)
        )

        linear_config = None
        if "Linear" in base_kernel_type:
            linear_config = LinearConstraint(variance_bounds=(0.01, 0.5))

        noise_config = NoiseStrategy(
            noise_level_bounds=(1e-6, 1e-3),
            mask_outliers=features.max_z_score > 3.0,
            outlier_indices=features.outlier_indices if features.max_z_score > 3.0 else []
        )

        return KernelBlueprint(
            base_kernel_type=base_kernel_type,
            rbf_config=rbf_config,
            linear_config=linear_config,
            noise_config=noise_config,
            reasoning="Fallback rule-based design"
        )

    def visualize_inference_process(self, save_path: str = None):
        """可视化LLM推理过程"""
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.gridspec import GridSpec
        
        if not self.inference_log:
            print("没有推理记录可可视化")
            return
        
        n_inferences = len(self.inference_log)
        
        # 创建图形
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 2, figure=fig)
        
        # 1. 特征演化图
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_feature_evolution(ax1)
        
        # 2. 核结构选择图
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_kernel_selection(ax2)
        
        # 3. 超参数演化图
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_hyperparameter_evolution(ax3)
        
        # 4. 推理时间分析
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_inference_timing(ax4)
        
        # 5. 决策理由图
        ax5 = fig.add_subplot(gs[2, :])
        self._plot_reasoning_analysis(ax5)
        
        plt.suptitle(f"LLM推理过程可视化 (共{n_inferences}次推理)", fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"推理过程可视化已保存到: {save_path}")
        else:
            plt.show()
    
    def _plot_feature_evolution(self, ax):
        """绘制特征演化图"""
        if not self.features_history:
            return
            
        features_to_plot = ['zero_crossing_rate', 'curvature', 'linear_trend_slope', 'max_z_score']
        feature_names = ['ZCR', 'Curvature', 'AVO Slope', 'Max Z-Score']
        
        for i, feature_name in enumerate(features_to_plot):
            values = [getattr(features, feature_name) for features in self.features_history]
            ax.plot(range(len(values)), values, marker='o', label=feature_names[i], linewidth=2)
        
        ax.set_xlabel('推理序号')
        ax.set_ylabel('特征值')
        ax.set_title('特征演化')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_kernel_selection(self, ax):
        """绘制核结构选择图"""
        if not self.blueprints_history:
            return
            
        kernel_types = [bp.base_kernel_type for bp in self.blueprints_history]
        unique_types = list(set(kernel_types))
        
        counts = [kernel_types.count(t) for t in unique_types]
        
        bars = ax.bar(unique_types, counts, color=['skyblue', 'lightcoral', 'lightgreen'])
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height}', ha='center', va='bottom')
        
        ax.set_xlabel('核结构类型')
        ax.set_ylabel('选择次数')
        ax.set_title('核结构选择分布')
        ax.grid(True, alpha=0.3)
    
    def _plot_hyperparameter_evolution(self, ax):
        """绘制超参数演化图"""
        if not self.blueprints_history:
            return
            
        length_scales = []
        variances = []
        
        for bp in self.blueprints_history:
            if bp.rbf_config:
                length_scales.append(bp.rbf_config.length_scale_initial)
                variances.append(bp.rbf_config.variance_initial if hasattr(bp.rbf_config, 'variance_initial') else 1.0)
        
        if length_scales:
            ax.plot(range(len(length_scales)), length_scales, marker='s', label='Length Scale', linewidth=2, color='blue')
            ax.set_ylabel('Length Scale', color='blue')
            ax.tick_params(axis='y', labelcolor='blue')
            
            ax2 = ax.twinx()
            ax2.plot(range(len(variances)), variances, marker='^', label='Variance', linewidth=2, color='red')
            ax2.set_ylabel('Variance', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            
            ax.set_xlabel('推理序号')
            ax.set_title('超参数演化')
            ax.grid(True, alpha=0.3)
    
    def _plot_inference_timing(self, ax):
        """绘制推理时间分析图"""
        if not self.inference_log:
            return
            
        timestamps = [entry['timestamp'] for entry in self.inference_log]
        start_time = timestamps[0]
        elapsed_times = [t - start_time for t in timestamps]
        
        ax.bar(range(len(elapsed_times)), elapsed_times, color='orange', alpha=0.7)
        ax.set_xlabel('推理序号')
        ax.set_ylabel('累计时间 (秒)')
        ax.set_title('推理时间分布')
        ax.grid(True, alpha=0.3)
    
    def _plot_reasoning_analysis(self, ax):
        """绘制决策理由分析图"""
        if not self.inference_log:
            return
            
        # 提取关键词
        keywords = ['RBF', 'Linear', 'Periodic', 'outlier', 'curvature', 'AVO', 'smooth', 'noise']
        keyword_counts = {kw: 0 for kw in keywords}
        
        for entry in self.inference_log:
            reasoning = entry.get('blueprint', {}).get('reasoning', '')
            for kw in keywords:
                if kw.lower() in reasoning.lower():
                    keyword_counts[kw] += 1
        
        # 过滤出出现过的关键词
        filtered_keywords = {k: v for k, v in keyword_counts.items() if v > 0}
        
        if filtered_keywords:
            ax.bar(filtered_keywords.keys(), filtered_keywords.values(), color='purple', alpha=0.7)
            ax.set_xlabel('关键词')
            ax.set_ylabel('出现次数')
            ax.set_title('决策理由关键词分析')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
    
    def save_inference_log(self, filepath: str):
        """保存推理日志到文件"""
        import json
        
        log_data = {
            "inference_log": self.inference_log,
            "features_history": [
                {
                    "zero_crossing_rate": f.zero_crossing_rate,
                    "curvature": f.curvature,
                    "linear_trend_slope": f.linear_trend_slope,
                    "avo_type": f.avo_type,
                    "max_z_score": f.max_z_score
                }
                for f in self.features_history
            ],
            "blueprints_history": [bp.to_dict() for bp in self.blueprints_history]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        print(f"推理日志已保存到: {filepath}")
    
    def load_inference_log(self, filepath: str):
        """从文件加载推理日志"""
        import json
        
        with open(filepath, 'r', encoding='utf-8') as f:
            log_data = json.load(f)
        
        self.inference_log = log_data.get('inference_log', [])
        
        # 重新创建特征对象
        self.features_history = []
        for feature_dict in log_data.get('features_history', []):
            features = SeismicFeatures(
                zero_crossing_rate=feature_dict['zero_crossing_rate'],
                curvature=feature_dict['curvature'],
                linear_trend_slope=feature_dict['linear_trend_slope'],
                avo_type=feature_dict['avo_type'],
                max_z_score=feature_dict['max_z_score'],
                # 其他特征设为默认值
                dominant_frequency=0.0, bandwidth=0.0, energy_envelope_mean=0.0,
                energy_decay_rate=0.0, dynamic_range_db=0.0, trend_r_squared=0.0,
                outlier_indices=[], phase_reversals=0, intercept=0.0, gradient=0.0,
                intercept_gradient_ratio=0.0, periodicity_score=0.0, dominant_period=None
            )
            self.features_history.append(features)
        
        # 重新创建蓝图对象
        self.blueprints_history = []
        for blueprint_dict in log_data.get('blueprints_history', []):
            self.blueprints_history.append(KernelBlueprint.from_dict(blueprint_dict))
        
        print(f"推理日志已从 {filepath} 加载")
