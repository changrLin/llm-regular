# 🚀 Vibe Coding Document: NeuroSymbolic AVO Velocity Analysis

**To**:  Trea (Senior AI/Geophysics Engineer)  
**From**: Architect  
**Version**: v2.0 Final  
**Project Goal**: High-Resolution Velocity Analysis via LLM-Informed Gaussian Processes

---

## 1. 核心任务 (The "One-Liner")

**消除地震速度谱中的虚假红色区域。**

传统方法（Fomel AB Semblance）容易对"错误速度导致的几何畸变"产生过拟合，导致速度谱模糊（红团过大）。

**我们的方案**：
> 引入 Agent (LLM) 动态设计高斯过程的核函数 K，利用 GP 后验均值 f 拟合观测振幅 y，通过物理约束项 f^T K^-1 f 惩罚违背物理规律的拟合（如高频抖动、相位反转），最后用 f 重新计算相似度系数，从而消除虚假红色。

**关键创新**：
- **不是** "用了正则化"
- **而是** "正则化项（K矩阵）由LLM根据数据特征现场合成"
- **术语**：基于物理语义推理的正则项生成 (Physics-Informed Regularization, PINR)

---

## 2. 系统架构 (Architecture)

### Layer 1: The Legislator (Agent 🧠)

**职责**：定规矩，不执行。

**输入**：单个时间窗口的横向振幅序列特征
```
SeismicFeatures: 
  - zero_crossing_rate: 0.42      # 横向变化快慢
  - curvature:  0.08               # RMO弯曲程度
  - trend_slope: -0.05            # AVO梯度
  - avo_type: "II"                # AVO类型
  - outlier_indices: [5, 18]      # 异常道
```

**输出**：KernelBlueprint (JSON)
```json
{
  "base_kernel_type": "RBF+Linear",
  "rbf_config": {
    "length_scale_bounds": [8, 20],
    "variance_bounds": [0.5, 2.0]
  },
  "linear_config": {
    "variance_bounds": [0.01, 0.5]
  },
  "noise_config": {
    "mask_outliers": true,
    "outlier_indices": [5, 18]
  },
  "reasoning":  "检测到II类AVO，启用Linear核；curvature偏高，缩小length_scale范围拒绝弯曲拟合"
}
```

**运行频率**：稀疏（每个CMP的10个关键时间点）

---

### Layer 2: The Executor (Math Engine ⚙️)

**职责**：高速循环计算。

**双重循环结构**：
```
for t in time_windows:  # 100个时间点
    # 获取blueprint（关键点调LLM，其他点插值）
    blueprint = get_or_interpolate_blueprint(t)
    kernel = KernelFactory.build(blueprint)
    
    for v in velocities:  # 50个速度
        # NMO校正
        y = nmo_slice(cmp_gather, t, v)  # 横向30个振幅
        
        # MAP求解（核心！）
        f_map = solve_map(kernel, X, y)
        
        # 重新计算相似度
        semblance[t, v] = 1 - ||y - f_map||² / ||y||²
```

**关键机制**：
- 正确速度 → y平直 → K允许 → f完美拟合 → 残差小 → **红色保留**
- 错误速度 → y弯曲 → K拒绝 → f拟合失败 → 残差大 → **红色消失**

---

## 3. 数学原理 (Why It Works)

### 3.1 传统方法的问题

**Fomel AB Semblance**：
```
最小化： ||y - (A + B·sin²θ)||²
问题：只要能减小残差，A和B可以是任何值
结果：错误的弯曲也能用大的B强行拟合 → 虚假红色
```

### 3.2 本系统的解决方案

**GP-Regularized Semblance (MAP估计)**：
```
f_MAP = argmax p(f | y)
      = argmax [p(y | f) · p(f)]
      
其中：
- p(y | f): 似然（数据拟合项）= exp(-||y - f||² / 2σ_n²)
- p(f): 先验（物理约束）= exp(-½ f^T K^-1 f)

如果f违背K的约束（如出现K不允许的弯曲）：
→ f^T K^-1 f 爆炸
→ p(f) → 0
→ f_MAP 无法靠近y
→ 残差 ||y - f_MAP|| 很大
→ 相似度 S = 1 - ||y - f||²/||y||² → 接近0（蓝色）
```

**解析解**（Cholesky分解）：
```
K_y = K + σ_n² I
L = cholesky(K_y)           # 下三角矩阵
α = cho_solve(L, y)         # 求解 K_y α = y
f_MAP = K @ α               # 后验均值
```

---

### 3.3 物理约束的三重防御机制

| 场景 | y的特征 | K矩阵的反应 | 结果 |
|------|---------|-------------|------|
| **RMO（残余时差）** | 抛物线弯曲 | RBF核（短length_scale）拒绝弯曲 | f拟合差 → 蓝色 |
| **相位反转** | 远道振幅反号 | 正相关核惩罚负相关 | 惩罚项爆炸 → 蓝色 |
| **真实II类AVO** | 线性衰减 | RBF+Linear核允许线性趋势 | f完美拟合 → 红色 ✅ |

---

## 4. 核心模块设计 (Module Specs)

### 4.1 特征工程模块 (src. features)

**职责**：将横向振幅序列转换为Agent可理解的语义特征。

**关键**：LLM无法直接"看"波形，必须通过特征工程桥接。

**核心特征类别**：

| 特征 | 物理意义 | 对K矩阵的影响 |
|------|---------|---------------|
| `zero_crossing_rate` | 横向变化快慢 | 高 → 短length_scale |
| `curvature` | RMO弯曲程度 | 高 → 拒绝弯曲 |
| `trend_slope` | AVO梯度 | 大 → 启用Linear核 |
| `avo_type` | I/II/III类判别 | II类 → RBF+Linear |
| `outlier_indices` | 异常道位置 | 在Ω矩阵中屏蔽 |
| `periodicity_score` | 周期性（多次波） | 高 → 加Periodic核 |

**输出数据结构**：
```python
@dataclass
class SeismicFeatures:
    zero_crossing_rate: float
    curvature: float
    trend_slope: float
    avo_type: str  # "I", "II", "III"
    outlier_indices: List[int]
    periodicity_score: float
    # ...  更多特征
```

---

### 4.2 Agent决策核心 (src.agent)

**职责**：语义推理 → 核结构选择 + 超参数约束。

**决策规则示例**：
```
IF avo_type == "I" AND curvature < 0.05:
    → base_kernel_type = "RBF"
    → length_scale_bounds = [20, 50]  # 平滑，长相关

IF avo_type == "II": 
    → base_kernel_type = "RBF+Linear"  # 允许线性反转
    → length_scale_bounds = [10, 25]

IF periodicity_score > 0.7:
    → base_kernel_type = "RBF+Periodic"  # 抑制多次波

IF max_z_score > 3.0:
    → mask_outliers = True  # 屏蔽异常道
```

**输出约束的物理意义**：
- `length_scale_bounds`: 定义"多远算远"（相邻道的相关性范围）
- `variance_bounds`: 定义信号方差的物理合理区间
- `outlier_indices`: 告诉系统"这些道不可信"

---

### 4.3 核函数工厂 (src.factory)

**职责**：将Agent的JSON蓝图实例化为数学对象。

**支持的核类型**：
- `RBF`: 平滑AVO
- `Linear` (DotProduct): II类AVO线性反转
- `Periodic` (ExpSineSquared): 周期性多次波
- `Matern32`: 粗糙但连续的信号
- 组合核：`RBF+Linear`, `RBF+Periodic`

**关键实现**：
```
kernel = ConstantKernel(σ²) * RBF(ℓ, bounds=[ℓ_min, ℓ_max])
       + DotProduct(σ_b²)  # 如果Agent要求

其中bounds是Agent划定的"物理安全区"
```

**异常处理**：
```
IF blueprint.length_scale_bounds[0] < 0:
    → 抛出 PhysicsViolationError
```

---

### 4.4 MAP求解器 (src.solver)

**职责**：数值稳定地计算GP后验均值。

**核心算法**：
```
输入： kernel, X, y
输出： f_MAP

1. 计算 K = kernel(X, X)  # (n, n)
2. 加噪声： K_y = K + σ_n² I
3. Cholesky分解： L = cholesky(K_y)  # 下三角
4. 求解： α = cho_solve((L, True), y)  # K_y α = y
5. 投影： f_MAP = K @ α
```

**禁止事项**：
- ❌ 不要用 `np.linalg.inv(K)`（数值不稳定）
- ✅ 必须用 Cholesky 分解

---

### 4.5 效率优化模块 (src.scheduler)

**问题**：单个CMP有100个时间窗口，如果每个都调LLM会很慢。

**解决方案**：RBF参数插值

**策略**：
1. Agent只在关键时间点运行（每10个窗口选1个 → 10次LLM调用）
2. 中间窗口的超参数通过插值获得（线性或三次样条）

**插值公式**：
```
关键点： t₁=0.5s → ℓ₁=25. 0
        t₂=1.0s → ℓ₂=18.0

目标点： t=0.75s

线性插值：
α = (t - t₁) / (t₂ - t₁) = 0.5
ℓ(t) = (1-α)·ℓ₁ + α·ℓ₂ = 21.5
```

**效果**：
- 处理时间从 ~20秒 优化到 ~12秒
- LLM调用从 100次 减少到 10次

---
## 5. 数据结构与接口定义 (Data Structures)
Copilot said: ````markdown # 🔧 Vibe Coding Document - Technical
Markdown
# 🔧 Vibe Coding Document - Technical Supplement

**补充版本**:  v2.1  
**目的**: 补充关键技术细节，使文档可直接转化为代码

---

## 3. 数据结构与接口定义 (Data Structures)

### 3.1 核心数据结构

#### CMPGather（输入数据容器）

```python
@dataclass
class CMPGather:
    """
    单个CMP道集的数据容器。
    
    Attributes:
        data: 振幅数据，shape (n_traces, n_time_samples)
              第一维：道号（不同偏移距/角度）
              第二维：时间采样点
        offsets: 偏移距数组，shape (n_traces,)，单位：米
        angles: 入射角数组，shape (n_traces,)，单位：度
                可从offsets通过几何关系计算得到
        time_axis: 时间轴，shape (n_time_samples,)，单位：秒
        dt: 时间采样间隔，单位：秒（如 0.002 = 2ms）
        sample_rate: 采样率，Hz（如 500Hz）
    
    ���例：
        30道 × 2000时间点 × 2ms采样 = 4秒记录长度
    """
    data:  np.ndarray          # shape: (30, 2000)
    offsets: np.ndarray       # shape: (30,), e.g.  [0, 50, 100, .. ., 1450]
    angles: np.ndarray        # shape: (30,), e.g. [0, 5, 10, ..., 30]
    time_axis: np.ndarray     # shape: (2000,), e.g. [0.0, 0.002, 0.004, ..., 3.998]
    dt: float                 # 0.002
    sample_rate: float        # 500.0
    
    @property
    def n_traces(self) -> int:
        return self.data.shape[0]
    
    @property
    def n_samples(self) -> int:
        return self.data.shape[1]
    
    def get_trace(self, trace_idx: int) -> np.ndarray:
        """
        获取单道数据。
        Returns:  shape (n_time_samples,)
        """
        return self.data[trace_idx, :]
    
    def get_amplitudes_at_time(self, t: float, window_ms: float = 0) -> np.ndarray:
        """
        提取所有道在指定时间的振幅（横向切片）。
        
        Args:
            t: 目标时间（秒）
            window_ms: 时间窗口（毫秒），0表示单点，>0表示窗口平均
        
        Returns: 
            shape (n_traces,)
        
        实现逻辑：
            1. 找到最接近t的时间索引:  idx = argmin(|time_axis - t|)
            2. 如果window_ms=0: 返回 data[: , idx]
            3. 如果window_ms>0: 返回窗口内的平均值
        """
        pass
```

---

#### SeismicFeatures（特征容器）

```python
@dataclass
class SeismicFeatures:
    """
    横向振幅序列的物理特征。
    所有特征都是从30个振幅值（一个时间切片）计算得到。
    """
    # === 频率特性 ===
    zero_crossing_rate: float       # [0, 1]，横向符号变化频率
    dominant_frequency: float       # Hz，主频（通过FFT）
    bandwidth: float                # Hz，有效带宽
    
    # === 振幅特性 ===
    energy_envelope_mean: float     # Hilbert包络均值
    energy_decay_rate: float        # 指数衰减系数λ
    dynamic_range_db: float         # 最大/最小振幅比（dB）
    
    # === 趋势特性 ===
    linear_trend_slope: float       # 线性回归斜率（AVO梯度B）
    curvature:  float                # 二阶导数均值（RMO弯曲度）
    trend_r_squared: float          # 线性拟合R²，[0, 1]
    
    # === 异常检测 ===
    outlier_indices: List[int]      # 异常道的索引列表
    max_z_score: float              # 最大Z-score值
    phase_reversals: int            # 相位反转次数
    
    # === AVO语义 ===
    avo_type: str                   # "I", "II", "III", "Unknown"
    intercept: float                # AVO截距A
    gradient:  float                 # AVO梯度B
    intercept_gradient_ratio: float # A/B比值
    
    # === 周期性 ===
    periodicity_score: float        # [0, 1]，自相关峰值
    dominant_period: Optional[float] # 主周期（道数）
```

---

#### KernelBlueprint（Agent输出）

```python
@dataclass
class RBFConstraint:
    """RBF核的超参数约束"""
    length_scale_initial: float           # 初始值（区间中点）
    length_scale_bounds: Tuple[float, float]  # [min, max]
    variance_bounds: Tuple[float, float]      # [min, max]

@dataclass
class LinearConstraint:
    """Linear核的超参数约束"""
    variance_bounds: Tuple[float, float]

@dataclass
class PeriodicConstraint:
    """Periodic核的超参数约束"""
    period_initial: float
    period_bounds:  Tuple[float, float]
    length_scale_bounds: Tuple[float, float]

@dataclass
class NoiseStrategy:
    """噪声处理策略"""
    noise_level_bounds: Tuple[float, float]  # σ_n²的范围
    mask_outliers: bool                      # 是否屏蔽异常道
    outlier_indices: List[int]               # 需要屏蔽的道索引

@dataclass
class KernelBlueprint:
    """
    Agent输出的完整核函数设计方案。
    这是LLM必须严格遵守的JSON Schema。
    """
    base_kernel_type: str              # "RBF" | "RBF+Linear" | "RBF+Periodic"
    rbf_config: RBFConstraint
    linear_config: Optional[LinearConstraint] = None
    periodic_config:  Optional[PeriodicConstraint] = None
    noise_config: NoiseStrategy
    reasoning:  str                     # Agent的推理过程（可解释性）
    
    def validate(self):
        """
        校验逻辑一致性：
        - 如果base_kernel_type包含"Linear"，linear_config不能为None
        - length_scale_bounds必须满足 min < max
        - 所有bounds必须 > 0
        """
        pass
```

---

### 3.2 NMO校正的数学实现

#### 公式定义

```
NMO双曲线方程（时间域）：
    t_nmo(x, v, t0) = sqrt(t0² + x²/v²)

其中：
    x:  偏移距（米）
    v: 速度（米/秒）
    t0: 零偏移距双程走时（秒）

物理意义：
    对于水平反射层，不同偏移距的反射波到达时间
    满足双曲线关系。NMO校正就是把这条双曲线
    "拉平"到 t=t0 这条水平线上。
```

#### 实现逻辑（伪代码）

```python
def apply_nmo_correction(
    cmp: CMPGather,
    t0: float,        # 零偏移距时间（秒）
    velocity: float   # 速度（米/秒）
) -> np.ndarray:
    """
    对CMP道集在t0时刻应用NMO校正，提取校正后的振幅序列。
    
    Returns:
        y_nmo: shape (n_traces,)，校正后的振幅
    
    算法步骤：
    ─────────────────────────────────────────────
    1. 对每一道：
       a. 计算该道的NMO时间：
          t_nmo_i = sqrt(t0² + offset_i²/v²)
       
       b. 在该道的时间轴上插值，获取t_nmo_i时刻的振幅：
          trace_i = cmp.get_trace(i)
          amp_i = interpolate(cmp.time_axis, trace_i, t_nmo_i)
       
       c. 边界处理：
          如果 t_nmo_i > time_axis[-1]（超出记录长度）: 
              → amp_i = 0.0 或 NaN
    
    2. 返回:  y_nmo = [amp_0, amp_1, ..., amp_{n-1}]
    
    插值方法建议：
        - 使用 scipy.interpolate.interp1d(kind='cubic')
        - 或者 np.interp（线性插值，更快但精度稍低）
    
    数值稳定性：
        - 当 offset >> v*t0 时，t_nmo会非常大
        - 需要检查 t_nmo < t_max，否则返回0
    """
    
    n_traces = cmp.n_traces
    y_nmo = np.zeros(n_traces)
    
    for i in range(n_traces):
        offset = cmp.offsets[i]
        
        # 计算NMO时间
        t_nmo = np.sqrt(t0**2 + (offset / velocity)**2)
        
        # 边界检查
        if t_nmo > cmp.time_axis[-1]:
            y_nmo[i] = 0.0  # 超出记录长度
            continue
        
        # 插值获取振幅
        trace = cmp.get_trace(i)
        y_nmo[i] = np.interp(t_nmo, cmp.time_axis, trace)
    
    return y_nmo
```

#### 正确性验证

```
测试用例：
───────────────────────────────────────
输入：
    t0 = 1.0s
    v_true = 2500 m/s（真实速度）
    CMP道集：水平反射层生成的合成数据

预期：
    y_nmo = apply_nmo_correction(cmp, t0, v_true)
    → y_nmo应该是常数（所有道振幅一致）
    → std(y_nmo) ≈ 0（如果无噪声）

反例：
    v_wrong = 3000 m/s（错误速度）
    y_nmo = apply_nmo_correction(cmp, t0, v_wrong)
    → y_nmo呈现弯曲（欠校正）
    → curvature(y_nmo) > 0.05
```

---

## 4. Agent决策核心的实现细节

### 4.1 特征计算的精确算法

#### curvature（弯曲度）

```
算法定义：
    curvature = mean(|d²y/dx²|)

离散化实现：
    1. 计算二阶差分：
       d2y = y[i-1] - 2*y[i] + y[i+1], i=1.. n-2
    
    2. 取绝对值的均值：
       curvature = mean(|d2y|)
    
    3. 归一化（可选）：
       curvature_normalized = curvature / max(|y|)

Python实现：
    d2y = np.diff(y, n=2)  # 二阶差分
    curvature = np.mean(np.abs(d2y))

物理意义：
    - curvature < 0.01:  几乎平直（正确速度）
    - curvature > 0.1:   显著弯曲（错误速度/RMO）
```

#### zero_crossing_rate（过零率）

```
算法定义：
    ZCR = (相邻样本符号变化次数) / (总样本数 - 1)

实现：
    sign_changes = sum(sign(y[i]) != sign(y[i+1]))
    zcr = sign_changes / (len(y) - 1)

边界情况：
    - 如果y包含0值，sign(0)=0，需要特殊处理
    - 建议：sign(y[i]) = 1 if y[i]>=0 else -1

物理意义：
    - zcr < 0.2: 平滑信号（长相关）
    - zcr > 0.6: 剧烈振荡（短相关）
```

#### avo_type（AVO类型判别）

```
算法：线性回归 y = A + B*sin²θ

步骤：
    1. 计算 sin²θ: 
       sin2_theta = (angles / 180 * pi).apply(sin).pow(2)
    
    2. 线性回归：
       from sklearn.linear_model import LinearRegression
       X = sin2_theta.reshape(-1, 1)
       model = LinearRegression().fit(X, y)
       A = model.intercept_
       B = model.coef_[0]
    
    3. 判别规则：
       IF A > 0 AND B < 0:
           IF |A| > |B|:  → "I类"
           ELSE:          → "II类"
       ELIF A < 0 AND B < 0:  → "III类"
       ELSE:              → "Unknown"

物理背景：
    I类：  常规气藏，截距主导
    II类： 页岩气，梯度主导，存在反转
    III类：亮点，截距和梯度都为负
```

---

### 4.2 Agent的Prompt工程

#### System Prompt（完整版）

```markdown
# SYSTEM PROMPT

你是一位地震数据处理专家，专长是AVO分析和速度谱优化。

## 任务
根据地震振幅序列的物理特征，设计高斯过程的核函数结构和超参数约束。

## 输入特征说明

### 频率特性
- **zero_crossing_rate** (ZCR): 范围[0,1]
  - <0.2: 信号平滑，变化缓慢
  - 0.2-0.5: 中等变化
  - >0.6: 剧烈振荡，高频成分多

### 趋势特性
- **curvature**:  二阶导数均值
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
    # 理由：检测到多���波干扰
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

### 3. 方差约束（variance_bounds）
```
基本原则：根据信号动态范围

variance_initial = std(y)²
variance_bounds = [variance_initial * 0.5, variance_initial * 2.0]
```

### 4. 噪声策略
```
IF max_z_score > 3. 0:
    → mask_outliers = True
    → outlier_indices = [输入中提供的列表]

noise_level_bounds = [1e-4, 1e-2]  # 典型地震数据的噪声水平
```

## 输出格式

严格遵守以下JSON Schema（不允许任何偏离）：

```json
{
  "base_kernel_type": "RBF" | "RBF+Linear" | "RBF+Periodic",
  "rbf_config":  {
    "length_scale_initial": <float>,
    "length_scale_bounds": [<float>, <float>],
    "variance_bounds": [<float>, <float>]
  },
  "linear_config": {  // 仅当base_kernel_type包含"Linear"时
    "variance_bounds": [<float>, <float>]
  },
  "noise_config": {
    "noise_level_bounds": [<float>, <float>],
    "mask_outliers": <boolean>,
    "outlier_indices": [<int>, ...]
  },
  "reasoning": "<string>"  // 详细解释你的决策依据
}
```

## 禁止行为
- ❌ 禁止输出单一点估计（如 "length_scale":  12. 3）
- ❌ 禁止输出负数或零的bounds
- ❌ 禁止bounds不满足 min < max
- ❌ 禁止reasoning字段为空

## 示例

### 输入
```json
{
  "zero_crossing_rate": 0.15,
  "curvature":  0.02,
  "trend_slope": -0.05,
  "trend_r_squared": 0.95,
  "avo_type": "I",
  "max_z_score": 1.8,
  "outlier_indices": []
}
```

### 输出
```json
{
  "base_kernel_type": "RBF",
  "rbf_config": {
    "length_scale_initial": 30.0,
    "length_scale_bounds": [20.0, 45.0],
    "variance_bounds": [0.8, 2.5]
  },
  "noise_config": {
    "noise_level_bounds": [0.0001, 0.01],
    "mask_outliers": false,
    "outlier_indices": []
  },
  "reasoning": "特征分析：ZCR=0.15表示信号平滑；curvature=0.02轻微弯曲；trend_r_squared=0.95高度线性；avo_type=I为常规气藏。决策：选择纯RBF核，长相关length_scale=[20,45]以允许平滑拟合；无异常道，不启用屏蔽。"
}
```
```

#### Few-shot Examples（关键案例）

```python
# 案例库（至少准备5个典型场景）

EXAMPLES = [
    {
        "name": "平滑I类AVO，正确速度",
        "input": {
            "zero_crossing_rate": 0.12,
            "curvature":  0.008,
            "avo_type": "I",
            "max_z_score": 1.5
        },
        "output": {
            "base_kernel_type": "RBF",
            "rbf_config": {
                "length_scale_bounds": [25, 50]
            },
            "reasoning": "信号平滑且几乎平直，选择长相关RBF"
        }
    },
    
    {
        "name": "II类AVO，线性反转",
        "input": {
            "zero_crossing_rate":  0.35,
            "curvature":  0.04,
            "trend_slope": -0.12,
            "avo_type": "II"
        },
        "output":  {
            "base_kernel_type": "RBF+Linear",
            "rbf_config": {
                "length_scale_bounds": [10, 25]
            },
            "linear_config": {
                "variance_bounds": [0.01, 0.5]
            },
            "reasoning": "II类AVO需要Linear核捕捉反转"
        }
    },
    
    {
        "name": "错误速度，显著RMO",
        "input":  {
            "zero_crossing_rate": 0.45,
            "curvature":  0.15,  # 高曲率！
            "avo_type": "I"
        },
        "output":  {
            "base_kernel_type": "RBF",
            "rbf_config": {
                "length_scale_bounds": [5, 12]  # 短相关，拒绝弯曲
            },
            "reasoning": "高curvature表明存在RMO，缩小length_scale以拒绝弯曲拟合"
        }
    }
]
```

---

### 4.3 LLM调用的技术实现

#### 使用OpenAI API

```python
from openai import OpenAI
from pydantic import ValidationError
import json

class SeismicAgent:
    def __init__(self, model:  str = "gpt-4", temperature: float = 0.1):
        self.client = OpenAI()
        self.model = model
        self.temperature = temperature
        self.system_prompt = SYSTEM_PROMPT  # 上面定义的完整prompt
    
    def synthesize(self, features: SeismicFeatures) -> KernelBlueprint:
        """
        调用LLM生成KernelBlueprint。
        
        实现要点：
        ────────────────────────────────────────
        1. 构建user prompt（序列化features为JSON）
        2. 调用LLM API
        3. 解析JSON响应
        4. Pydantic验证（强制schema）
        5. 重试机制（最多3次）
        """
        
        # 构建user prompt
        user_prompt = self._build_user_prompt(features)
        
        # 重试循环
        for attempt in range(3):
            try:
                # 调用API
                response = self.client. chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    messages=[
                        {"role":  "system", "content": self. system_prompt},
                        {"role": "user", "content":  user_prompt}
                    ],
                    response_format={"type": "json_object"}  # 强制JSON
                )
                
                # 解析JSON
                content = response.choices[0].message.content
                blueprint_dict = json.loads(content)
                
                # Pydantic验证
                blueprint = KernelBlueprint(**blueprint_dict)
                blueprint.validate()  # 自定义验证逻辑
                
                return blueprint
            
            except (json.JSONDecodeError, ValidationError) as e:
                if attempt == 2:  # 最后一次尝试
                    # 降级到规则引擎
                    return self._fallback_rules(features)
                # 否则重试
                continue
    
    def _build_user_prompt(self, features: SeismicFeatures) -> str:
        """将features序列化为可读的JSON"""
        return f"""
请分析以下地震数据特征并设计核函数：

{json.dumps({
    "zero_crossing_rate": features.zero_crossing_rate,
    "curvature": features. curvature,
    "trend_slope": features.linear_trend_slope,
    "trend_r_squared": features. trend_r_squared,
    "avo_type": features. avo_type,
    "max_z_score": features.max_z_score,
    "outlier_indices": features.outlier_indices,
    "periodicity_score": features.periodicity_score
}, indent=2)}

请输出核函数设计方案（JSON格式）。
"""
    
    def _fallback_rules(self, features: SeismicFeatures) -> KernelBlueprint:
        """
        降级策略：基于规则的blueprint生成。
        当LLM失败时使用。
        """
        # 简化的规则引擎
        if features. avo_type == "II":
            base_type = "RBF+Linear"
        else:
            base_type = "RBF"
        
        if features.zero_crossing_rate < 0.3:
            ls_bounds = (20.0, 50.0)
        else:
            ls_bounds = (5.0, 15.0)
        
        return KernelBlueprint(
            base_kernel_type=base_type,
            rbf_config=RBFConstraint(
                length_scale_initial=(ls_bounds[0] + ls_bounds[1]) / 2,
                length_scale_bounds=ls_bounds,
                variance_bounds=(0.5, 2.0)
            ),
            noise_config=NoiseStrategy(
                noise_level_bounds=(1e-4, 1e-2),
                mask_outliers=features.max_z_score > 3.0,
                outlier_indices=features.outlier_indices
            ),
            reasoning="Fallback to rule-based strategy"
        )
```

---

## 5. MAP求解器的详细实现

### 5.1 solve_map函数签名

```python
def solve_map(
    kernel:  Kernel,           # sklearn.gaussian_process.kernels.Kernel对象
    X: np.ndarray,            # shape (n, 1)，角度或道号，必须是2D
    y: np.ndarray,            # shape (n,)，观测振幅，1D
    noise_level: float = 1e-6 # σ_n²，观测噪声方差
) -> np.ndarray:              # 返回 f_MAP，shape (n,)
    """
    计算GP的后验均值（MAP估计）。
    
    数学公式：
        f_MAP = K @ (K + σ_n²I)^{-1} @ y
              = K @ α
        其中 α = (K + σ_n²I)^{-1} @ y
    
    实现策略：
        使用Cholesky分解避免直接求逆
    
    数值稳定性：
        - 如果Cholesky失败（矩阵不正定），增大noise_level重试
        - 最多重试3次，增量为10x
    
    返回：
        f_MAP:  后验均值，shape (n,)
    """
    
    # === Step 1: 计算K矩阵 ===
    # 注意：sklearn的kernel要求X必须是2D
    K = kernel(X, X)  # shape:  (n, n)
    
    # === Step 2: 加噪声项 ===
    K_y = K + noise_level * np.eye(len(K))
    
    # === Step 3: Cholesky分解（带重试） ===
    for attempt in range(3):
        try:
            L = scipy.linalg.cholesky(K_y, lower=True)
            break
        except np.linalg. LinAlgError:
            # 矩阵不正定，增大噪声项
            noise_level *= 10
            K_y = K + noise_level * np.eye(len(K))
            if attempt == 2:
                # 最后一次仍失败，返回最小二乘解
                return np. linalg.lstsq(K, y, rcond=None)[0]
    
    # === Step 4: 求解 α = K_y^{-1} @ y ===
    alpha = scipy.linalg.cho_solve((L, True), y)
    
    # === Step 5: 计算 f_MAP = K @ α ===
    f_map = K @ alpha
    
    return f_map
```

### 5.2 异常值屏蔽的实现

```python
def solve_map_with_masking(
    kernel: Kernel,
    X: np.ndarray,              # shape (n, 1)
    y: np.ndarray,              # shape (n,)
    outlier_indices: List[int], # 需要屏蔽的索引
    base_noise:  float = 1e-6
) -> np.ndarray:
    """
    带异常值屏蔽的MAP求解。
    
    策略：
        为每个观测点分配独立的噪声水平：
        - 正常点:  σ_n² = base_noise (如 1e-6)
        - 异常点: σ_n² = 1e10 (极大噪声 = 极低权重)
    
    实现：
        K_y = K + Λ
        其中 Λ 是对角矩阵：
            Λ[i,i] = 1e10  如果 i in outlier_indices
            Λ[i,i] = base_noise  否则
    """
    
    # 构建噪声对角矩阵
    n = len(y)
    noise_diag = np.full(n, base_noise)
    noise_diag[outlier_indices] = 1e10  # 屏蔽异常点
    
    # 计算K矩阵
    K = kernel(X, X)
    K_y = K + np.diag(noise_diag)
    
    # Cholesky求解
    L = scipy.linalg.cholesky(K_y, lower=True)
    alpha = scipy.linalg.cho_solve((L, True), y)
    f_map = K @ alpha
    
    return f_map
```

---

## 6. 效率优化的边界情况处理

### 6.1 RBF参数插值的完整逻辑

```python
class AdaptiveKernelScheduler:
    """
    管理Agent调用的时空调度。
    """
    
    def interpolate_blueprint(
        self,
        blueprints_key: Dict[float, KernelBlueprint],  # 关键时间点的blueprint
        t_target: float                                 # 目标时间
    ) -> KernelBlueprint:
        """
        插值生成目标时间的blueprint。
        
        边界情况处理：
        ───────────────────────────────────────
        1. 恰好在关键点上 → 直接返回
        2. 在关键点之间 → 线性插值
        3. 小于最小关键点 → 返回最小关键点的blueprint
        4. 大于最大关键点 → 返回最大关键点的blueprint
        5. 核类型不一致 → 使用左端点的核类型
        """
        
        t_keys = sorted(blueprints_key.keys())
        
        # === 情况1: 恰好在关键点 ===
        if t_target in blueprints_key:
            return blueprints_key[t_target]
        
        # === 情况3: 小于最小值（外插） ===
        if t_target < t_keys[0]: 
            return blueprints_key[t_keys[0]]
        
        # === 情况4: 大于最大值（外插） ===
        if t_target > t_keys[-1]:
            return blueprints_key[t_keys[-1]]
        
        # === 情况2: 内插 ===
        # 找到包围区间 [t_left, t_right]
        t_left = max([t for t in t_keys if t <= t_target])
        t_right = min([t for t in t_keys if t >= t_target])
        
        bp_left = blueprints_key[t_left]
        bp_right = blueprints_key[t_right]
        
        # 插值系数
        alpha = (t_target - t_left) / (t_right - t_left)
        
        # === 情况5: 核类型冲突处理 ===
        if bp_left.base_kernel_type != bp_right.base_kernel_type:
            # 策略：使用左端点的核类型
            # 原因：时间上更接近的特征可能更可靠
            base_type = bp_left.base_kernel_type
        else:
            base_type = bp_left.base_kernel_type
        
        # 插值超参数
        ls_init = (1-alpha) * bp_left.rbf_config.length_scale_initial \
                + alpha * bp_right.rbf_config.length_scale_initial
        
        ls_min = min(bp_left.rbf_config.length_scale_bounds[0],
                     bp_right.rbf_config.length_scale_bounds[0])
        ls_max = max(bp_left.rbf_config.length_scale_bounds[1],
                     bp_right.rbf_config.length_scale_bounds[1])
        
        # 构建插值后的blueprint
        return KernelBlueprint(
            base_kernel_type=base_type,
            rbf_config=RBFConstraint(
                length_scale_initial=ls_init,
                length_scale_bounds=(ls_min, ls_max),
                variance_bounds=bp_left.rbf_config.variance_bounds
            ),
            noise_config=bp_left.noise_config,  # 噪声策略不插值
            reasoning=f"Interpolated between t={t_left:. 2f}s and t={t_right:. 2f}s (alpha={alpha:.2f})"
        )
```

---

## 7. 端到端流程示例（伪代码）

```python
# ===== 完整处理流程 =====

def process_single_cmp(cmp:  CMPGather) -> np.ndarray:
    """
    处理单个CMP道集，生成速度谱。
    
    Returns:
        semblance: shape (n_time, n_velocity)
    """
    
    # ─── 配置参数 ───
    time_windows = np.arange(0. 5, 2.0, 0.01)  # 150个时间窗口
    velocities = np.linspace(2000, 3500, 50)  # 50个速度候选
    
    # ─── 初始化组件 ───
    extractor = FeatureExtractor()
    agent = SeismicAgent(model="gpt-4")
    factory = KernelFactory()
    scheduler = AdaptiveKernelScheduler()
    
    # ─── Step 1: 稀疏Agent调用 ───
    key_times = time_windows[:: 15]  # 每15个窗口选一个 → 10个关键点
    blueprints_key = {}
    
    for t in key_times:
        # 提取横向振幅序列（未校正）
        y_raw = cmp.get_amplitudes_at_time(t)
        X = cmp.angles. reshape(-1, 1)  # shape:  (30, 1)
        
        # 特征提取
        features = extractor.extract(y_raw, X)
        
        # Agent调用（LLM）
        blueprints_key[t] = agent.synthesize(features)
    
    # ─── Step 2: 速度扫描（双重循环）───
    semblance = np.zeros((len(time_windows), len(velocities)))
    
    for i, t in enumerate(time_windows):
        # 获取blueprint（关键点直接取，其他插值）
        blueprint = scheduler.interpolate_blueprint(blueprints_key, t)
        
        # 构建kernel
        kernel = factory.build(blueprint)
        
        for j, v in enumerate(velocities):
            # NMO校正
            y_nmo = apply_nmo_correction(cmp, t, v)
            
            # MAP求解
            if blueprint.noise_config.mask_outliers:
                f_map = solve_map_with_masking(
                    kernel, X, y_nmo,
                    blueprint.noise_config.outlier_indices
                )
            else:
                f_map = solve_map(kernel, X, y_nmo)
            
            # 计算相似度
            residual = y_nmo - f_map
            semblance[i, j] = 1 - np.linalg.norm(residual)**2 / np.linalg.norm(y_nmo)**2
    
    return semblance


# ===== 使用示例 =====

# 加载数据（假设有数据读取模块）
cmp = load_cmp_from_segy("data/cmp_001.sgy")

# 处理
velocity_spectrum = process_single_cmp(cmp)

# 可视化
import matplotlib.pyplot as plt
plt. imshow(velocity_spectrum, aspect='auto', cmap='hot',
           extent=[2000, 3500, 2. 0, 0.5])
plt.xlabel("Velocity (m/s)")
plt.ylabel("Time (s)")
plt.title("NeuroSymbolic Velocity Spectrum")
plt.colorbar(label="Semblance")
plt.show()
```

---

## 8. 测试与验证

### 8.1 合成数据生成

```python
def generate_synthetic_cmp(
    true_velocity: float = 2500,      # 米/秒
    avo_type: str = "II",             # "I", "II", "III"
    add_rmo: bool = False,            # 是否添加RMO（模拟错误速度）
    noise_level: float = 0.02         # 噪声标准差
) -> CMPGather:
    """
    生成合成CMP道集用于测试。
    
    算法：
    ────────────────────────────────────────
    1. 定义几何：30道，偏移距0-1450米
    2. 定义时间轴：0-4秒，2ms采样
    3. 生成AVO曲线：
       IF avo_type == "I": 
           R(θ) = 0.5 - 0.2*sin²θ
       IF avo_type == "II": 
           R(θ) = 0.3 - 0.8*sin²θ  # 梯度主导
       IF avo_type == "III":
           R(θ) = -0.2 - 0.5*sin²θ
    
    4. 对每道：
       a. 计算NMO时间:  t_nmo = sqrt(t0² + x²/v²)
       b. 在t_nmo处插入Ricker子波，振幅=R(θ)
       c. 如果add_rmo=True，额外添加抛物线弯曲
    
    5. 添加高斯白噪声
    """
    pass  # 实现细节略，但逻辑清晰
```

### 8.2 单元测试用例

```python
# 测试1：正确速度应该产生高相似度
def test_correct_velocity():
    cmp = generate_synthetic_cmp(true_velocity=2500, avo_type="I")
    t = 1.0
    v_true = 2500
    
    y_nmo = apply_nmo_correction(cmp, t, v_true)
    
    # 期望：y_nmo应该是平直的
    curvature = np.mean(np.abs(np.diff(y_nmo, n=2)))
    assert curvature < 0.01, "正确速度下curvature应该很小"

# 测试2：错误速度应该产生低相似度
def test_wrong_velocity():
    cmp = generate_synthetic_cmp(true_velocity=2500, avo_type="I")
    t = 1.0
    v_wrong = 3000
    
    y_nmo = apply_nmo_correction(cmp, t, v_wrong)
    
    # 期望：y_nmo应该弯曲
    curvature = np.mean(np.abs(np.diff(y_nmo, n=2)))
    assert curvature > 0.05, "错误速度下curvature应该很大"

# 测试3：Agent应该识别RMO
def test_agent_detects_rmo():
    cmp = generate_synthetic_cmp(true_velocity=2500, add_rmo=True)
    y_with_rmo = cmp.get_amplitudes_at_time(1.0)
    X = cmp.angles.reshape(-1, 1)
    
    features = FeatureExtractor().extract(y_with_rmo, X)
    
    # 期望：高curvature
    assert features.curvature > 0.08
    
    # 期望：Agent输出短length_scale
    blueprint = SeismicAgent().synthesize(features)
    assert blueprint.rbf_config.length_scale_bounds[1] < 20
```

---

## 9. 性能基准与优化目标

### 9.1 性能指标

```
硬件环境（参考）：
    CPU: Intel i7-12700 (12核)
    RAM: 32GB
    Python: 3.10
    NumPy: 1.24
    SciPy: 1.10

性能目标：
    单个CMP处理时间:  < 15秒
    - 150 time × 50 velocity = 7500 次GP计算
    - 10次LLM调用（稀疏策略）
    
    性能分解：
    - LLM调用:  10 × 0.5s = 5秒
    - GP计算: 7500 × 1ms = 7.5秒
    - 其他（特征提取、插值）: 2.5秒
    - 总计: ~15秒

优化检查点：
    1. Kernel对象是否在循环外实例化？
    2. Cholesky分解是否使用了LAPACK？
    3. X的shape是否正确（避免反复reshape）？
    4. 是否有不必要的数据拷贝？
```

---

**END OF TECHNICAL SUPPLEMENT**

---

## 10. 数据流 (End-to-End Pipeline)

```
输入： 1个CMP道集 (30 traces × 2000 time samples)
       ↓
┌──────────────────────────────────────────────┐
│ 时空调度： 选择10个关键时间点                 │
│ t_key = [0.5s, 0.7s, 0.9s, .. ., 1.9s]       │
└──────────────────────────────────────────────┘
       ↓
  ┌────┴────┐
  ↓ (关键点) ↓ (其他点)
┌──────────┐  ┌──────────┐
│特征提取   │  │参数插值   │
│→Features │  │ℓ(t), σ²(t)│
└──────────┘  └──────────┘
  ↓            ↓
┌────���─────┐  │
│Agent(LLM)│  │
│→Blueprint│←─┘
└──────────┘
       ↓
┌──────────────────────────────────────────────┐
│ 双重循环：                                    │
│ for t in all_times (100):                    │
│     blueprint = get_or_interpolate(t)        │
│     kernel = Factory.build(blueprint)        │
│                                              │
│     for v in velocities (50):                │
│         y = nmo_slice(cmp, t, v)  # 横向30振幅│
│         f_map = solve_map(kernel, X, y)      │
│         S[t,v] = 1 - ||y-f||²/||y||²         │
└──────────────────────────────────────────────┘
       ↓
输出： 1张速度谱 (100 time × 50 velocity)
      处理时间： ~12秒
```

---

## 11. 为什么能消除虚假红色 (Physical Intuition)

### 11.1 场景A：RMO导致的虚假红色

**错误速度 → 同相轴弯曲**

```
传统方法：
y = [0.5, 0.48, 0.42, 0.38, 0.40, 0.45]  # 中间下凹
最小二乘拟合： f = A + B·sin²θ
→ B取很大的负值强行拟合
→ 残差小 → 红色出现 ❌

本系统：
Agent检测： curvature = 0.12 (高)
→ 设计： length_scale_bounds = [5, 12] (短相关)
→ K矩阵拒绝这种弯曲
→ f_MAP = [0.49, 0.48, 0.47, 0.46, 0.45, 0.44]  # 强制平直
→ 残差 ||y - f_MAP|| = 0.08 (大)
→ 相似度低 → 蓝色 ✅
```

### 11.2 场景B：相位反转的虚假红色

**错误速度 → 远道相位反转**

```
传统方法：
y = [0.5, 0.45, 0.38, 0.30, -0.2, -0.35]  # 远道反相
最小二乘无约束 → 能拟合
→ 残差中等 → 黄色/浅红色 ❌

本系统：
Agent检测： phase_reversals = 1
→ K矩阵的正相关假设被违背
→ f^T K^-1 f 惩罚项爆炸
→ f_MAP倾向于全正或全负（不允许反相）
→ 残差巨大 → 深蓝色 ✅
```

### 11.3 场景C：真实II类AVO（保留红色）

**正确速度 + II类反转**

```
传统方法：
y = [0.5, 0.45, 0.38, 0.30, 0.20, 0.12]  # 线性衰减
→ 能拟合 → 红色

本系统：
Agent检测： avo_type = "II", trend_slope = -0.08
→ 启用 RBF+Linear 核
→ Linear核允许线性趋势
→ f_MAP = [0.50, 0.44, 0.39, 0.31, 0.21, 0.13]
→ 残差很小 → 红色保留 ✅✅✅
```

---

## 12. 关键设计决策总结

| 设计点 | 决策 | 理由 |
|--------|------|------|
| **核心数学** | MAP估计（后验最大化） | 平衡数据拟合与物理约束 |
| **Agent角色** | 立法者（定约束，不计算） | 分离智能与速度 |
| **效率策略** | 稀疏LLM调用 + 插值 | 10x加速 |
| **数值稳定** | Cholesky分解 | 避免矩阵求逆的数值问题 |
| **可解释性** | reasoning字段 + 特征可视化 | 符合学术标准 |

---

## 13. 预期效果

**速度谱对比（时间-速度平面）**

```
传统AB Semblance:                NeuroSymbolic Semblance:
    Velocity (m/s)                  Velocity (m/s)
    2000  2500  3000                2000  2500  3000
T   ╔════════════════╗          T   ╔════════════════╗
i 0 ║ ░░▓▓▓▓▓░░      ║ ❌       i 0 ║ ░░░██░░░       ║ ✅
m   ║   ░▓▓▓▓▓▓░     ║ 模糊     m   ║    ░█░         ║ 清晰
e 1 ║    ░▓▓▓▓░      ║          e 1 ║     █          ║ 尖锐
(s) ╚════════════════╝          (s) ╚════════════════╝
    ▓ = 虚假红色（RMO/相位污染）    █ = 真实速度（物理约束后）
```

---

## 14. 核心创新再强调

**这不是**：
- ❌ 用了高斯过程
- ❌ 用了正则化
- ❌ 用了AI

**这是**：
- ✅ **正则化项（K矩阵）由LLM根据数据特征动态合成**
- ✅ **"自适应结构化正则化" (Adaptive Structural Regularization)**
- ✅ **物理先验 + 语义推理 + 数学优化的三位一体**

---

**END OF DOCUMENT**

---

*Version History:*  
v1.0 - Initial draft  
v2.0 - Final version with efficiency optimization & feature engineering details