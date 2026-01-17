from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class CMPGather:
    data: np.ndarray
    offsets: np.ndarray
    angles: np.ndarray
    time_axis: np.ndarray
    dt: float
    sample_rate: float

    @property
    def n_traces(self) -> int:
        return self.data.shape[0]

    @property
    def n_samples(self) -> int:
        return self.data.shape[1]

    def get_trace(self, trace_idx: int) -> np.ndarray:
        return self.data[trace_idx, :]

    def get_amplitudes_at_time(self, t: float, window_ms: float = 0) -> np.ndarray:
        idx = np.argmin(np.abs(self.time_axis - t))
        if window_ms == 0:
            return self.data[:, idx]
        else:
            window_samples = int(window_ms / 1000 / self.dt)
            start = max(0, idx - window_samples // 2)
            end = min(self.n_samples, idx + window_samples // 2)
            return np.mean(self.data[:, start:end], axis=1)


@dataclass
class SeismicFeatures:
    zero_crossing_rate: float
    dominant_frequency: float
    bandwidth: float
    energy_envelope_mean: float
    energy_decay_rate: float
    dynamic_range_db: float
    linear_trend_slope: float
    curvature: float
    trend_r_squared: float
    outlier_indices: List[int]
    max_z_score: float
    phase_reversals: int
    avo_type: str
    intercept: float
    gradient: float
    intercept_gradient_ratio: float
    periodicity_score: float
    dominant_period: Optional[float]


@dataclass
class RBFConstraint:
    length_scale_initial: float
    length_scale_bounds: tuple
    variance_bounds: tuple


@dataclass
class LinearConstraint:
    variance_bounds: tuple


@dataclass
class PeriodicConstraint:
    period_initial: float
    period_bounds: tuple
    length_scale_bounds: tuple


@dataclass
class NoiseStrategy:
    noise_level_bounds: tuple
    mask_outliers: bool
    outlier_indices: List[int]


@dataclass
class KernelBlueprint:
    base_kernel_type: str
    rbf_config: RBFConstraint
    linear_config: Optional[LinearConstraint] = None
    periodic_config: Optional[PeriodicConstraint] = None
    noise_config: Optional[NoiseStrategy] = None
    reasoning: str = ""

    def validate(self):
        if self.linear_config is None and "Linear" in self.base_kernel_type:
            raise ValueError("Linear config required for Linear kernel")
        if self.periodic_config is None and "Periodic" in self.base_kernel_type:
            raise ValueError("Periodic config required for Periodic kernel")
        if self.rbf_config.length_scale_bounds[0] >= self.rbf_config.length_scale_bounds[1]:
            raise ValueError("length_scale_bounds must be min < max")
        if self.rbf_config.variance_bounds[0] <= 0 or self.rbf_config.variance_bounds[1] <= 0:
            raise ValueError("variance_bounds must be positive")
