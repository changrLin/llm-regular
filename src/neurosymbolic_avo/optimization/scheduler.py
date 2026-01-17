import numpy as np
from typing import List, Dict, Optional
from ..agent.blueprint import KernelBlueprint
from .interpolator import SplineInterpolator


class AdaptiveKernelScheduler:
    def __init__(
        self,
        n_sparse_samples: int = 10,
        interpolation_method: str = "cubic"
    ):
        self.n_sparse_samples = n_sparse_samples
        self.interpolator = SplineInterpolator(method=interpolation_method)
        self.blueprint_cache: Dict[float, KernelBlueprint] = {}

    def get_blueprint(
        self,
        t: float,
        time_windows: np.ndarray,
        design_callback
    ) -> KernelBlueprint:
        if t in self.blueprint_cache:
            return self.blueprint_cache[t]

        sparse_indices = self._select_sparse_indices(time_windows)
        sparse_times = time_windows[sparse_indices]

        if len(self.blueprint_cache) == 0:
            for sparse_t in sparse_times:
                blueprint = design_callback(sparse_t)
                self.blueprint_cache[sparse_t] = blueprint

        if t in self.blueprint_cache:
            return self.blueprint_cache[t]

        cached_times = np.array(list(self.blueprint_cache.keys()))
        cached_blueprints = [self.blueprint_cache[t] for t in cached_times]

        if len(cached_times) >= 2:
            sort_idx = np.argsort(cached_times)
            cached_times = cached_times[sort_idx]
            cached_blueprints = [cached_blueprints[i] for i in sort_idx]

            interpolated_blueprints = self.interpolator.interpolate_blueprint(
                cached_times,
                cached_blueprints,
                np.array([t])
            )
            blueprint = interpolated_blueprints[0]
        else:
            blueprint = design_callback(t)

        self.blueprint_cache[t] = blueprint
        return blueprint

    def _select_sparse_indices(self, time_windows: np.ndarray) -> np.ndarray:
        n_windows = len(time_windows)
        if n_windows <= self.n_sparse_samples:
            return np.arange(n_windows)

        indices = np.linspace(0, n_windows - 1, self.n_sparse_samples, dtype=int)
        return indices

    def clear_cache(self):
        self.blueprint_cache.clear()

    def get_cached_times(self) -> List[float]:
        return sorted(self.blueprint_cache.keys())
