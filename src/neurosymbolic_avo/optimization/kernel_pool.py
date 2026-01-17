import numpy as np
from typing import Dict, Tuple, Optional
from ..agent.blueprint import KernelBlueprint
from ..kernel.factory import KernelFactory


class KernelPool:
    def __init__(self, quantization_digits: int = 2):
        self.quantization_digits = quantization_digits
        self.pool: Dict[str, object] = {}

    def _generate_key(
        self,
        blueprint: KernelBlueprint,
        n_samples: int
    ) -> str:
        key_parts = [
            blueprint.base_kernel_type,
            f"ls_{round(blueprint.rbf_config.length_scale_initial, self.quantization_digits)}",
            f"var_{round((blueprint.rbf_config.variance_bounds[0] + blueprint.rbf_config.variance_bounds[1]) / 2, self.quantization_digits)}",
            f"n_{n_samples}"
        ]

        if blueprint.linear_config is not None:
            key_parts.append(f"lin_{round((blueprint.linear_config.variance_bounds[0] + blueprint.linear_config.variance_bounds[1]) / 2, self.quantization_digits)}")

        if blueprint.periodic_config is not None:
            key_parts.append(f"per_{round(blueprint.periodic_config.period_initial, self.quantization_digits)}")

        return "_".join(key_parts)

    def get_kernel(
        self,
        blueprint: KernelBlueprint,
        n_samples: int
    ) -> object:
        key = self._generate_key(blueprint, n_samples)

        if key not in self.pool:
            kernel = KernelFactory.build(blueprint, n_samples)
            self.pool[key] = kernel

        return self.pool[key]

    def get_kernel_matrix(
        self,
        blueprint: KernelBlueprint,
        X: np.ndarray
    ) -> np.ndarray:
        n_samples = len(X)
        kernel = self.get_kernel(blueprint, n_samples)
        return kernel(X)

    def clear(self):
        self.pool.clear()

    def size(self) -> int:
        return len(self.pool)

    def get_stats(self) -> Dict[str, int]:
        stats = {}
        for key in self.pool.keys():
            base_type = key.split("_")[0]
            stats[base_type] = stats.get(base_type, 0) + 1
        return stats
