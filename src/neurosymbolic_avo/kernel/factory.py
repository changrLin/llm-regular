import numpy as np
from sklearn.gaussian_process.kernels import (
    RBF,
    DotProduct,
    ExpSineSquared,
    ConstantKernel as C,
    WhiteKernel
)
from typing import Optional
from ..agent.blueprint import KernelBlueprint


class KernelFactory:
    @staticmethod
    def build(blueprint: KernelBlueprint, n_samples: int) -> object:
        base_kernel = KernelFactory._build_base_kernel(blueprint)

        if blueprint.noise_config is not None:
            noise_level = (blueprint.noise_config.noise_level_bounds[0] +
                         blueprint.noise_config.noise_level_bounds[1]) / 2
            kernel = base_kernel + WhiteKernel(noise_level=noise_level)
        else:
            kernel = base_kernel

        return kernel

    @staticmethod
    def _build_base_kernel(blueprint: KernelBlueprint) -> object:
        rbf = C(
            constant_value=(blueprint.rbf_config.variance_bounds[0] +
                         blueprint.rbf_config.variance_bounds[1]) / 2,
            constant_value_bounds=blueprint.rbf_config.variance_bounds
        ) * RBF(
            length_scale=blueprint.rbf_config.length_scale_initial,
            length_scale_bounds=blueprint.rbf_config.length_scale_bounds
        )

        if blueprint.base_kernel_type == "RBF":
            return rbf

        elif blueprint.base_kernel_type == "RBF+Linear":
            if blueprint.linear_config is None:
                raise ValueError("Linear config required for RBF+Linear kernel")
            linear = DotProduct(
                sigma_0_bounds=blueprint.linear_config.variance_bounds
            )
            return rbf + linear

        elif blueprint.base_kernel_type == "RBF+Periodic":
            if blueprint.periodic_config is None:
                raise ValueError("Periodic config required for RBF+Periodic kernel")
            periodic = ExpSineSquared(
                periodicity=blueprint.periodic_config.period_initial,
                periodicity_bounds=blueprint.periodic_config.period_bounds,
                length_scale=blueprint.periodic_config.length_scale_bounds[0],
                length_scale_bounds=blueprint.periodic_config.length_scale_bounds
            )
            return rbf + periodic

        else:
            raise ValueError(f"Unknown kernel type: {blueprint.base_kernel_type}")

    @staticmethod
    def compute_kernel_matrix(kernel: object, X: np.ndarray) -> np.ndarray:
        return kernel(X)

    @staticmethod
    def apply_outlier_mask(
        kernel_matrix: np.ndarray,
        outlier_indices: list
    ) -> np.ndarray:
        if not outlier_indices:
            return kernel_matrix

        masked_kernel = kernel_matrix.copy()
        for idx in outlier_indices:
            masked_kernel[idx, :] = 0.0
            masked_kernel[:, idx] = 0.0
            masked_kernel[idx, idx] = 1.0

        return masked_kernel
