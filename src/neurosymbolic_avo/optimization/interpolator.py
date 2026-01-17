import numpy as np
from scipy.interpolate import CubicSpline, interp1d
from typing import List, Dict, Optional
from ..agent.blueprint import KernelBlueprint


class SplineInterpolator:
    def __init__(self, method: str = "cubic"):
        self.method = method

    def interpolate_length_scale(
        self,
        key_times: np.ndarray,
        key_values: np.ndarray,
        target_times: np.ndarray
    ) -> np.ndarray:
        if len(key_times) < 2:
            return np.full_like(target_times, key_values[0])

        if self.method == "cubic":
            if len(key_times) >= 4:
                spline = CubicSpline(key_times, key_values, bc_type='natural')
                return spline(target_times)
            else:
                interp = interp1d(key_times, key_values, kind='linear', fill_value='extrapolate')
                return interp(target_times)
        else:
            interp = interp1d(key_times, key_values, kind=self.method, fill_value='extrapolate')
            return interp(target_times)

    def interpolate_variance(
        self,
        key_times: np.ndarray,
        key_values: np.ndarray,
        target_times: np.ndarray
    ) -> np.ndarray:
        return self.interpolate_length_scale(key_times, key_values, target_times)

    def interpolate_blueprint(
        self,
        key_times: np.ndarray,
        key_blueprints: List[KernelBlueprint],
        target_times: np.ndarray
    ) -> List[KernelBlueprint]:
        if len(key_times) < 2:
            return [key_blueprints[0]] * len(target_times)

        length_scales = [bp.rbf_config.length_scale_initial for bp in key_blueprints]
        variances = [(bp.rbf_config.variance_bounds[0] + bp.rbf_config.variance_bounds[1]) / 2
                     for bp in key_blueprints]

        interpolated_length_scales = self.interpolate_length_scale(key_times, length_scales, target_times)
        interpolated_variances = self.interpolate_variance(key_times, variances, target_times)

        interpolated_blueprints = []
        for i, t in enumerate(target_times):
            blueprint = key_blueprints[np.argmin(np.abs(key_times - t))]
            new_blueprint = KernelBlueprint(
                base_kernel_type=blueprint.base_kernel_type,
                rbf_config=blueprint.rbf_config,
                linear_config=blueprint.linear_config,
                periodic_config=blueprint.periodic_config,
                noise_config=blueprint.noise_config,
                reasoning=blueprint.reasoning
            )
            new_blueprint.rbf_config.length_scale_initial = float(interpolated_length_scales[i])
            interpolated_blueprints.append(new_blueprint)

        return interpolated_blueprints
