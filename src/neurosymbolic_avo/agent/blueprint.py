from dataclasses import dataclass
from typing import Optional, List
from ..core.data_structures import RBFConstraint, LinearConstraint, PeriodicConstraint, NoiseStrategy


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

    def to_dict(self) -> dict:
        result = {
            "base_kernel_type": self.base_kernel_type,
            "rbf_config": {
                "length_scale_initial": self.rbf_config.length_scale_initial,
                "length_scale_bounds": self.rbf_config.length_scale_bounds,
                "variance_bounds": self.rbf_config.variance_bounds
            },
            "reasoning": self.reasoning
        }
        if self.linear_config:
            result["linear_config"] = {
                "variance_bounds": self.linear_config.variance_bounds
            }
        if self.periodic_config:
            result["periodic_config"] = {
                "period_initial": self.periodic_config.period_initial,
                "period_bounds": self.periodic_config.period_bounds,
                "length_scale_bounds": self.periodic_config.length_scale_bounds
            }
        if self.noise_config:
            result["noise_config"] = {
                "noise_level_bounds": self.noise_config.noise_level_bounds,
                "mask_outliers": self.noise_config.mask_outliers,
                "outlier_indices": self.noise_config.outlier_indices
            }
        return result

    @classmethod
    def from_dict(cls, data: dict) -> 'KernelBlueprint':
        rbf_config = RBFConstraint(
            length_scale_initial=data["rbf_config"]["length_scale_initial"],
            length_scale_bounds=tuple(data["rbf_config"]["length_scale_bounds"]),
            variance_bounds=tuple(data["rbf_config"]["variance_bounds"])
        )
        linear_config = None
        if "linear_config" in data:
            linear_config = LinearConstraint(
                variance_bounds=tuple(data["linear_config"]["variance_bounds"])
            )
        periodic_config = None
        if "periodic_config" in data:
            periodic_config = PeriodicConstraint(
                period_initial=data["periodic_config"]["period_initial"],
                period_bounds=tuple(data["periodic_config"]["period_bounds"]),
                length_scale_bounds=tuple(data["periodic_config"]["length_scale_bounds"])
            )
        noise_config = None
        if "noise_config" in data:
            noise_config = NoiseStrategy(
                noise_level_bounds=tuple(data["noise_config"]["noise_level_bounds"]),
                mask_outliers=data["noise_config"]["mask_outliers"],
                outlier_indices=data["noise_config"]["outlier_indices"]
            )
        return cls(
            base_kernel_type=data["base_kernel_type"],
            rbf_config=rbf_config,
            linear_config=linear_config,
            periodic_config=periodic_config,
            noise_config=noise_config,
            reasoning=data.get("reasoning", "")
        )
