from .core.data_structures import CMPGather, SeismicFeatures
from .core.nmo import apply_nmo_correction, apply_nmo_correction_vectorized
from .core.features import FeatureExtractor
from .core.avo_classification import classify_avo, AVOType, AVOClassification

from .agent.llm_agent import SeismicAgent
from .agent.blueprint import KernelBlueprint, RBFConstraint, LinearConstraint, PeriodicConstraint, NoiseStrategy

from .kernel.factory import KernelFactory

from .solver.map_solver import solve_map, solve_map_with_masking, compute_semblance, compute_log_marginal_likelihood

from .optimization.scheduler import AdaptiveKernelScheduler
from .optimization.interpolator import SplineInterpolator
from .optimization.kernel_pool import KernelPool

from .io.segy_reader import load_cmp_from_segy
from .io.synthetic import generate_synthetic_cmp, generate_synthetic_dataset

from .visualization.velocity_spectrum import plot_velocity_spectrum, plot_comparison, plot_feature_distribution

from .pipeline import process_single_cmp, process_cmp_optimized, process_batch_cmps

__version__ = "0.1.0"
__author__ = "Your Name"

__all__ = [
    "CMPGather",
    "SeismicFeatures",
    "apply_nmo_correction",
    "apply_nmo_correction_vectorized",
    "FeatureExtractor",
    "classify_avo",
    "AVOType",
    "AVOClassification",
    "SeismicAgent",
    "KernelBlueprint",
    "RBFConstraint",
    "LinearConstraint",
    "PeriodicConstraint",
    "NoiseStrategy",
    "KernelFactory",
    "solve_map",
    "solve_map_with_masking",
    "compute_semblance",
    "compute_log_marginal_likelihood",
    "AdaptiveKernelScheduler",
    "SplineInterpolator",
    "KernelPool",
    "load_cmp_from_segy",
    "generate_synthetic_cmp",
    "generate_synthetic_dataset",
    "plot_velocity_spectrum",
    "plot_comparison",
    "plot_feature_distribution",
    "process_single_cmp",
    "process_cmp_optimized",
    "process_batch_cmps",
]
