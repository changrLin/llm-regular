import numpy as np
from typing import Optional, Dict, List
from tqdm import tqdm

from .core.data_structures import CMPGather, SeismicFeatures
from .core.nmo import apply_nmo_correction
from .core.features import FeatureExtractor
from .agent.llm_agent import SeismicAgent
from .agent.blueprint import KernelBlueprint
from .kernel.factory import KernelFactory
from .solver.map_solver import solve_map_with_masking, compute_semblance
from .optimization.scheduler import AdaptiveKernelScheduler
from .optimization.kernel_pool import KernelPool


def process_single_cmp(
    cmp: CMPGather,
    time_windows: Optional[np.ndarray] = None,
    velocities: Optional[np.ndarray] = None,
    config: Optional[dict] = None
) -> np.ndarray:
    if time_windows is None:
        time_windows = np.arange(0.5, 2.0, 0.01)
    if velocities is None:
        velocities = np.linspace(2000, 3500, 50)

    n_time = len(time_windows)
    n_velocity = len(velocities)
    semblance = np.zeros((n_time, n_velocity))

    feature_extractor = FeatureExtractor(cmp)
    X = np.arange(cmp.n_traces).reshape(-1, 1)

    for t_idx, t in enumerate(tqdm(time_windows, desc="Processing time windows")):
        y = cmp.get_amplitudes_at_time(t)

        features = feature_extractor.extract(y, cmp.angles)

        for v_idx, v in enumerate(velocities):
            y_nmo = apply_nmo_correction(cmp, t, v)

            blueprint = _get_default_blueprint(features)

            kernel = KernelFactory.build(blueprint, cmp.n_traces)
            K = kernel(X)

            f_map, _ = solve_map_with_masking(
                K,
                X,
                y_nmo,
                outlier_indices=features.outlier_indices if features.max_z_score > 3.0 else None
            )

            semblance[t_idx, v_idx] = compute_semblance(y_nmo, f_map)

    return semblance


def process_cmp_optimized(
    cmp: CMPGather,
    agent: Optional[SeismicAgent] = None,
    time_windows: Optional[np.ndarray] = None,
    velocities: Optional[np.ndarray] = None,
    config: Optional[dict] = None
) -> np.ndarray:
    if time_windows is None:
        time_windows = np.arange(0.5, 2.0, 0.01)
    if velocities is None:
        velocities = np.linspace(2000, 3500, 50)

    n_time = len(time_windows)
    n_velocity = len(velocities)
    semblance = np.zeros((n_time, n_velocity))

    feature_extractor = FeatureExtractor(cmp)
    X = np.arange(cmp.n_traces).reshape(-1, 1)

    n_sparse_samples = config.get('n_sparse_samples', 10) if config else 10
    scheduler = AdaptiveKernelScheduler(n_sparse_samples=n_sparse_samples)
    kernel_pool = KernelPool()

    def design_callback(t: float) -> KernelBlueprint:
        y = cmp.get_amplitudes_at_time(t)
        features = feature_extractor.extract(y, cmp.angles)

        if agent is not None:
            return agent.design_kernel_with_fallback(features)
        else:
            return _get_default_blueprint(features)

    for t_idx, t in enumerate(tqdm(time_windows, desc="Processing time windows")):
        y = cmp.get_amplitudes_at_time(t)
        features = feature_extractor.extract(y, cmp.angles)

        blueprint = scheduler.get_blueprint(t, time_windows, design_callback)

        for v_idx, v in enumerate(velocities):
            y_nmo = apply_nmo_correction(cmp, t, v)

            K = kernel_pool.get_kernel_matrix(blueprint, X)

            f_map, _ = solve_map_with_masking(
                K,
                X,
                y_nmo,
                outlier_indices=features.outlier_indices if features.max_z_score > 3.0 else None
            )

            semblance[t_idx, v_idx] = compute_semblance(y_nmo, f_map)

    return semblance


def process_batch_cmps(
    cmp_dict: Dict[int, CMPGather],
    agent: SeismicAgent,
    time_windows: Optional[np.ndarray] = None,
    velocities: Optional[np.ndarray] = None,
    config: Optional[dict] = None
) -> Dict[int, np.ndarray]:
    results = {}

    for cmp_num, cmp in tqdm(cmp_dict.items(), desc="Processing CMPs"):
        results[cmp_num] = process_cmp_optimized(
            cmp,
            agent=agent,
            time_windows=time_windows,
            velocities=velocities,
            config=config
        )

    return results


def _get_default_blueprint(features: SeismicFeatures) -> KernelBlueprint:
    from .agent.blueprint import KernelBlueprint, RBFConstraint, LinearConstraint, NoiseStrategy

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
        reasoning="Default rule-based design"
    )
