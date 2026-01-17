import numpy as np
from typing import Optional, Tuple
from ..core.data_structures import CMPGather


def generate_synthetic_cmp(
    n_traces: int = 30,
    n_samples: int = 2000,
    dt: float = 0.002,
    t0: float = 1.0,
    true_velocity: float = 2500,
    avo_type: str = "II",
    noise_level: float = 0.05,
    max_offset: float = 1450.0
) -> CMPGather:
    offsets = np.linspace(0, max_offset, n_traces)
    time_axis = np.arange(n_samples) * dt
    data = np.zeros((n_traces, n_samples))

    t0_idx = int(t0 / dt)
    window_samples = int(0.1 / dt)
    start_idx = max(0, t0_idx - window_samples // 2)
    end_idx = min(n_samples, t0_idx + window_samples // 2)

    angles = np.rad2deg(np.arctan(offsets / (2 * true_velocity * t0)))

    if avo_type == "I":
        intercept = 0.1
        gradient = -0.05
    elif avo_type == "II":
        intercept = 0.02
        gradient = -0.12
    elif avo_type == "III":
        intercept = -0.08
        gradient = -0.1
    else:
        intercept = 0.05
        gradient = -0.05

    for i, angle in enumerate(angles):
        sin2_theta = np.sin(np.deg2rad(angle))**2
        amplitude = intercept + gradient * sin2_theta

        wavelet = ricker_wavelet(window_samples, 30.0)
        wavelet = wavelet * amplitude

        for j in range(start_idx, end_idx):
            idx = j - start_idx
            if 0 <= idx < len(wavelet):
                data[i, j] += wavelet[idx]

    noise = np.random.normal(0, noise_level, (n_traces, n_samples))
    data += noise

    return CMPGather(
        data=data,
        offsets=offsets,
        angles=angles,
        time_axis=time_axis,
        dt=dt,
        sample_rate=1.0 / dt
    )


def ricker_wavelet(length: int, frequency: float) -> np.ndarray:
    t = np.linspace(-1, 1, length)
    factor = (np.pi * frequency * t) ** 2
    wavelet = (1 - 2 * factor) * np.exp(-factor)
    return wavelet


def generate_synthetic_dataset(
    n_cmps: int = 10,
    cmp_range: Tuple[int, int] = (1000, 2000),
    avo_types: Optional[list] = None,
    **kwargs
) -> dict:
    if avo_types is None:
        avo_types = ["I", "II", "III"]

    cmp_data = {}
    cmp_numbers = np.linspace(cmp_range[0], cmp_range[1], n_cmps, dtype=int)

    for i, cmp_num in enumerate(cmp_numbers):
        avo_type = avo_types[i % len(avo_types)]
        cmp = generate_synthetic_cmp(avo_type=avo_type, **kwargs)
        cmp_data[cmp_num] = cmp

    return cmp_data
