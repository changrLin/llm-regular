import numpy as np
from typing import Optional
from .data_structures import CMPGather


def apply_nmo_correction(
    cmp: CMPGather,
    t0: float,
    velocity: float
) -> np.ndarray:
    n_traces = cmp.n_traces
    y_nmo = np.zeros(n_traces)

    for i in range(n_traces):
        offset = cmp.offsets[i]
        t_nmo = np.sqrt(t0**2 + (offset / velocity)**2)

        if t_nmo > cmp.time_axis[-1]:
            y_nmo[i] = 0.0
            continue

        trace = cmp.get_trace(i)
        y_nmo[i] = np.interp(t_nmo, cmp.time_axis, trace)

    return y_nmo


def apply_nmo_correction_vectorized(
    cmp: CMPGather,
    t0: float,
    velocity: float
) -> np.ndarray:
    offsets = cmp.offsets
    t_nmo = np.sqrt(t0**2 + (offsets / velocity)**2)
    y_nmo = np.zeros(cmp.n_traces)

    for i in range(cmp.n_traces):
        if t_nmo[i] > cmp.time_axis[-1]:
            y_nmo[i] = 0.0
        else:
            trace = cmp.get_trace(i)
            y_nmo[i] = np.interp(t_nmo[i], cmp.time_axis, trace)

    return y_nmo
