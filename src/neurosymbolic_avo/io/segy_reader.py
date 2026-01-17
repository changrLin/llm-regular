import numpy as np
import segyio
from typing import Optional
from ..core.data_structures import CMPGather


def load_cmp_from_segy(
    filepath: str,
    cmp_number: int,
    inline_byte: int = 189,
    xline_byte: int = 193,
    offset_byte: int = 37
) -> CMPGather:
    with segyio.open(filepath, ignore_geometry=True) as f:
        sample_rate = segyio.tools.dt(f) / 1000
        n_samples = len(f.trace[0])
        dt = sample_rate / 1000
        time_axis = np.arange(n_samples) * dt

        traces_data = []
        offsets = []

        for trace_idx, trace_header in enumerate(f.header):
            trace_cmp = trace_header[segyio.TraceField.CDP]

            if trace_cmp == cmp_number:
                trace_data = f.trace[trace_idx]
                traces_data.append(trace_data)

                offset = trace_header[segyio.TraceField.offset]
                offsets.append(abs(offset))

        if not traces_data:
            raise ValueError(f"No traces found for CMP {cmp_number}")

        data = np.array(traces_data)
        offsets = np.array(offsets)

        sort_idx = np.argsort(offsets)
        data = data[sort_idx, :]
        offsets = offsets[sort_idx]

        v_avg = 2500
        t_avg = time_axis[n_samples // 2]
        depth_approx = v_avg * t_avg / 2

        angles = np.rad2deg(np.arctan(offsets / (2 * depth_approx)))

        return CMPGather(
            data=data,
            offsets=offsets,
            angles=angles,
            time_axis=time_axis,
            dt=dt,
            sample_rate=1.0 / dt
        )
