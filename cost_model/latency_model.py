def compute_latency(workload: float, util: float, tiles: int, compute_power: float) -> float:
    if util <= 0 or tiles <= 0 or compute_power <= 0:
        raise ValueError("util, tiles, and compute_power must be > 0")
    return workload / (util * tiles * compute_power)


def noc_latency(data_size: float, bandwidth: float) -> float:
    if bandwidth <= 0:
        raise ValueError("bandwidth must be > 0")
    return data_size / bandwidth
