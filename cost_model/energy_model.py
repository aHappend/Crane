def compute_energy(latency: float, power: float) -> float:
    if latency < 0 or power < 0:
        raise ValueError("latency and power must be >= 0")
    return latency * power


def communication_energy(data_size: float, energy_per_unit: float) -> float:
    if data_size < 0 or energy_per_unit < 0:
        raise ValueError("data_size and energy_per_unit must be >= 0")
    return data_size * energy_per_unit


def edp(latency: float, energy: float) -> float:
    return latency * energy
