from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PaperHardware72:
    """Hardware profile from Crane paper Section 7.2 (NVDLA-style tiles).

    Units in this dataclass follow the paper notations:
    - frequency_hz: Hz
    - macs_per_tile: int8 MAC count per tile
    - sram_per_tile_mb: MB per tile
    - noc_bandwidth_gb_s: GB/s
    - dram_bandwidth_per_tops_gb_s: GB/s per TOPS
    - compute_energy_pj_per_op: pJ/op
    - noc_energy_pj_per_bit: pJ/bit
    - dram_energy_pj_per_bit: pJ/bit
    """

    frequency_hz: float = 1e9
    macs_per_tile: int = 1024
    ops_per_mac: int = 2
    sram_per_tile_mb: float = 1.0
    noc_bandwidth_gb_s: float = 24.0
    dram_bandwidth_per_tops_gb_s: float = 0.5
    compute_energy_pj_per_op: float = 0.018
    noc_energy_pj_per_bit: float = 0.7
    dram_energy_pj_per_bit: float = 7.5

    def total_tops(self, num_pes: int) -> float:
        return (float(num_pes) * float(self.macs_per_tile) * float(self.ops_per_mac) * float(self.frequency_hz)) / 1e12

    def dram_bandwidth_gb_s(self, num_pes: int) -> float:
        return self.dram_bandwidth_per_tops_gb_s * self.total_tops(num_pes)


DEFAULT_PAPER_7_2 = PaperHardware72()


def paper_7_2_search_params(
    num_pes: int,
    profile: PaperHardware72 | None = None,
    include_noc_energy_in_traffic: bool = True,
    dram_capacity_mb: float = 1e12,
) -> dict[str, float]:
    """Return SearchConfig-compatible hardware parameters.

    Returned keys map directly to SearchConfig fields:
    - sram_capacity (MB)
    - dram_capacity (MB)
    - noc_bandwidth (MB/s)
    - dram_energy_per_unit (J/MB)
    - compute_power_per_tile (op/s)
    - compute_energy_per_op (J/op)
    """

    p = profile or DEFAULT_PAPER_7_2

    # Eq.7.2-style: DRAM BW scales with TOPS; traffic bottleneck by min(NoC, DRAM).
    dram_bw_gb_s = p.dram_bandwidth_gb_s(num_pes)
    eff_bw_gb_s = min(p.noc_bandwidth_gb_s, dram_bw_gb_s)

    # Convert pJ/bit -> J/MB: pJ * 8e6 bit/MB * 1e-12 = pJ * 8e-6.
    traffic_energy_pj_per_bit = p.dram_energy_pj_per_bit
    if include_noc_energy_in_traffic:
        traffic_energy_pj_per_bit += p.noc_energy_pj_per_bit

    return {
        "sram_capacity": float(num_pes) * p.sram_per_tile_mb,
        "dram_capacity": float(dram_capacity_mb),
        "noc_bandwidth": eff_bw_gb_s * 1000.0,
        "dram_energy_per_unit": traffic_energy_pj_per_bit * 8e-6,
        "compute_power_per_tile": float(p.macs_per_tile) * float(p.ops_per_mac) * float(p.frequency_hz),
        "compute_energy_per_op": p.compute_energy_pj_per_op * 1e-12,
    }
