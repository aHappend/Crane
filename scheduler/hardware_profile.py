from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PaperHardware72:
    """Hardware profile from Crane paper Section 7.2 (inference setup).

    Paper parameters (Section 7.2):
    - clock: 1 GHz
    - P: 1024 int8 MACs per tile
    - SRAM: 1 MB per tile
    - BW_NoC: 24 GB/s
    - BW_D: 0.5 GB/TOPS
    - E_comp,unit: 0.018 pJ/op
    - E_NoC,unit: 0.7 pJ/bit
    - E_DRAM,unit: 7.5 pJ/bit

    Notes on units used in code:
    - We treat FLOPs as "ops" in Eq.17/20.
    - 1 MB = 1e6 bytes, 1 GB = 1e9 bytes.
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

    # Eq.19/21 hop factors for DRAM-sourced traffic via NoC.
    dram_noc_hops: float = 1.0

    def tops_per_tile(self) -> float:
        return (
            float(self.macs_per_tile)
            * float(self.ops_per_mac)
            * float(self.frequency_hz)
        ) / 1e12

    def total_tops(self, num_pes: int) -> float:
        return float(num_pes) * self.tops_per_tile()

    def dram_bandwidth_gb_s(self, num_pes: int) -> float:
        return self.dram_bandwidth_per_tops_gb_s * self.total_tops(num_pes)


DEFAULT_PAPER_7_2 = PaperHardware72()


def _pj_per_bit_to_j_per_mb(pj_per_bit: float) -> float:
    # pJ/bit -> J/MB: pJ * 8e6 bit/MB * 1e-12 = pJ * 8e-6
    return float(pj_per_bit) * 8e-6


def paper_7_2_search_params(
    num_pes: int,
    profile: PaperHardware72 | None = None,
    dram_capacity_mb: float = 1e12,
) -> dict[str, float]:
    """Return SearchConfig-compatible hardware parameters (Section 7.2).

    Returned keys:
    - sram_capacity: MB
    - dram_capacity: MB
    - noc_bandwidth: MB/s
    - dram_bandwidth: MB/s
    - noc_energy_per_unit: J/MB
    - dram_energy_per_unit: J/MB
    - dram_noc_hops: hop count
    - compute_power_per_tile: op/s
    - compute_energy_per_op: J/op

    Compatibility key:
    - traffic_energy_per_unit: J/MB, equals (E_NoC*hops + E_DRAM)
    """

    p = profile or DEFAULT_PAPER_7_2

    noc_bw_mb_s = float(p.noc_bandwidth_gb_s) * 1000.0
    dram_bw_mb_s = float(p.dram_bandwidth_gb_s(num_pes)) * 1000.0

    noc_j_per_mb = _pj_per_bit_to_j_per_mb(p.noc_energy_pj_per_bit)
    dram_j_per_mb = _pj_per_bit_to_j_per_mb(p.dram_energy_pj_per_bit)
    traffic_j_per_mb = float(p.dram_noc_hops) * noc_j_per_mb + dram_j_per_mb

    return {
        "sram_capacity": float(num_pes) * float(p.sram_per_tile_mb),
        "dram_capacity": float(dram_capacity_mb),
        "noc_bandwidth": noc_bw_mb_s,
        "dram_bandwidth": dram_bw_mb_s,
        "noc_energy_per_unit": noc_j_per_mb,
        "dram_energy_per_unit": dram_j_per_mb,
        "dram_noc_hops": float(p.dram_noc_hops),
        "traffic_energy_per_unit": traffic_j_per_mb,
        "compute_power_per_tile": (
            float(p.macs_per_tile) * float(p.ops_per_mac) * float(p.frequency_hz)
        ),
        "compute_energy_per_op": float(p.compute_energy_pj_per_op) * 1e-12,
    }
