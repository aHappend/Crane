from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from scheduler.scheduling_table import SchedulingTable
from scheduler.optimization import SolverReport, configure_solver, solver_report
from scheduler.traffic import clamped_integer, edge_intervals, estimate_traffic, parents_by_child

try:
    from ortools.linear_solver import pywraplp
except Exception:  # pragma: no cover
    pywraplp = None


@dataclass
class MemoryTable:
    """Track SRAM and DRAM lower bounds per state and block."""

    sram: np.ndarray
    dram: np.ndarray

    @classmethod
    def zeros(cls, num_states: int, num_blocks: int) -> "MemoryTable":
        return cls(
            sram=np.zeros((num_states, num_blocks), dtype=float),
            dram=np.zeros((num_states, num_blocks), dtype=float),
        )

    def set_sram(self, state: int, block: int, value: float) -> None:
        self.sram[state, block] = value

    def set_dram(self, state: int, block: int, value: float) -> None:
        self.dram[state, block] = value


@dataclass
class MemoryOptimizationResult:
    table: MemoryTable
    objective: float
    solver_name: str
    report: SolverReport | None = None


def build_memory_table(sct: SchedulingTable, sram_keep_ratio: float = 0.6) -> MemoryTable:
    """Fallback heuristic for MeT when MILP is unavailable."""

    met = MemoryTable.zeros(sct.num_states, sct.num_blocks)
    for i in range(sct.num_states):
        for j in range(sct.num_blocks):
            cumulative = sct.get(i, j)
            met.set_sram(i, j, cumulative * (1.0 - sram_keep_ratio))
            met.set_dram(i, j, cumulative * 0.75)
    return met


def _estimate_dep_traffic_from_tables(sct, met, block_volumes, block_dependencies,
                                    block_input_volumes=None, block_weight_volumes=None, edge_volumes=None):
    return estimate_traffic(sct, met, block_volumes, block_dependencies,
                            block_input_volumes, block_weight_volumes, edge_volumes)


def _optimize_memory_with_ortools(
    sct: SchedulingTable,
    block_volumes: Sequence[float],
    block_dependencies: Sequence[tuple[int, int]],
    sram_capacity: float,
    dram_capacity: float,
    noc_bandwidth: float,
    dram_bandwidth: float | None,
    noc_energy_per_unit: float,
    dram_energy_per_unit: float,
    dram_noc_hops: float,
    weight_latency: float,
    weight_energy: float,
    use_edp_objective: bool,
    enforce_end_dram_dependency: bool,
    end_dram_upper_bounds: Sequence[int] | None,
    force_final_sram_empty: bool,
    solver_time_limit_s: float,
    block_input_volumes: Sequence[float] | None,
    block_weight_volumes: Sequence[float] | None,
    edge_volumes: dict[tuple[int, int], float] | None,
) -> MemoryOptimizationResult:
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if solver is None:
        raise RuntimeError("OR-Tools SCIP solver is not available")

    configure_solver(solver, solver_time_limit_s)
    n_states = sct.num_states
    n_blocks = sct.num_blocks

    if end_dram_upper_bounds is not None and len(end_dram_upper_bounds) != n_blocks:
        raise ValueError("end_dram_upper_bounds length must equal number of blocks")

    ms = []
    md = []
    for i in range(n_states):
        ms_row = []
        md_row = []
        for j in range(n_blocks):
            ub = int(round(sct.get(i, j)))
            ms_row.append(solver.IntVar(0, ub, f"ms_{i}_{j}"))
            md_row.append(solver.IntVar(0, ub, f"md_{i}_{j}"))
        ms.append(ms_row)
        md.append(md_row)

    # Eq.8 monotonicity across states.
    for i in range(n_states - 1):
        for j in range(n_blocks):
            solver.Add(ms[i][j] <= ms[i + 1][j])
            solver.Add(md[i][j] <= md[i + 1][j])

    # Eq.9 bounded by ScT.
    for i in range(n_states):
        for j in range(n_blocks):
            solver.Add(ms[i][j] <= int(round(sct.get(i, j))))
            solver.Add(md[i][j] <= int(round(sct.get(i, j))))

    # Eq.10 dependency-safe memory availability.
    # min(MeT_S, MeT_D) <= ScT_{i-1,child} is an OR-constraint.
    # Linearize with one binary gate variable.
    for parent, child in block_dependencies:
        if parent < 0 or child < 0 or parent >= n_blocks or child >= n_blocks:
            continue
        for i in range(1, n_states):
            child_done_prev = int(round(sct.get(i - 1, child)))
            parent_done_prev = int(round(sct.get(i - 1, parent)))
            big_m = max(0, parent_done_prev - child_done_prev)
            gate = solver.IntVar(0, 1, f"eq10_gate_{i}_{parent}_{child}")
            solver.Add(ms[i - 1][parent] <= child_done_prev + big_m * gate)
            solver.Add(md[i - 1][parent] <= child_done_prev + big_m * (1 - gate))

    # Eq.13 (forward training): end-state DRAM order by dependency.
    if enforce_end_dram_dependency and n_states > 0:
        end_i = n_states - 1
        for parent, child in block_dependencies:
            if parent < 0 or child < 0 or parent >= n_blocks or child >= n_blocks:
                continue
            solver.Add(md[end_i][parent] >= md[end_i][child])

    if n_states > 0 and end_dram_upper_bounds is not None:
        end_i = n_states - 1
        for j in range(n_blocks):
            ub = max(0, min(int(round(sct.get(end_i, j))), int(end_dram_upper_bounds[j])))
            solver.Add(md[end_i][j] <= ub)

    if n_states > 0 and force_final_sram_empty:
        end_i = n_states - 1
        for j in range(n_blocks):
            solver.Add(ms[end_i][j] == int(round(sct.get(end_i, j))))

    # Eq.11 / Eq.12 memory capacity.
    for i in range(n_states):
        solver.Add(
            solver.Sum((float(sct.get(i, j)) - ms[i][j]) * float(block_volumes[j]) for j in range(n_blocks))
            <= float(sram_capacity)
        )
        solver.Add(
            solver.Sum((float(sct.get(i, j)) - md[i][j]) * float(block_volumes[j]) for j in range(n_blocks))
            <= float(dram_capacity)
        )

    # Each dependency supplies the *full* consumer interval. Data are identified
    # by sample indices, not by an interchangeable count of live activations.
    deps = parents_by_child(n_blocks, block_dependencies)
    direct_mb = old_mb = source_mb = 0.0
    dram_terms = []
    h_c = h_s = 1.0
    h_d = float(dram_noc_hops)
    dram_bw = float(dram_bandwidth) if dram_bandwidth is not None else float(noc_bandwidth)
    if min(noc_bandwidth, dram_bw) <= 0 or h_d < 1:
        raise ValueError("bandwidths must be positive and DRAM hop count at least one")
    for i in range(n_states):
        for child in range(n_blocks):
            count = sct.delta(i, child)
            if count <= 0:
                continue
            if block_weight_volumes is not None:
                source_mb += count * block_weight_volumes[child]
            if block_input_volumes is not None:
                source_mb += count * block_input_volumes[child]
            if not deps[child]:
                if block_input_volumes is None:
                    source_mb += count * block_volumes[child]
                continue
            for parent in deps[child]:
                lo, hi, direct = edge_intervals(sct, i, parent, child)
                volume = float(block_volumes[parent] if edge_volumes is None else edge_volumes[parent, child])
                direct_mb += direct * volume
                old_mb += (hi - lo) * volume
                if hi <= lo:
                    continue
                if i == 0:
                    raise ValueError("historical activations need an explicit previous memory state")
                discarded = clamped_integer(solver, ms[i - 1][parent], int(lo), int(hi),
                                            int(sct.get(i - 1, parent)), f"dram_read_{i}_{parent}_{child}")
                dram_terms.append(discarded * volume)
    dram_volume = source_mb + solver.Sum(dram_terms)
    # With uniform per-MB costs and H_D >= H_S=1, both Ltraffic and Etraffic
    # increase monotonically with DRAM reads. Minimizing reads exactly minimizes
    # their product: a loose McCormick envelope and unit-sensitive tie-breaker
    # are unnecessary. Weighting positive latency/energy preserves this ordering.
    total_reads = max(1e-30, source_mb + old_mb + direct_mb)
    solver.Minimize(dram_volume / total_reads)

    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        raise RuntimeError("MeT MILP failed to find feasible solution")

    report = solver_report(solver, status, "monotone_traffic_edp", total_reads)

    met = MemoryTable.zeros(n_states, n_blocks)
    for i in range(n_states):
        for j in range(n_blocks):
            met.set_sram(i, j, float(int(round(ms[i][j].solution_value()))))
            met.set_dram(i, j, float(int(round(md[i][j].solution_value()))))

    dep_c, dep_s, dep_d = _estimate_dep_traffic_from_tables(
        sct=sct,
        met=met,
        block_volumes=block_volumes,
        block_dependencies=block_dependencies,
        block_input_volumes=block_input_volumes,
        block_weight_volumes=block_weight_volumes,
        edge_volumes=edge_volumes,
    )
    noc_traffic = dep_c * h_c + dep_s * h_s + dep_d * h_d
    traffic_latency = noc_traffic / max(1e-9, float(noc_bandwidth)) + dep_d / max(1e-9, dram_bw)
    traffic_energy = (
        noc_traffic * max(0.0, float(noc_energy_per_unit))
        + dep_d * max(0.0, float(dram_energy_per_unit))
    )

    if use_edp_objective:
        obj = float((weight_latency * traffic_latency) * (weight_energy * traffic_energy))
    else:
        obj = float(weight_latency * traffic_latency + weight_energy * traffic_energy)

    return MemoryOptimizationResult(table=met, objective=obj, solver_name="ortools-scip", report=report)


def optimize_memory_table(
    sct: SchedulingTable,
    block_volumes: Sequence[float],
    block_dependencies: Sequence[tuple[int, int]],
    sram_capacity: float,
    dram_capacity: float,
    heuristic_sram_keep_ratio: float = 0.6,
    noc_bandwidth: float = 4096.0,
    dram_bandwidth: float | None = None,
    noc_energy_per_unit: float = 0.0,
    dram_energy_per_unit: float = 0.0075,
    dram_noc_hops: float = 1.0,
    weight_latency: float = 1.0,
    weight_energy: float = 1.0,
    use_edp_objective: bool = True,
    allow_fallback: bool = True,
    enforce_end_dram_dependency: bool = False,
    end_dram_upper_bounds: Sequence[int] | None = None,
    force_final_sram_empty: bool = False,
    solver_time_limit_s: float = 30.0,
    block_input_volumes: Sequence[float] | None = None,
    block_weight_volumes: Sequence[float] | None = None,
    edge_volumes: dict[tuple[int, int], float] | None = None,
) -> MemoryOptimizationResult:
    if pywraplp is None:
        if not allow_fallback:
            raise RuntimeError("OR-Tools is unavailable and fallback is disabled")
        met = build_memory_table(sct, sram_keep_ratio=heuristic_sram_keep_ratio)
        return MemoryOptimizationResult(table=met, objective=0.0, solver_name="heuristic-fallback")

    try:
        return _optimize_memory_with_ortools(
            sct=sct,
            block_volumes=block_volumes,
            block_dependencies=block_dependencies,
            sram_capacity=sram_capacity,
            dram_capacity=dram_capacity,
            noc_bandwidth=noc_bandwidth,
            dram_bandwidth=dram_bandwidth,
            noc_energy_per_unit=noc_energy_per_unit,
            dram_energy_per_unit=dram_energy_per_unit,
            dram_noc_hops=dram_noc_hops,
            weight_latency=weight_latency,
            weight_energy=weight_energy,
            use_edp_objective=use_edp_objective,
            enforce_end_dram_dependency=enforce_end_dram_dependency,
            end_dram_upper_bounds=end_dram_upper_bounds,
            force_final_sram_empty=force_final_sram_empty,
            solver_time_limit_s=solver_time_limit_s,
            block_input_volumes=block_input_volumes,
            block_weight_volumes=block_weight_volumes,
            edge_volumes=edge_volumes,
        )
    except Exception:
        if not allow_fallback:
            raise
        met = build_memory_table(sct, sram_keep_ratio=heuristic_sram_keep_ratio)
        return MemoryOptimizationResult(table=met, objective=0.0, solver_name="heuristic-fallback")







