from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from scheduler.scheduling_table import SchedulingTable

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


def build_memory_table(sct: SchedulingTable, sram_keep_ratio: float = 0.6) -> MemoryTable:
    """Fallback heuristic for MeT when MILP is unavailable."""

    met = MemoryTable.zeros(sct.num_states, sct.num_blocks)
    for i in range(sct.num_states):
        for j in range(sct.num_blocks):
            cumulative = sct.get(i, j)
            met.set_sram(i, j, cumulative * (1.0 - sram_keep_ratio))
            met.set_dram(i, j, cumulative * 0.75)
    return met


def _add_mccormick_product_objective(
    solver: "pywraplp.Solver",
    x_expr,
    y_expr,
    x_ub: float,
    y_ub: float,
):
    x = solver.NumVar(0.0, float(max(0.0, x_ub)), "mem_x")
    y = solver.NumVar(0.0, float(max(0.0, y_ub)), "mem_y")
    z = solver.NumVar(0.0, solver.infinity(), "mem_xy")

    solver.Add(x == x_expr)
    solver.Add(y == y_expr)

    solver.Add(z >= x_ub * y + y_ub * x - x_ub * y_ub)
    solver.Add(z <= x_ub * y)
    solver.Add(z <= y_ub * x)
    solver.Add(z >= 0.0)

    return x, y, z


def _deps_by_child(
    n_blocks: int,
    block_dependencies: Sequence[tuple[int, int]],
) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for p, c in block_dependencies:
        if 0 <= p < n_blocks and 0 <= c < n_blocks and p != c:
            out.setdefault(c, []).append(p)
    return out


def _estimate_dep_traffic_from_tables(
    sct: SchedulingTable,
    met: MemoryTable,
    block_volumes: Sequence[float],
    block_dependencies: Sequence[tuple[int, int]],
) -> tuple[float, float, float]:
    """Estimate (Dep_C, Dep_S, Dep_D) volume from concrete ScT/MeT."""

    n_states = sct.num_states
    n_blocks = sct.num_blocks
    deps = _deps_by_child(n_blocks=n_blocks, block_dependencies=block_dependencies)

    delta = [[0.0 for _ in range(n_blocks)] for _ in range(n_states)]
    for i in range(n_states):
        for j in range(n_blocks):
            cur = float(sct.get(i, j))
            prev = float(sct.get(i - 1, j)) if i > 0 else 0.0
            delta[i][j] = max(0.0, cur - prev)

    dep_c = 0.0
    dep_s = 0.0
    dep_d = 0.0

    for i in range(n_states):
        prev_i = max(0, i - 1)
        for child in range(n_blocks):
            need_sb = float(delta[i][child])
            if need_sb <= 0:
                continue

            parents = deps.get(child, [])
            if not parents:
                dep_d += need_sb * float(block_volumes[child])
                continue

            per_parent_need = need_sb / float(max(1, len(parents)))
            for parent in parents:
                vol = float(block_volumes[parent])

                parent_new = float(delta[i][parent])
                from_c = min(per_parent_need, parent_new)
                remain = per_parent_need - from_c

                sram_live = max(0.0, float(sct.get(prev_i, parent)) - float(met.sram[prev_i, parent]))
                from_s = min(remain, sram_live)
                remain -= from_s

                dram_live = max(0.0, float(sct.get(prev_i, parent)) - float(met.dram[prev_i, parent]))
                from_d = min(remain, dram_live)
                remain -= from_d
                if remain > 0.0:
                    from_d += remain

                dep_c += from_c * vol
                dep_s += from_s * vol
                dep_d += from_d * vol

    return dep_c, dep_s, dep_d


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
) -> MemoryOptimizationResult:
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if solver is None:
        raise RuntimeError("OR-Tools SCIP solver is not available")

    n_states = sct.num_states
    n_blocks = sct.num_blocks

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

    # Eq.23 objective with explicit Eq.19/Eq.21 decomposition.
    deps = _deps_by_child(n_blocks=n_blocks, block_dependencies=block_dependencies)

    delta = [[0.0 for _ in range(n_blocks)] for _ in range(n_states)]
    for i in range(n_states):
        for j in range(n_blocks):
            cur = float(sct.get(i, j))
            prev = float(sct.get(i - 1, j)) if i > 0 else 0.0
            delta[i][j] = max(0.0, cur - prev)

    h_c = 1.0
    h_s = 1.0
    h_d = max(0.0, float(dram_noc_hops))

    dep_c_const = 0.0
    dep_d_const = 0.0
    noc_terms = []
    dep_d_terms = []

    for i in range(n_states):
        prev_i = max(0, i - 1)
        for child in range(n_blocks):
            need_sb = float(delta[i][child])
            if need_sb <= 0.0:
                continue

            parents = deps.get(child, [])
            if not parents:
                dep_d_const += need_sb * float(block_volumes[child])
                continue

            per_parent_need = need_sb / float(max(1, len(parents)))
            for parent in parents:
                vol = float(block_volumes[parent])
                parent_new = float(delta[i][parent])
                from_c_sb = min(per_parent_need, parent_new)
                dep_c_const += from_c_sb * vol

                remain_sb = max(0.0, per_parent_need - from_c_sb)
                if remain_sb <= 1e-12:
                    continue

                dep_s = solver.NumVar(0.0, remain_sb, f"dep_s_{i}_{parent}_{child}")
                dep_d = solver.NumVar(0.0, remain_sb, f"dep_d_{i}_{parent}_{child}")

                sram_live_expr = float(sct.get(prev_i, parent)) - ms[prev_i][parent]
                dram_live_expr = float(sct.get(prev_i, parent)) - md[prev_i][parent]
                solver.Add(dep_s <= sram_live_expr)
                solver.Add(dep_d <= dram_live_expr)
                solver.Add(dep_s + dep_d == remain_sb)

                noc_terms.append(dep_s * vol * h_s)
                noc_terms.append(dep_d * vol * h_d)
                dep_d_terms.append(dep_d * vol)

    noc_traffic_expr = dep_c_const * h_c + dep_d_const * h_d
    if noc_terms:
        noc_traffic_expr += solver.Sum(noc_terms)

    dep_d_volume_expr = dep_d_const
    if dep_d_terms:
        dep_d_volume_expr += solver.Sum(dep_d_terms)

    dram_bw = float(dram_bandwidth) if dram_bandwidth is not None else float(noc_bandwidth)
    traffic_latency_expr = (
        noc_traffic_expr / max(1e-9, float(noc_bandwidth))
        + dep_d_volume_expr / max(1e-9, dram_bw)
    )
    traffic_energy_expr = (
        noc_traffic_expr * max(0.0, float(noc_energy_per_unit))
        + dep_d_volume_expr * max(0.0, float(dram_energy_per_unit))
    )

    if use_edp_objective:
        wlat = max(1e-9, float(weight_latency))
        wene = max(1e-9, float(weight_energy))
        scaled_latency_expr = wlat * traffic_latency_expr
        scaled_energy_expr = wene * traffic_energy_expr

        vol_ub = sum(float(sct.get(i, j)) * float(block_volumes[j]) for i in range(n_states) for j in range(n_blocks))
        hop_ub = max(h_c, h_s, h_d)
        noc_ub = max(0.0, vol_ub * hop_ub)
        lat_ub = noc_ub / max(1e-9, float(noc_bandwidth)) + vol_ub / max(1e-9, dram_bw)
        ene_ub = noc_ub * max(0.0, float(noc_energy_per_unit)) + vol_ub * max(0.0, float(dram_energy_per_unit))
        _, _, z = _add_mccormick_product_objective(
            solver=solver,
            x_expr=scaled_latency_expr,
            y_expr=scaled_energy_expr,
            x_ub=max(1e-6, wlat * lat_ub),
            y_ub=max(1e-6, wene * ene_ub),
        )

        # Tie-breaker: mildly prefer smaller SRAM live volume.
        sram_live_volume = solver.Sum(
            (float(sct.get(i, j)) - ms[i][j]) * float(block_volumes[j])
            for i in range(n_states)
            for j in range(n_blocks)
        )
        solver.Minimize(z + 1e-6 * sram_live_volume)
    else:
        solver.Minimize(weight_latency * traffic_latency_expr + weight_energy * traffic_energy_expr)

    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        raise RuntimeError("MeT MILP failed to find feasible solution")

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

    return MemoryOptimizationResult(table=met, objective=obj, solver_name="ortools-scip")


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
        )
    except Exception:
        if not allow_fallback:
            raise
        met = build_memory_table(sct, sram_keep_ratio=heuristic_sram_keep_ratio)
        return MemoryOptimizationResult(table=met, objective=0.0, solver_name="heuristic-fallback")
