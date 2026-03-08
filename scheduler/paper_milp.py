from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from scheduler.scheduling_table import SchedulingTable, build_weighted_sct

try:
    from ortools.linear_solver import pywraplp
except Exception:  # pragma: no cover
    pywraplp = None


@dataclass
class ScTOptimizationResult:
    sct: SchedulingTable
    state_workloads: list[int]
    state_latency_coeff: list[float]
    state_energy_coeff: list[float]
    objective: float
    solver_name: str


def _processing_window(block_idx: int, num_blocks: int) -> range:
    """Canonical pipeline-derived active-state range for one block."""

    start = block_idx
    end = num_blocks + block_idx - 1
    if end < start:
        return range(0)
    return range(start, end + 1)


def _active_blocks(state: int, num_blocks: int) -> list[int]:
    return [j for j in range(num_blocks) if state in _processing_window(j, num_blocks)]


def _state_cost_coeffs(
    block_flops: Sequence[float],
    num_states: int,
    num_pes: int,
    compute_power_per_tile: float,
    energy_per_op: float,
) -> tuple[list[float], list[float]]:
    """Compute per-state latency/energy unit coefficients following Eq.16-20 spirit.

    For each state i:
    - latency coefficient uses bottleneck sub-block latency in state-i (Eq.18 max_j)
    - energy coefficient sums normalized FLOPs across active sub-blocks (Eq.20)
    """

    lat: list[float] = []
    ene: list[float] = []

    for i in range(num_states):
        active = _active_blocks(i, len(block_flops))
        if not active:
            lat.append(0.0)
            ene.append(0.0)
            continue

        total_state_flops = sum(max(1e-9, float(block_flops[j])) for j in active)
        per_block_latency: list[float] = []
        per_block_energy: list[float] = []

        for j in active:
            # Figure-4 style proportional tile assignment (continuous approximation).
            tile_share = float(num_pes) * float(block_flops[j]) / total_state_flops
            tile_share = max(1e-6, tile_share)

            # Utilization proxy in [0.70, 0.95].
            util = 0.70 + 0.25 * min(1.0, tile_share / max(1.0, float(num_pes)))
            util = max(1e-6, util)

            l_ij = float(block_flops[j]) / (util * tile_share * max(1e-9, compute_power_per_tile))
            e_ij = energy_per_op * float(block_flops[j]) / util
            per_block_latency.append(l_ij)
            per_block_energy.append(e_ij)

        lat.append(max(per_block_latency))
        ene.append(sum(per_block_energy))

    return lat, ene


def _add_mccormick_product_objective(
    solver: "pywraplp.Solver",
    x_expr,
    y_expr,
    x_ub: float,
    y_ub: float,
):
    x = solver.NumVar(0.0, float(max(0.0, x_ub)), "obj_x")
    y = solver.NumVar(0.0, float(max(0.0, y_ub)), "obj_y")
    z = solver.NumVar(0.0, solver.infinity(), "obj_xy")

    solver.Add(x == x_expr)
    solver.Add(y == y_expr)

    # McCormick envelope for x,y in [0, U].
    solver.Add(z >= x_ub * y + y_ub * x - x_ub * y_ub)
    solver.Add(z <= x_ub * y)
    solver.Add(z <= y_ub * x)
    solver.Add(z >= 0.0)

    solver.Minimize(z)
    return x, y, z


def _solve_with_ortools(
    block_flops: Sequence[float],
    block_outputs: Sequence[float],
    total_sub_batches: int,
    block_dependencies: Iterable[tuple[int, int]],
    num_pes: int,
    weight_latency: float,
    weight_energy: float,
    num_states: int | None,
    min_active_states: int,
    min_batch_if_active: int,
    max_batches_per_state: Sequence[int] | None,
    use_edp_objective: bool,
    dependency_gap: int,
    compute_power_per_tile: float,
    energy_per_op: float,
    final_counts_per_block: Sequence[int] | None,
    initial_counts_per_block: Sequence[int] | None,
) -> ScTOptimizationResult:
    del block_outputs  # not used in ScT compute-side MILP.

    n_blocks = len(block_flops)
    if n_blocks == 0:
        raise ValueError("no blocks provided")

    if num_states is None:
        num_states = 2 * n_blocks - 1
    if num_states <= 0:
        raise ValueError("num_states must be > 0")

    if max_batches_per_state is not None and len(max_batches_per_state) != num_states:
        raise ValueError("max_batches_per_state length must equal num_states")

    if final_counts_per_block is not None and len(final_counts_per_block) != n_blocks:
        raise ValueError("final_counts_per_block length must equal number of blocks")
    if initial_counts_per_block is not None and len(initial_counts_per_block) != n_blocks:
        raise ValueError("initial_counts_per_block length must equal number of blocks")

    default_total = int(total_sub_batches)
    per_block_final = [
        int(final_counts_per_block[j]) if final_counts_per_block is not None else default_total
        for j in range(n_blocks)
    ]
    per_block_init = [
        int(initial_counts_per_block[j]) if initial_counts_per_block is not None else 0
        for j in range(n_blocks)
    ]
    if any(v < 0 for v in per_block_final) or any(v < 0 for v in per_block_init):
        raise ValueError("final/init counts must be >= 0")
    if any(per_block_init[j] > per_block_final[j] for j in range(n_blocks)):
        raise ValueError("initial count cannot exceed final count")

    max_final = max(per_block_final) if per_block_final else default_total

    solver = pywraplp.Solver.CreateSolver("SCIP")
    if solver is None:
        raise RuntimeError("OR-Tools SCIP solver is not available")

    sct = [[solver.IntVar(0, int(max_final), f"sct_{i}_{j}") for j in range(n_blocks)] for i in range(num_states)]
    w = [solver.IntVar(0, int(max_final), f"w_{i}") for i in range(num_states)]

    y = [solver.IntVar(0, 1, f"y_{i}") for i in range(num_states)]
    for i in range(num_states):
        cap = int(max_final) if max_batches_per_state is None else int(max_batches_per_state[i])
        cap = max(0, cap)
        solver.Add(w[i] <= cap)
        solver.Add(w[i] <= int(max_final) * y[i])
        if min_batch_if_active > 0:
            solver.Add(w[i] >= int(min_batch_if_active) * y[i])

    solver.Add(solver.Sum(y) >= int(max(0, min_active_states)))

    # Eq.1/2/3/4/6 for ScT.
    for j in range(n_blocks):
        win = list(_processing_window(j, n_blocks))
        init_j = int(per_block_init[j])
        final_j = int(per_block_final[j])
        need_j = final_j - init_j

        # Eq.2 before block-j activation.
        for i in range(0, j):
            solver.Add(sct[i][j] == init_j)

        # Eq.6 cumulative accumulation within involved states.
        for i in win:
            if i >= num_states:
                continue
            if i == 0:
                solver.Add(sct[i][j] == init_j + w[i])
            elif i == j:
                # i-1 is outside active window for block-j, so previous cumulative is init_j.
                solver.Add(sct[i][j] == init_j + w[i])
            else:
                solver.Add(sct[i][j] == sct[i - 1][j] + w[i])

        # Eq.3 after block-j completed.
        finish = n_blocks + j - 1
        for i in range(finish, num_states):
            solver.Add(sct[i][j] == final_j)

        # Constraint-1 completeness for each sub-block involved state set A_Bj.
        win_vars = [w[i] for i in win if 0 <= i < num_states]
        if not win_vars:
            raise RuntimeError("invalid processing window")
        solver.Add(solver.Sum(win_vars) == int(need_j))

        # Eq.4 monotonicity.
        for i in range(num_states - 1):
            solver.Add(sct[i + 1][j] >= sct[i][j])

    # Eq.5 dependency safety.
    dep_gap = max(0, int(dependency_gap))
    for p, c in block_dependencies:
        if p < 0 or c < 0 or p >= n_blocks or c >= n_blocks or p == c:
            continue

        # OCR of Eq.5 maps to parent progress ahead of dependent child.
        child_start = c
        child_end = min(num_states - 1, n_blocks + c - 2)
        for i in range(child_start, child_end + 1):
            solver.Add(sct[i][p] >= sct[i][c] + dep_gap)

    # inactive states (in non-canonical custom num_states) have zero workload.
    for i in range(num_states):
        if len(_active_blocks(i, n_blocks)) == 0:
            solver.Add(w[i] == 0)
            solver.Add(y[i] == 0)

    # Eq.22: minimize compute-related EDP with McCormick linearization.
    lat_coeff, ene_coeff = _state_cost_coeffs(
        block_flops=block_flops,
        num_states=num_states,
        num_pes=num_pes,
        compute_power_per_tile=compute_power_per_tile,
        energy_per_op=energy_per_op,
    )

    latency_expr = solver.Sum(w[i] * float(lat_coeff[i]) for i in range(num_states))
    energy_expr = solver.Sum(w[i] * float(ene_coeff[i]) for i in range(num_states))

    if use_edp_objective:
        wlat = max(1e-9, float(weight_latency))
        wene = max(1e-9, float(weight_energy))
        scaled_latency_expr = wlat * latency_expr
        scaled_energy_expr = wene * energy_expr

        lat_ub = float(max_final) * sum(max(0.0, c) for c in lat_coeff)
        ene_ub = float(max_final) * sum(max(0.0, c) for c in ene_coeff)
        _add_mccormick_product_objective(
            solver=solver,
            x_expr=scaled_latency_expr,
            y_expr=scaled_energy_expr,
            x_ub=max(1e-6, wlat * lat_ub),
            y_ub=max(1e-6, wene * ene_ub),
        )
    else:
        solver.Minimize(weight_latency * latency_expr + weight_energy * energy_expr)

    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        raise RuntimeError("ScT MILP failed to find feasible schedule")

    table = np.zeros((num_states, n_blocks), dtype=float)
    workloads: list[int] = []
    for i in range(num_states):
        workloads.append(int(round(w[i].solution_value())))
        for j in range(n_blocks):
            table[i, j] = int(round(sct[i][j].solution_value()))

    latency = sum(workloads[i] * lat_coeff[i] for i in range(num_states))
    energy = sum(workloads[i] * ene_coeff[i] for i in range(num_states))
    objective = float(latency * energy) if use_edp_objective else float(weight_latency * latency + weight_energy * energy)

    return ScTOptimizationResult(
        sct=SchedulingTable(table=table),
        state_workloads=workloads,
        state_latency_coeff=lat_coeff,
        state_energy_coeff=ene_coeff,
        objective=objective,
        solver_name="ortools-scip",
    )


def _solve_fallback(
    block_flops: Sequence[float],
    block_outputs: Sequence[float],
    total_sub_batches: int,
    num_pes: int,
    weight_latency: float,
    weight_energy: float,
    num_states: int | None,
    use_edp_objective: bool,
    compute_power_per_tile: float,
    energy_per_op: float,
) -> ScTOptimizationResult:
    del block_outputs

    if num_states is None:
        num_states = 2 * len(block_flops) - 1

    sct = build_weighted_sct(num_states, list(block_flops), total_sub_batches)
    lat_coeff, ene_coeff = _state_cost_coeffs(
        block_flops=block_flops,
        num_states=num_states,
        num_pes=num_pes,
        compute_power_per_tile=compute_power_per_tile,
        energy_per_op=energy_per_op,
    )
    delta = np.zeros_like(sct.table)
    delta[0, :] = sct.table[0, :]
    delta[1:, :] = sct.table[1:, :] - sct.table[:-1, :]
    state_workloads = [int(max(0.0, float(np.max(delta[i, :])))) for i in range(num_states)]

    latency = sum(state_workloads[i] * lat_coeff[i] for i in range(num_states))
    energy = sum(state_workloads[i] * ene_coeff[i] for i in range(num_states))
    objective = float(latency * energy) if use_edp_objective else float(weight_latency * latency + weight_energy * energy)

    return ScTOptimizationResult(
        sct=sct,
        state_workloads=state_workloads,
        state_latency_coeff=lat_coeff,
        state_energy_coeff=ene_coeff,
        objective=objective,
        solver_name="weighted-fallback",
    )


def optimize_sct_table(
    block_flops: Sequence[float],
    block_outputs: Sequence[float],
    total_sub_batches: int,
    block_dependencies: Iterable[tuple[int, int]],
    num_pes: int,
    weight_latency: float,
    weight_energy: float,
    num_states: int | None = None,
    min_active_states: int = 1,
    min_batch_if_active: int = 1,
    max_batches_per_state: Sequence[int] | None = None,
    use_edp_objective: bool = True,
    dependency_gap: int = 0,
    compute_power_per_tile: float = 1.0,
    energy_per_op: float = 1e-12,
    allow_fallback: bool = True,
    final_counts_per_block: Sequence[int] | None = None,
    initial_counts_per_block: Sequence[int] | None = None,
) -> ScTOptimizationResult:
    if pywraplp is None:
        if not allow_fallback:
            raise RuntimeError("OR-Tools is unavailable and fallback is disabled")
        # Fallback path only supports canonical equal-final-count ScT.
        if final_counts_per_block is not None or initial_counts_per_block is not None:
            raise RuntimeError("fallback does not support custom block final/initial counts")
        return _solve_fallback(
            block_flops,
            block_outputs,
            total_sub_batches,
            num_pes,
            weight_latency,
            weight_energy,
            num_states,
            use_edp_objective,
            compute_power_per_tile,
            energy_per_op,
        )

    try:
        return _solve_with_ortools(
            block_flops,
            block_outputs,
            total_sub_batches,
            block_dependencies,
            num_pes,
            weight_latency,
            weight_energy,
            num_states,
            min_active_states,
            min_batch_if_active,
            max_batches_per_state,
            use_edp_objective,
            dependency_gap,
            compute_power_per_tile,
            energy_per_op,
            final_counts_per_block,
            initial_counts_per_block,
        )
    except Exception:
        if not allow_fallback:
            raise
        if final_counts_per_block is not None or initial_counts_per_block is not None:
            raise
        return _solve_fallback(
            block_flops,
            block_outputs,
            total_sub_batches,
            num_pes,
            weight_latency,
            weight_energy,
            num_states,
            use_edp_objective,
            compute_power_per_tile,
            energy_per_op,
        )

