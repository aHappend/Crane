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



def _normalize_state_bound_matrix(
    name: str,
    matrix: Sequence[Sequence[int | None]] | None,
    num_states: int,
    num_blocks: int,
) -> list[list[int | None]] | None:
    if matrix is None:
        return None
    if len(matrix) != num_states:
        raise ValueError(f"{name} row count must equal number of states")

    out: list[list[int | None]] = []
    for i, row in enumerate(matrix):
        if len(row) != num_blocks:
            raise ValueError(f"{name}[{i}] column count must equal number of blocks")
        norm_row: list[int | None] = []
        for v in row:
            norm_row.append(None if v is None else int(v))
        out.append(norm_row)
    return out


def _normalize_state_block_float_matrix(
    name: str,
    matrix: Sequence[Sequence[float | None]] | None,
    num_states: int,
    num_blocks: int,
) -> list[list[float | None]] | None:
    if matrix is None:
        return None
    if len(matrix) != num_states:
        raise ValueError(f"{name} row count must equal number of states")

    out: list[list[float | None]] = []
    for i, row in enumerate(matrix):
        if len(row) != num_blocks:
            raise ValueError(f"{name}[{i}] column count must equal number of blocks")
        out.append([None if v is None else float(v) for v in row])
    return out


def _factorizations4(total_tiles: int) -> list[tuple[int, int, int, int]]:
    total = max(1, int(total_tiles))
    out: list[tuple[int, int, int, int]] = []
    for a in range(1, total + 1):
        if total % a != 0:
            continue
        rem1 = total // a
        for b in range(1, rem1 + 1):
            if rem1 % b != 0:
                continue
            rem2 = rem1 // b
            for c in range(1, rem2 + 1):
                if rem2 % c != 0:
                    continue
                d = rem2 // c
                out.append((a, b, c, d))
    return out


def _infer_default_map_dims(flops: float) -> tuple[float, float, float, float]:
    val = max(1.0, float(flops))
    d1 = max(1.0, val ** 0.25)
    d2 = max(1.0, val ** 0.25)
    d3 = max(1.0, val ** 0.25)
    d4 = max(1.0, val / max(1.0, d1 * d2 * d3))
    return d1, d2, d3, d4


def _best_utilization_for_tiles(
    dims: Sequence[float],
    tiles: int,
) -> float:
    shape = [max(1.0, float(v)) for v in dims]
    best = 1e-6
    for k1, k2, k3, k4 in _factorizations4(max(1, int(tiles))):
        factors = [float(k1), float(k2), float(k3), float(k4)]
        util = 1.0
        for dim_q, fac_q in zip(shape, factors):
            packs = max(1.0, float(np.ceil(dim_q / fac_q)))
            util *= dim_q / max(1.0, packs * fac_q)
        if util > best:
            best = util
    return max(1e-6, min(1.0, best))


def _integer_tile_allocation(
    active: Sequence[int],
    block_flops: Sequence[float],
    num_pes: int,
) -> tuple[list[int], float]:
    if not active:
        return [], 1.0

    weights = [max(1e-9, float(block_flops[j])) for j in active]
    total = sum(weights)
    alloc = [1 for _ in active]
    if num_pes > len(active):
        remaining = num_pes - len(active)
        raw = [remaining * w / max(1e-9, total) for w in weights]
        extra = [int(np.floor(v)) for v in raw]
        alloc = [a + e for a, e in zip(alloc, extra)]
        used = sum(extra)
        frac_order = sorted(
            range(len(active)),
            key=lambda idx: (raw[idx] - np.floor(raw[idx]), weights[idx]),
            reverse=True,
        )
        for idx in frac_order[: max(0, remaining - used)]:
            alloc[idx] += 1

    overcommit = max(1.0, float(sum(alloc)) / max(1.0, float(num_pes)))
    return alloc, overcommit


def _state_cost_coeffs(
    block_flops: Sequence[float],
    block_map_dims: Sequence[Sequence[float]] | None,
    block_unit_latency_override: Sequence[float | None] | None,
    block_unit_energy_override: Sequence[float | None] | None,
    state_block_latency_override: Sequence[Sequence[float | None]] | None,
    state_block_energy_override: Sequence[Sequence[float | None]] | None,
    num_states: int,
    num_pes: int,
    compute_power_per_tile: float,
    energy_per_op: float,
) -> tuple[list[float], list[float]]:
    """Compute per-state latency/energy coefficients with SET-style tile factorization."""

    lat: list[float] = []
    ene: list[float] = []

    for i in range(num_states):
        active = _active_blocks(i, len(block_flops))
        if not active:
            lat.append(0.0)
            ene.append(0.0)
            continue

        alloc, overcommit = _integer_tile_allocation(active, block_flops, num_pes)
        per_block_latency: list[float] = []
        per_block_energy: list[float] = []

        for local_idx, j in enumerate(active):
            if (
                state_block_latency_override is not None
                and state_block_energy_override is not None
                and i < len(state_block_latency_override)
                and i < len(state_block_energy_override)
                and j < len(state_block_latency_override[i])
                and j < len(state_block_energy_override[i])
                and state_block_latency_override[i][j] is not None
                and state_block_energy_override[i][j] is not None
            ):
                per_block_latency.append(max(0.0, float(state_block_latency_override[i][j])))
                per_block_energy.append(max(0.0, float(state_block_energy_override[i][j])))
                continue

            if (
                block_unit_latency_override is not None
                and block_unit_energy_override is not None
                and j < len(block_unit_latency_override)
                and j < len(block_unit_energy_override)
                and block_unit_latency_override[j] is not None
                and block_unit_energy_override[j] is not None
            ):
                per_block_latency.append(max(0.0, float(block_unit_latency_override[j])))
                per_block_energy.append(max(0.0, float(block_unit_energy_override[j])))
                continue

            tiles = max(1, int(alloc[local_idx]))
            dims = (
                [float(v) for v in block_map_dims[j]]
                if block_map_dims is not None
                else list(_infer_default_map_dims(block_flops[j]))
            )
            util = _best_utilization_for_tiles(dims=dims, tiles=tiles)
            l_ij = overcommit * float(block_flops[j]) / (util * float(tiles) * max(1e-9, compute_power_per_tile))
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
    block_map_dims: Sequence[Sequence[float]] | None,
    block_unit_latency_override: Sequence[float | None] | None,
    block_unit_energy_override: Sequence[float | None] | None,
    state_block_latency_override: Sequence[Sequence[float | None]] | None,
    state_block_energy_override: Sequence[Sequence[float | None]] | None,
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
    state_count_lower_bounds: Sequence[Sequence[int | None]] | None,
    state_count_upper_bounds: Sequence[Sequence[int | None]] | None,
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

    lower_bounds = _normalize_state_bound_matrix(
        "state_count_lower_bounds",
        state_count_lower_bounds,
        num_states,
        n_blocks,
    )
    upper_bounds = _normalize_state_bound_matrix(
        "state_count_upper_bounds",
        state_count_upper_bounds,
        num_states,
        n_blocks,
    )
    state_lat_overrides = _normalize_state_block_float_matrix(
        "state_block_latency_override",
        state_block_latency_override,
        num_states,
        n_blocks,
    )
    state_ene_overrides = _normalize_state_block_float_matrix(
        "state_block_energy_override",
        state_block_energy_override,
        num_states,
        n_blocks,
    )

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
    if hasattr(solver, "SetSolverSpecificParametersAsString"):
        solver.SetSolverSpecificParametersAsString(
            "display/verblevel = 0\nparallel/maxnthreads = 1\nseparating/gomory/freq = -1\n"
        )

    sct = [[solver.IntVar(0, int(max_final), f"sct_{i}_{j}") for j in range(n_blocks)] for i in range(num_states)]
    w = [solver.IntVar(0, int(max_final), f"w_{i}") for i in range(num_states)]

    for i in range(num_states):
        for j in range(n_blocks):
            lb = None if lower_bounds is None else lower_bounds[i][j]
            ub = None if upper_bounds is None else upper_bounds[i][j]
            if lb is not None and lb < 0:
                raise ValueError("state_count_lower_bounds must be >= 0")
            if ub is not None and ub < 0:
                raise ValueError("state_count_upper_bounds must be >= 0")
            if lb is not None and lb > int(max_final):
                raise ValueError("state_count_lower_bounds exceeds feasible maximum")
            if ub is not None and lb is not None and lb > ub:
                raise ValueError("state count lower bound cannot exceed upper bound")
            if lb is not None:
                solver.Add(sct[i][j] >= int(lb))
            if ub is not None:
                solver.Add(sct[i][j] <= int(ub))

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
        block_map_dims=block_map_dims,
        block_unit_latency_override=block_unit_latency_override,
        block_unit_energy_override=block_unit_energy_override,
        state_block_latency_override=state_block_latency_override,
        state_block_energy_override=state_block_energy_override,
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
    block_map_dims: Sequence[Sequence[float]] | None,
    block_unit_latency_override: Sequence[float | None] | None,
    block_unit_energy_override: Sequence[float | None] | None,
    state_block_latency_override: Sequence[Sequence[float | None]] | None,
    state_block_energy_override: Sequence[Sequence[float | None]] | None,
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
        block_map_dims=block_map_dims,
        block_unit_latency_override=block_unit_latency_override,
        block_unit_energy_override=block_unit_energy_override,
        state_block_latency_override=state_block_latency_override,
        state_block_energy_override=state_block_energy_override,
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
    block_map_dims: Sequence[Sequence[float]] | None = None,
    block_unit_latency_override: Sequence[float | None] | None = None,
    block_unit_energy_override: Sequence[float | None] | None = None,
    state_block_latency_override: Sequence[Sequence[float | None]] | None = None,
    state_block_energy_override: Sequence[Sequence[float | None]] | None = None,
    allow_fallback: bool = True,
    final_counts_per_block: Sequence[int] | None = None,
    initial_counts_per_block: Sequence[int] | None = None,
    state_count_lower_bounds: Sequence[Sequence[int | None]] | None = None,
    state_count_upper_bounds: Sequence[Sequence[int | None]] | None = None,
) -> ScTOptimizationResult:
    if pywraplp is None:
        if not allow_fallback:
            raise RuntimeError("OR-Tools is unavailable and fallback is disabled")
        if final_counts_per_block is not None or initial_counts_per_block is not None or state_count_lower_bounds is not None or state_count_upper_bounds is not None:
            raise RuntimeError("fallback does not support custom block final/initial counts")
        return _solve_fallback(
            block_flops,
            block_outputs,
            block_map_dims,
            block_unit_latency_override,
            block_unit_energy_override,
            state_block_latency_override,
            state_block_energy_override,
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
            block_map_dims,
            block_unit_latency_override,
            block_unit_energy_override,
            state_block_latency_override,
            state_block_energy_override,
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
            state_count_lower_bounds,
            state_count_upper_bounds,
        )
    except Exception:
        if not allow_fallback:
            raise
        if final_counts_per_block is not None or initial_counts_per_block is not None or state_count_lower_bounds is not None or state_count_upper_bounds is not None:
            raise
        return _solve_fallback(
            block_flops,
            block_outputs,
            block_map_dims,
            block_unit_latency_override,
            block_unit_energy_override,
            state_block_latency_override,
            state_block_energy_override,
            total_sub_batches,
            num_pes,
            weight_latency,
            weight_energy,
            num_states,
            use_edp_objective,
            compute_power_per_tile,
            energy_per_op,
        )











