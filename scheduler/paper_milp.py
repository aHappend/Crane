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
    act = []
    for j in range(num_blocks):
        if state in _processing_window(j, num_blocks):
            act.append(j)
    return act


def _state_cost_coeffs(
    block_flops: Sequence[float],
    block_outputs: Sequence[float],
    num_states: int,
    num_pes: int,
) -> tuple[list[float], list[float]]:
    lat: list[float] = []
    ene: list[float] = []

    for s in range(num_states):
        active = _active_blocks(s, len(block_flops))
        if not active:
            lat.append(0.0)
            ene.append(0.0)
            continue

        active_pe = max(1, min(num_pes, len(active)))
        util = 0.82 + 0.04 * (active_pe / max(1, num_pes))
        per_state_flops = sum(block_flops[j] for j in active)
        per_state_outputs = sum(block_outputs[j] for j in active)

        lat.append(per_state_flops / (util * active_pe))
        ene.append(per_state_outputs * (0.55 + 0.1 * active_pe))

    return lat, ene


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
) -> ScTOptimizationResult:
    n_blocks = len(block_flops)
    if n_blocks == 0:
        raise ValueError("no blocks provided")

    if num_states is None:
        num_states = 2 * n_blocks - 1
    if num_states <= 0:
        raise ValueError("num_states must be > 0")

    if max_batches_per_state is not None and len(max_batches_per_state) != num_states:
        raise ValueError("max_batches_per_state length must equal num_states")

    solver = pywraplp.Solver.CreateSolver("SCIP")
    if solver is None:
        raise RuntimeError("OR-Tools SCIP solver is not available")

    sct = [[solver.IntVar(0, int(total_sub_batches), f"sct_{i}_{j}") for j in range(n_blocks)] for i in range(num_states)]
    w = [solver.IntVar(0, int(total_sub_batches), f"w_{i}") for i in range(num_states)]
    d = [[solver.IntVar(0, int(total_sub_batches), f"d_{i}_{j}") for j in range(n_blocks)] for i in range(num_states)]

    y = [solver.IntVar(0, 1, f"y_{i}") for i in range(num_states)]
    for i in range(num_states):
        cap = int(total_sub_batches) if max_batches_per_state is None else int(max_batches_per_state[i])
        cap = max(0, cap)
        solver.Add(w[i] <= cap)
        solver.Add(w[i] <= int(total_sub_batches) * y[i])
        if min_batch_if_active > 0:
            solver.Add(w[i] >= int(min_batch_if_active) * y[i])

    solver.Add(solver.Sum(y) >= int(max(1, min_active_states)))

    # Eq.2 style boundary and monotonicity.
    for j in range(n_blocks):
        for i in range(num_states):
            if i < j:
                solver.Add(sct[i][j] == 0)
            if i > 0:
                solver.Add(sct[i][j] >= sct[i - 1][j])

    # Eq.6 style cumulative relation with per-state workload variables.
    for j in range(n_blocks):
        win = set(_processing_window(j, n_blocks))
        for i in range(num_states):
            if i in win:
                solver.Add(d[i][j] == w[i])
            else:
                solver.Add(d[i][j] == 0)

            if i == 0:
                solver.Add(sct[i][j] == d[i][j])
            else:
                solver.Add(sct[i][j] == sct[i - 1][j] + d[i][j])

        solver.Add(sct[num_states - 1][j] == int(total_sub_batches))

    for i in range(num_states):
        if len(_active_blocks(i, n_blocks)) == 0:
            solver.Add(w[i] == 0)
            solver.Add(y[i] == 0)

    # Eq.5-like dependency safety: child cannot outrun parent.
    for p, c in block_dependencies:
        if p < 0 or c < 0 or p >= n_blocks or c >= n_blocks or p == c:
            continue
        for i in range(num_states):
            solver.Add(sct[i][c] <= sct[i][p])

    lat_coeff, ene_coeff = _state_cost_coeffs(block_flops, block_outputs, num_states, num_pes)

    latency_expr = solver.Sum(w[i] * float(lat_coeff[i]) for i in range(num_states))
    energy_expr = solver.Sum(w[i] * float(ene_coeff[i]) for i in range(num_states))
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
    objective = weight_latency * latency + weight_energy * energy

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
) -> ScTOptimizationResult:
    if num_states is None:
        num_states = 2 * len(block_flops) - 1

    sct = build_weighted_sct(num_states, list(block_flops), total_sub_batches)
    lat_coeff, ene_coeff = _state_cost_coeffs(block_flops, block_outputs, num_states, num_pes)
    delta = np.zeros_like(sct.table)
    delta[0, :] = sct.table[0, :]
    delta[1:, :] = sct.table[1:, :] - sct.table[:-1, :]
    state_workloads = [int(max(0.0, float(np.max(delta[i, :])))) for i in range(num_states)]

    latency = sum(state_workloads[i] * lat_coeff[i] for i in range(num_states))
    energy = sum(state_workloads[i] * ene_coeff[i] for i in range(num_states))
    objective = weight_latency * latency + weight_energy * energy

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
) -> ScTOptimizationResult:
    if pywraplp is None:
        return _solve_fallback(
            block_flops,
            block_outputs,
            total_sub_batches,
            num_pes,
            weight_latency,
            weight_energy,
            num_states,
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
        )
    except Exception:
        return _solve_fallback(
            block_flops,
            block_outputs,
            total_sub_batches,
            num_pes,
            weight_latency,
            weight_energy,
            num_states,
        )

