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


def _optimize_memory_with_ortools(
    sct: SchedulingTable,
    block_volumes: Sequence[float],
    block_dependencies: Sequence[tuple[int, int]],
    sram_capacity: float,
    dram_capacity: float,
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

    # Eq.9 bounds already covered by variable upper bounds; keep explicit for clarity.
    for i in range(n_states):
        for j in range(n_blocks):
            solver.Add(ms[i][j] <= int(round(sct.get(i, j))))
            solver.Add(md[i][j] <= int(round(sct.get(i, j))))

    # Eq.10 dependency-safe memory availability (conservative linear form).
    for parent, child in block_dependencies:
        if parent < 0 or child < 0 or parent >= n_blocks or child >= n_blocks:
            continue
        for i in range(1, n_states):
            child_done_prev = int(round(sct.get(i - 1, child)))
            solver.Add(ms[i - 1][parent] <= child_done_prev)
            solver.Add(md[i - 1][parent] <= child_done_prev)

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

    # Minimize DRAM footprint first, then SRAM footprint.
    dram_term = solver.Sum((float(sct.get(i, j)) - md[i][j]) * float(block_volumes[j]) for i in range(n_states) for j in range(n_blocks))
    sram_term = solver.Sum((float(sct.get(i, j)) - ms[i][j]) * float(block_volumes[j]) for i in range(n_states) for j in range(n_blocks))
    solver.Minimize(dram_term + 0.05 * sram_term)

    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        raise RuntimeError("MeT MILP failed to find feasible solution")

    met = MemoryTable.zeros(n_states, n_blocks)
    for i in range(n_states):
        for j in range(n_blocks):
            met.set_sram(i, j, float(int(round(ms[i][j].solution_value()))))
            met.set_dram(i, j, float(int(round(md[i][j].solution_value()))))

    obj = sum((sct.get(i, j) - met.dram[i, j]) * float(block_volumes[j]) for i in range(n_states) for j in range(n_blocks))
    obj += 0.05 * sum((sct.get(i, j) - met.sram[i, j]) * float(block_volumes[j]) for i in range(n_states) for j in range(n_blocks))

    return MemoryOptimizationResult(table=met, objective=float(obj), solver_name="ortools-scip")


def optimize_memory_table(
    sct: SchedulingTable,
    block_volumes: Sequence[float],
    block_dependencies: Sequence[tuple[int, int]],
    sram_capacity: float,
    dram_capacity: float,
    heuristic_sram_keep_ratio: float = 0.6,
) -> MemoryOptimizationResult:
    if pywraplp is None:
        met = build_memory_table(sct, sram_keep_ratio=heuristic_sram_keep_ratio)
        return MemoryOptimizationResult(table=met, objective=0.0, solver_name="heuristic-fallback")

    try:
        return _optimize_memory_with_ortools(
            sct=sct,
            block_volumes=block_volumes,
            block_dependencies=block_dependencies,
            sram_capacity=sram_capacity,
            dram_capacity=dram_capacity,
        )
    except Exception:
        met = build_memory_table(sct, sram_keep_ratio=heuristic_sram_keep_ratio)
        return MemoryOptimizationResult(table=met, objective=0.0, solver_name="heuristic-fallback")

