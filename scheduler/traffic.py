"""Shared sub-batch interval accounting for the MILP and reported costs.

Each consumer needs the indicated sample indices from *every* dependency.
Available SRAM/DRAM ranges are open-left, closed-right cumulative intervals.
Counting equal-length but disjoint intervals as direct transfer is incorrect.
"""
from __future__ import annotations

from typing import Sequence

from scheduler.scheduling_table import SchedulingTable


def parents_by_child(count: int, dependencies: Sequence[tuple[int, int]]) -> dict[int, list[int]]:
    parents = {j: [] for j in range(count)}
    for p, c in sorted(set(dependencies)):
        if not 0 <= p < count or not 0 <= c < count or p == c:
            raise ValueError(f"invalid dependency {(p, c)} for {count} blocks")
        parents[c].append(p)
    return parents


def edge_intervals(sct: SchedulingTable, state: int, parent: int, child: int) -> tuple[float, float, float]:
    """Return required lower bound, old-output upper bound, direct count."""
    lower, upper = sct.previous(state, child), sct.get(state, child)
    if upper < lower - 1e-7:
        raise ValueError("ScT progress must be monotone")
    if upper > sct.get(state, parent) + 1e-7:
        raise ValueError("consumer requires sub-batches not produced by its parent")
    old_upper = min(upper, sct.previous(state, parent))
    direct = max(0.0, upper - max(lower, sct.previous(state, parent)))
    return lower, max(lower, old_upper), direct


def estimate_traffic(
    sct: SchedulingTable,
    met,
    block_volumes: Sequence[float],
    block_dependencies: Sequence[tuple[int, int]],
    block_input_volumes: Sequence[float] | None = None,
    block_weight_volumes: Sequence[float] | None = None,
    edge_volumes: dict[tuple[int, int], float] | None = None,
) -> tuple[float, float, float]:
    """Return direct, SRAM and DRAM MB; missing data is an error, not a fetch."""
    parents = parents_by_child(sct.num_blocks, block_dependencies)
    direct_mb = sram_mb = dram_mb = 0.0
    for i in range(sct.num_states):
        for child in range(sct.num_blocks):
            count = sct.delta(i, child)
            if count <= 0:
                continue
            if block_weight_volumes is not None:
                dram_mb += count * block_weight_volumes[child]
            if block_input_volumes is not None:
                dram_mb += count * block_input_volumes[child]
            if not parents[child]:
                if block_input_volumes is None:
                    dram_mb += count * block_volumes[child]
                continue
            for parent in parents[child]:
                lower, old_upper, direct = edge_intervals(sct, i, parent, child)
                ms = float(met.sram[i - 1, parent]) if i > 0 else 0.0
                md = float(met.dram[i - 1, parent]) if i > 0 else 0.0
                if old_upper > lower and min(ms, md) > lower + 1e-7:
                    raise ValueError("required activation was discarded from both SRAM and DRAM")
                from_s = max(0.0, old_upper - max(lower, ms))
                from_d = old_upper - lower - from_s
                volume = block_volumes[parent] if edge_volumes is None else edge_volumes[parent, child]
                direct_mb += direct * volume
                sram_mb += from_s * volume
                dram_mb += from_d * volume
    return direct_mb, sram_mb, dram_mb


def clamped_integer(solver, variable, lower: int, upper: int, bound: int, name: str):
    """Exactly encode min(max(variable-lower, 0), upper-lower)."""
    width = upper - lower
    if width <= 0:
        return 0.0
    out = solver.IntVar(0, width, name)
    below, middle, above = [solver.BoolVar(f"{name}_{x}") for x in ("below", "middle", "above")]
    solver.Add(below + middle + above == 1)
    big_m = max(1, bound + abs(lower) + abs(upper))
    solver.Add(variable <= lower + big_m * (1 - below))
    solver.Add(variable >= upper - big_m * (1 - above))
    solver.Add(variable >= lower - big_m * (1 - middle))
    solver.Add(variable <= upper + big_m * (1 - middle))
    solver.Add(out <= width * (1 - below))
    solver.Add(out >= width * above)
    solver.Add(out >= variable - lower - big_m * (1 - middle))
    solver.Add(out <= variable - lower + big_m * (1 - middle))
    return out
