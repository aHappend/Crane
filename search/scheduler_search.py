from dataclasses import dataclass
from math import floor
from typing import Sequence

from cost_model.energy_model import edp
from scheduler.block import Block
from scheduler.memory_table import MemoryTable, build_memory_table
from scheduler.milp_solver import MilpSolution, MilpSolver
from scheduler.scheduling_table import SchedulingTable, build_weighted_sct


@dataclass
class SearchConfig:
    batch_size: int
    candidate_sub_batches: Sequence[int]
    sram_capacity: float
    dram_capacity: float
    num_pes: int = 4
    num_states: int | None = None
    weight_latency: float = 1.0
    weight_energy: float = 1.0
    min_active_states: int = 1
    min_batch_if_active: int = 1
    max_state_share: float = 1.0


@dataclass
class SearchResult:
    best_sub_batch: int
    state_order: list[str]
    state_categories: list[str]
    active_pes: list[int]
    state_unit_latency: list[float]
    state_unit_energy: list[float]
    sct: SchedulingTable
    met: MemoryTable
    milp_solution: MilpSolution
    total_latency: float
    total_energy: float
    total_edp: float


def _memory_feasible(
    sct: SchedulingTable,
    met: MemoryTable,
    block_volumes: Sequence[float],
    sram_capacity: float,
    dram_capacity: float,
) -> bool:
    for i in range(sct.num_states):
        sram_use = 0.0
        dram_use = 0.0
        for j, v in enumerate(block_volumes):
            sram_use += (sct.get(i, j) - met.sram[i, j]) * v
            dram_use += (sct.get(i, j) - met.dram[i, j]) * v
        if sram_use > sram_capacity or dram_use > dram_capacity:
            return False
    return True


def _build_pipeline_state_metadata(num_pes: int) -> tuple[list[str], list[str], list[int]]:
    state_order: list[str] = []
    categories: list[str] = []
    active_pes: list[int] = []

    for i in range(num_pes - 1):
        state_order.append(f"W{i}")
        categories.append("warmup")
        active_pes.append(i + 1)

    state_order.append("M")
    categories.append("steady")
    active_pes.append(num_pes)

    for i in range(num_pes - 1):
        state_order.append(f"D{i}")
        categories.append("drain")
        active_pes.append(num_pes - 1 - i)

    return state_order, categories, active_pes


def _build_generic_state_metadata(num_states: int, num_pes: int) -> tuple[list[str], list[str], list[int]]:
    state_order = [f"S{i}" for i in range(num_states)]
    categories = ["generic" for _ in range(num_states)]
    active = [max(1, min(num_pes, i + 1)) for i in range(num_states)]
    return state_order, categories, active


def _build_state_metadata(config: SearchConfig) -> tuple[int, list[str], list[str], list[int]]:
    if config.num_pes <= 0:
        raise ValueError("num_pes must be > 0")

    inferred_states = 2 * config.num_pes - 1
    num_states = inferred_states if config.num_states is None else config.num_states

    if num_states == inferred_states:
        state_order, categories, active_pes = _build_pipeline_state_metadata(config.num_pes)
    else:
        state_order, categories, active_pes = _build_generic_state_metadata(num_states, config.num_pes)

    return num_states, state_order, categories, active_pes


def _build_state_costs(
    total_flops: float,
    total_outputs: float,
    num_pes: int,
    active_pes: Sequence[int],
) -> tuple[list[float], list[float]]:
    latency: list[float] = []
    energy: list[float] = []

    for active in active_pes:
        parallel = max(1.0, 0.9 * active)
        bubble_penalty = 1.0 + 0.35 * (num_pes - active) / num_pes
        latency.append(total_flops * bubble_penalty / parallel)

        power_factor = 0.55 + 0.12 * active
        stall_penalty = 1.0 + 0.20 * (num_pes - active) / num_pes
        energy.append(total_outputs * power_factor * stall_penalty)

    return latency, energy


def _state_caps(total_sub_batches: int, num_states: int, max_state_share: float) -> list[int] | None:
    if max_state_share <= 0 or max_state_share > 1.0:
        raise ValueError("max_state_share must be in (0, 1]")

    if max_state_share >= 1.0:
        return None

    cap = max(1, floor(total_sub_batches * max_state_share))
    return [cap for _ in range(num_states)]


def search_schedule(
    blocks: Sequence[Block],
    config: SearchConfig,
    solver: MilpSolver | None = None,
) -> SearchResult:
    if solver is None:
        solver = MilpSolver()

    num_states, state_order, categories, active_pes = _build_state_metadata(config)

    block_flops = [b.total_flops() for b in blocks]
    block_outputs = [b.total_output_size() for b in blocks]
    total_flops = sum(block_flops)
    total_outputs = sum(block_outputs)

    state_latency, state_energy = _build_state_costs(total_flops, total_outputs, config.num_pes, active_pes)

    best: SearchResult | None = None

    for sub_batch in config.candidate_sub_batches:
        if config.batch_size % sub_batch != 0:
            continue
        total_sub_batches = config.batch_size // sub_batch

        min_active = max(1, min(config.min_active_states, total_sub_batches, num_states))
        if min_active * config.min_batch_if_active > total_sub_batches:
            continue

        max_caps = _state_caps(total_sub_batches, num_states, config.max_state_share)
        if max_caps is not None:
            if sum(max_caps) < total_sub_batches:
                continue
            if sum(1 for c in max_caps if c >= config.min_batch_if_active) < min_active:
                continue

        sct = build_weighted_sct(num_states, block_flops, total_sub_batches)
        met = build_memory_table(sct, sram_keep_ratio=0.6)

        if not _memory_feasible(sct, met, block_outputs, config.sram_capacity, config.dram_capacity):
            continue

        milp_solution = solver.optimize_state_mix(
            total_sub_batches=total_sub_batches,
            state_latency=state_latency,
            state_energy=state_energy,
            weight_latency=config.weight_latency,
            weight_energy=config.weight_energy,
            max_batches_per_state=max_caps,
            min_active_states=min_active,
            min_batch_if_active=config.min_batch_if_active,
        )

        total_latency = milp_solution.latency
        total_energy = milp_solution.energy
        total_edp = edp(total_latency, total_energy)

        result = SearchResult(
            best_sub_batch=sub_batch,
            state_order=state_order,
            state_categories=categories,
            active_pes=active_pes,
            state_unit_latency=state_latency,
            state_unit_energy=state_energy,
            sct=sct,
            met=met,
            milp_solution=milp_solution,
            total_latency=total_latency,
            total_energy=total_energy,
            total_edp=total_edp,
        )
        if best is None or result.total_edp < best.total_edp:
            best = result

    if best is None:
        raise RuntimeError("No feasible schedule found. Check batch and capacity constraints.")
    return best
