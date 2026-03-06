from __future__ import annotations

from dataclasses import dataclass
from math import floor
from typing import Sequence

from cost_model.energy_model import communication_energy, edp
from cost_model.latency_model import noc_latency
from scheduler.block import Block, derive_block_dependencies, merge_linear_blocks
from scheduler.memory_table import MemoryOptimizationResult, MemoryTable, optimize_memory_table
from scheduler.milp_solver import MilpSolution
from scheduler.paper_milp import optimize_sct_table
from scheduler.scheduling_table import SchedulingTable


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
    enable_chain_block_merge: bool = True
    max_layers_per_block: int = 3
    min_layers_per_block: int = 2
    noc_bandwidth: float = 4096.0
    dram_energy_per_unit: float = 0.0075

    # Optional balancing constraints for state allocation.
    min_active_states: int = 1
    min_batch_if_active: int = 1
    max_state_share: float = 1.0


@dataclass
class SearchResult:
    best_sub_batch: int
    state_order: list[str]
    state_categories: list[str]
    active_pes: list[int]
    state_active_blocks: list[list[int]]
    scheduled_blocks: list[str]
    block_dependencies: list[tuple[int, int]]
    state_unit_latency: list[float]
    state_unit_energy: list[float]
    sct: SchedulingTable
    met: MemoryTable
    milp_solution: MilpSolution
    sct_solver_name: str
    met_solver_name: str
    compute_latency: float
    compute_energy: float
    memory_latency: float
    memory_energy: float
    total_latency: float
    total_energy: float
    total_edp: float


def _active_blocks_for_state(state: int, num_blocks: int) -> list[int]:
    # For block j (0-based), its canonical active states are [j, num_blocks + j - 1].
    active = []
    for j in range(num_blocks):
        if j <= state <= num_blocks + j - 1:
            active.append(j)
    return active


def _build_state_metadata(
    num_blocks: int,
    num_pes: int,
    num_states: int | None,
) -> tuple[int, list[str], list[str], list[int], list[list[int]]]:
    if num_blocks <= 0:
        raise ValueError("num_blocks must be > 0")
    if num_pes <= 0:
        raise ValueError("num_pes must be > 0")

    canonical_states = 2 * num_blocks - 1
    n_states = canonical_states if num_states is None else num_states
    if n_states <= 0:
        raise ValueError("num_states must be > 0")

    state_order = [f"S{i}" for i in range(n_states)]
    categories: list[str] = []
    active_blocks: list[list[int]] = []
    active_pes: list[int] = []

    for i in range(n_states):
        if n_states == canonical_states:
            blocks_i = _active_blocks_for_state(i, num_blocks)
            if i < num_blocks - 1:
                cat = "warmup"
            elif i == num_blocks - 1:
                cat = "steady"
            else:
                cat = "drain"
        else:
            blocks_i = _active_blocks_for_state(min(i, canonical_states - 1), num_blocks)
            cat = "generic"

        categories.append(cat)
        active_blocks.append(blocks_i)
        active_pes.append(max(1, min(num_pes, len(blocks_i))))

    return n_states, state_order, categories, active_pes, active_blocks


def _default_linear_dependencies(num_blocks: int) -> list[tuple[int, int]]:
    return [(i, i + 1) for i in range(num_blocks - 1)]


def _state_caps(total_sub_batches: int, num_states: int, max_state_share: float) -> list[int] | None:
    if max_state_share <= 0 or max_state_share > 1.0:
        raise ValueError("max_state_share must be in (0, 1]")

    if max_state_share >= 1.0:
        return None

    cap = max(1, floor(total_sub_batches * max_state_share))
    return [cap for _ in range(num_states)]



def _caps_cover_pipeline_windows(
    caps: Sequence[int],
    total_sub_batches: int,
    num_blocks: int,
    num_states: int,
) -> bool:
    for j in range(num_blocks):
        window_sum = 0
        for i in range(num_states):
            if j <= i <= num_blocks + j - 1:
                window_sum += int(caps[i])
        if window_sum < total_sub_batches:
            return False
    return True
def _estimate_memory_cost(
    sct: SchedulingTable,
    met: MemoryTable,
    block_volumes: Sequence[float],
    noc_bandwidth: float,
    dram_energy_per_unit: float,
) -> tuple[float, float]:
    dram_live_volume = 0.0
    for i in range(sct.num_states):
        for j in range(sct.num_blocks):
            dram_live_volume += (sct.get(i, j) - float(met.dram[i, j])) * float(block_volumes[j])

    mem_latency = noc_latency(dram_live_volume, max(1e-9, noc_bandwidth))
    mem_energy = communication_energy(dram_live_volume, max(0.0, dram_energy_per_unit))
    return mem_latency, mem_energy


def _prepare_blocks_and_dependencies(
    blocks: Sequence[Block],
    cfg: SearchConfig,
) -> tuple[list[Block], list[tuple[int, int]]]:
    prepared = [Block(name=b.name, layers=list(b.layers), sub_blocks=list(b.sub_blocks)) for b in blocks]
    deps = derive_block_dependencies(prepared)
    if not deps and len(prepared) > 1:
        deps = _default_linear_dependencies(len(prepared))

    if not cfg.enable_chain_block_merge:
        return prepared, deps

    merged_blocks, merged_deps, _ = merge_linear_blocks(
        prepared,
        deps,
        max_layers_per_block=cfg.max_layers_per_block,
        min_layers_per_block=cfg.min_layers_per_block,
    )
    if not merged_deps and len(merged_blocks) > 1:
        merged_deps = _default_linear_dependencies(len(merged_blocks))

    return merged_blocks, merged_deps


def search_schedule(
    blocks: Sequence[Block],
    config: SearchConfig,
    solver: object | None = None,
) -> SearchResult:
    del solver  # kept for compatibility with older entry points

    if not blocks:
        raise ValueError("blocks must not be empty")

    work_blocks, block_deps = _prepare_blocks_and_dependencies(blocks, config)

    block_flops = [b.total_flops() for b in work_blocks]
    block_outputs = [b.total_output_size() for b in work_blocks]
    block_names = [b.name for b in work_blocks]

    n_states, state_order, categories, active_pes, state_active_blocks = _build_state_metadata(
        num_blocks=len(work_blocks),
        num_pes=config.num_pes,
        num_states=config.num_states,
    )

    best: SearchResult | None = None

    for sub_batch in config.candidate_sub_batches:
        if sub_batch <= 0 or config.batch_size % sub_batch != 0:
            continue

        total_sub_batches = config.batch_size // sub_batch

        min_active = max(1, min(config.min_active_states, total_sub_batches, n_states))
        if min_active * config.min_batch_if_active > total_sub_batches:
            continue

        caps = _state_caps(total_sub_batches, n_states, config.max_state_share)
        if caps is not None:
            if sum(caps) < total_sub_batches:
                continue
            if not _caps_cover_pipeline_windows(caps, total_sub_batches, len(work_blocks), n_states):
                caps = None

        sct_opt = optimize_sct_table(
            block_flops=block_flops,
            block_outputs=block_outputs,
            total_sub_batches=total_sub_batches,
            block_dependencies=block_deps,
            num_pes=config.num_pes,
            weight_latency=config.weight_latency,
            weight_energy=config.weight_energy,
            num_states=n_states,
            min_active_states=min_active,
            min_batch_if_active=config.min_batch_if_active,
            max_batches_per_state=caps,
        )

        met_opt: MemoryOptimizationResult = optimize_memory_table(
            sct=sct_opt.sct,
            block_volumes=block_outputs,
            block_dependencies=block_deps,
            sram_capacity=config.sram_capacity,
            dram_capacity=config.dram_capacity,
            heuristic_sram_keep_ratio=0.6,
        )

        compute_latency = sum(
            sct_opt.state_workloads[i] * sct_opt.state_latency_coeff[i]
            for i in range(len(sct_opt.state_workloads))
        )
        compute_energy = sum(
            sct_opt.state_workloads[i] * sct_opt.state_energy_coeff[i]
            for i in range(len(sct_opt.state_workloads))
        )

        memory_latency, memory_energy = _estimate_memory_cost(
            sct=sct_opt.sct,
            met=met_opt.table,
            block_volumes=block_outputs,
            noc_bandwidth=config.noc_bandwidth,
            dram_energy_per_unit=config.dram_energy_per_unit,
        )

        total_latency = compute_latency + memory_latency
        total_energy = compute_energy + memory_energy
        total_edp = edp(total_latency, total_energy)

        result = SearchResult(
            best_sub_batch=sub_batch,
            state_order=state_order,
            state_categories=categories,
            active_pes=active_pes,
            state_active_blocks=state_active_blocks,
            scheduled_blocks=block_names,
            block_dependencies=block_deps,
            state_unit_latency=sct_opt.state_latency_coeff,
            state_unit_energy=sct_opt.state_energy_coeff,
            sct=sct_opt.sct,
            met=met_opt.table,
            milp_solution=MilpSolution(
                state_batches=list(sct_opt.state_workloads),
                latency=compute_latency,
                energy=compute_energy,
                objective=sct_opt.objective,
                solver_name=sct_opt.solver_name,
            ),
            sct_solver_name=sct_opt.solver_name,
            met_solver_name=met_opt.solver_name,
            compute_latency=compute_latency,
            compute_energy=compute_energy,
            memory_latency=memory_latency,
            memory_energy=memory_energy,
            total_latency=total_latency,
            total_energy=total_energy,
            total_edp=total_edp,
        )

        if best is None or result.total_edp < best.total_edp:
            best = result

    if best is None:
        raise RuntimeError("No feasible schedule found. Check batch and capacity constraints.")

    return best




