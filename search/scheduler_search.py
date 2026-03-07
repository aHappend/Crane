from __future__ import annotations

from dataclasses import dataclass
from math import floor
from typing import Sequence

from cost_model.energy_model import communication_energy, edp
from cost_model.latency_model import noc_latency
from scheduler.block import Block, derive_block_dependencies, merge_linear_blocks
from scheduler.memory_table import MemoryOptimizationResult, MemoryTable, optimize_memory_table
from scheduler.milp_solver import MilpSolution
from scheduler.paper_milp import ScTOptimizationResult, optimize_sct_table
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
    dram_bandwidth: float | None = None
    noc_energy_per_unit: float = 0.0
    dram_energy_per_unit: float = 0.0075
    dram_noc_hops: float = 1.0

    # Compute-side hardware model parameters (default: normalized).
    compute_power_per_tile: float = 1.0
    compute_energy_per_op: float = 1e-12

    # Optional balancing constraints for state allocation.
    min_active_states: int = 1
    min_batch_if_active: int = 1
    max_state_share: float = 1.0

    # Paper-like search flow.
    strict_paper_mode: bool = True
    top_k1: int = 4
    top_k2: int = 2
    use_edp_objective: bool = True
    dependency_gap: int = 0
    allow_solver_fallback: bool = False
    latency_combine_mode: str = "max"


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


@dataclass
class _Candidate:
    sub_batch: int
    total_sub_batches: int
    sct_opt: ScTOptimizationResult
    compute_latency: float
    compute_energy: float
    compute_edp: float
    met_opt: MemoryOptimizationResult | None = None
    memory_latency: float = 0.0
    memory_energy: float = 0.0
    memory_edp: float = 0.0


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


def _combine_total_latency(compute_latency: float, memory_latency: float, mode: str) -> float:
    mode_norm = (mode or "max").strip().lower()
    if mode_norm == "sum":
        return float(compute_latency + memory_latency)
    if mode_norm != "max":
        raise ValueError(f"unsupported latency_combine_mode={mode}")
    return float(max(compute_latency, memory_latency))


def _estimate_memory_cost(
    sct: SchedulingTable,
    met: MemoryTable,
    block_volumes: Sequence[float],
    noc_bandwidth: float,
    dram_bandwidth: float | None,
    noc_energy_per_unit: float,
    dram_energy_per_unit: float,
    dram_noc_hops: float,
) -> tuple[float, float]:
    dram_live_volume = 0.0
    for i in range(sct.num_states):
        for j in range(sct.num_blocks):
            dram_live_volume += (sct.get(i, j) - float(met.dram[i, j])) * float(block_volumes[j])

    # Eq.19 (simplified for DRAM-sourced traffic):
    # L_traffic = V/BW_NoC * H_D + V/BW_D
    noc_term = noc_latency(dram_live_volume * max(0.0, float(dram_noc_hops)), max(1e-9, float(noc_bandwidth)))
    dram_bw = float(dram_bandwidth) if dram_bandwidth is not None else float(noc_bandwidth)
    dram_term = noc_latency(dram_live_volume, max(1e-9, dram_bw))
    mem_latency = noc_term + dram_term

    # Eq.21 (simplified): E = V*(E_NoC*H_D + E_DRAM)
    per_mb = max(0.0, float(noc_energy_per_unit)) * max(0.0, float(dram_noc_hops)) + max(0.0, float(dram_energy_per_unit))
    mem_energy = communication_energy(dram_live_volume, per_mb)
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


def _build_result_from_candidate(
    candidate: _Candidate,
    block_names: list[str],
    block_deps: list[tuple[int, int]],
    state_order: list[str],
    categories: list[str],
    active_pes: list[int],
    state_active_blocks: list[list[int]],
    latency_combine_mode: str,
) -> SearchResult:
    assert candidate.met_opt is not None

    total_latency = _combine_total_latency(
        candidate.compute_latency, candidate.memory_latency, latency_combine_mode
    )
    total_energy = candidate.compute_energy + candidate.memory_energy
    total_edp = edp(total_latency, total_energy)

    return SearchResult(
        best_sub_batch=candidate.sub_batch,
        state_order=state_order,
        state_categories=categories,
        active_pes=active_pes,
        state_active_blocks=state_active_blocks,
        scheduled_blocks=block_names,
        block_dependencies=block_deps,
        state_unit_latency=candidate.sct_opt.state_latency_coeff,
        state_unit_energy=candidate.sct_opt.state_energy_coeff,
        sct=candidate.sct_opt.sct,
        met=candidate.met_opt.table,
        milp_solution=MilpSolution(
            state_batches=list(candidate.sct_opt.state_workloads),
            latency=candidate.compute_latency,
            energy=candidate.compute_energy,
            objective=candidate.sct_opt.objective,
            solver_name=candidate.sct_opt.solver_name,
        ),
        sct_solver_name=candidate.sct_opt.solver_name,
        met_solver_name=candidate.met_opt.solver_name,
        compute_latency=candidate.compute_latency,
        compute_energy=candidate.compute_energy,
        memory_latency=candidate.memory_latency,
        memory_energy=candidate.memory_energy,
        total_latency=total_latency,
        total_energy=total_energy,
        total_edp=total_edp,
    )


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

    stage1_candidates: list[_Candidate] = []

    # Stage-1: enumerate BS_sub and optimize ScT (Eq.1-6, Eq.22).
    for sub_batch in config.candidate_sub_batches:
        if sub_batch <= 0 or config.batch_size % sub_batch != 0:
            continue

        total_sub_batches = config.batch_size // sub_batch

        min_active = max(1, min(int(config.min_active_states), n_states))

        caps = _state_caps(total_sub_batches, n_states, config.max_state_share)
        if caps is not None:
            if not _caps_cover_pipeline_windows(caps, total_sub_batches, len(work_blocks), n_states):
                caps = None

        try:
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
                use_edp_objective=config.use_edp_objective,
                dependency_gap=config.dependency_gap,
                compute_power_per_tile=config.compute_power_per_tile,
                energy_per_op=config.compute_energy_per_op,
                allow_fallback=config.allow_solver_fallback,
            )
        except Exception:
            continue

        compute_latency = sum(
            sct_opt.state_workloads[i] * sct_opt.state_latency_coeff[i]
            for i in range(len(sct_opt.state_workloads))
        )
        compute_energy = sum(
            sct_opt.state_workloads[i] * sct_opt.state_energy_coeff[i]
            for i in range(len(sct_opt.state_workloads))
        )

        stage1_candidates.append(
            _Candidate(
                sub_batch=sub_batch,
                total_sub_batches=total_sub_batches,
                sct_opt=sct_opt,
                compute_latency=compute_latency,
                compute_energy=compute_energy,
                compute_edp=edp(compute_latency, compute_energy),
            )
        )

    if not stage1_candidates:
        raise RuntimeError("No feasible ScT candidate found. Check batch and constraints.")

    stage1_candidates.sort(key=lambda x: x.sct_opt.objective)
    if config.strict_paper_mode:
        k1 = max(1, min(int(config.top_k1), len(stage1_candidates)))
        stage1_candidates = stage1_candidates[:k1]

    # Stage-2: optimize MeT (Eq.7-12, Eq.23) on top-K1.
    stage2_candidates: list[_Candidate] = []
    for cand in stage1_candidates:
        try:
            met_opt = optimize_memory_table(
                sct=cand.sct_opt.sct,
                block_volumes=block_outputs,
                block_dependencies=block_deps,
                sram_capacity=config.sram_capacity,
                dram_capacity=config.dram_capacity,
                heuristic_sram_keep_ratio=0.6,
                noc_bandwidth=config.noc_bandwidth,
                dram_bandwidth=config.dram_bandwidth,
                noc_energy_per_unit=config.noc_energy_per_unit,
                dram_energy_per_unit=config.dram_energy_per_unit,
                dram_noc_hops=config.dram_noc_hops,
                weight_latency=config.weight_latency,
                weight_energy=config.weight_energy,
                use_edp_objective=config.use_edp_objective,
                allow_fallback=config.allow_solver_fallback,
            )
        except Exception:
            continue

        memory_latency, memory_energy = _estimate_memory_cost(
            sct=cand.sct_opt.sct,
            met=met_opt.table,
            block_volumes=block_outputs,
            noc_bandwidth=config.noc_bandwidth,
            dram_bandwidth=config.dram_bandwidth,
            noc_energy_per_unit=config.noc_energy_per_unit,
            dram_energy_per_unit=config.dram_energy_per_unit,
            dram_noc_hops=config.dram_noc_hops,
        )

        cand.met_opt = met_opt
        cand.memory_latency = memory_latency
        cand.memory_energy = memory_energy
        cand.memory_edp = edp(memory_latency, memory_energy)
        stage2_candidates.append(cand)

    if not stage2_candidates:
        raise RuntimeError("No feasible MeT candidate found after ScT optimization.")

    stage2_candidates.sort(key=lambda x: x.met_opt.objective if x.met_opt is not None else float("inf"))
    if config.strict_paper_mode:
        k2 = max(1, min(int(config.top_k2), len(stage2_candidates)))
        stage2_candidates = stage2_candidates[:k2]

    # Final choose by total EDP.
    best_cand = min(
        stage2_candidates,
        key=lambda x: edp(
            _combine_total_latency(x.compute_latency, x.memory_latency, config.latency_combine_mode),
            x.compute_energy + x.memory_energy,
        ),
    )

    return _build_result_from_candidate(
        candidate=best_cand,
        block_names=block_names,
        block_deps=block_deps,
        state_order=state_order,
        categories=categories,
        active_pes=active_pes,
        state_active_blocks=state_active_blocks,
        latency_combine_mode=config.latency_combine_mode,
    )





