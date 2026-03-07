from __future__ import annotations

from dataclasses import dataclass, field, replace
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
    noc_hops_compute: float = 1.0
    noc_hops_sram: float = 1.0

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

    # Hierarchical optimization (Section 6 style).
    enable_hierarchical_pipeline: bool = False
    max_hierarchy_depth: int = 2
    max_hierarchy_iters: int = 3
    hierarchy_theta: float = 0.02
    hierarchy_smoothing: float = 0.5
    hierarchy_min_child_blocks: int = 2


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
    hierarchy_level: int = 0
    hierarchy_notes: list[str] = field(default_factory=list)
    hierarchy_traces: list[dict[str, object]] = field(default_factory=list)


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


def _estimate_dep_traffic(
    sct: SchedulingTable,
    met: MemoryTable,
    block_volumes: Sequence[float],
    block_dependencies: Sequence[tuple[int, int]],
) -> tuple[float, float, float]:
    """Estimate (Dep_C, Dep_S, Dep_D) in MB using ScT/MeT + dependencies.

    This follows Eq.19/21 variable semantics at a coarse granularity:
    - Dep_C: direct producer->consumer transfer in the same state
    - Dep_S: transfer from SRAM-resident parent outputs
    - Dep_D: transfer from DRAM when SRAM cannot satisfy demand
    """

    n_states = sct.num_states
    n_blocks = sct.num_blocks

    deps_by_child: dict[int, list[int]] = {}
    for p, c in block_dependencies:
        if 0 <= p < n_blocks and 0 <= c < n_blocks and p != c:
            deps_by_child.setdefault(c, []).append(p)

    # Delta processed sub-batches per state/block.
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

            parents = deps_by_child.get(child, [])
            if not parents:
                # Source/input block: conservatively model as DRAM-fed.
                dep_d += need_sb * float(block_volumes[child])
                continue

            per_parent_need = need_sb / float(max(1, len(parents)))
            for parent in parents:
                vol = float(block_volumes[parent])

                # Dep_C: direct same-state producer->consumer transfer.
                parent_new = float(delta[i][parent])
                from_c = min(per_parent_need, parent_new)
                remain = per_parent_need - from_c

                # Dep_S: read from SRAM range tracked by (MeT_S, ScT].
                sram_live = max(0.0, float(sct.get(prev_i, parent)) - float(met.sram[prev_i, parent]))
                from_s = min(remain, sram_live)
                remain -= from_s

                # Dep_D: fallback to DRAM range tracked by (MeT_D, ScT].
                dram_live = max(0.0, float(sct.get(prev_i, parent)) - float(met.dram[prev_i, parent]))
                from_d = min(remain, dram_live)
                remain -= from_d

                if remain > 0.0:
                    # Conservative completion from DRAM when historical ranges are insufficient.
                    from_d += remain

                dep_c += from_c * vol
                dep_s += from_s * vol
                dep_d += from_d * vol

    return dep_c, dep_s, dep_d


def _estimate_memory_cost(
    sct: SchedulingTable,
    met: MemoryTable,
    block_volumes: Sequence[float],
    block_dependencies: Sequence[tuple[int, int]],
    noc_bandwidth: float,
    dram_bandwidth: float | None,
    noc_energy_per_unit: float,
    dram_energy_per_unit: float,
    dram_noc_hops: float,
    noc_hops_compute: float,
    noc_hops_sram: float,
) -> tuple[float, float]:
    dep_c, dep_s, dep_d = _estimate_dep_traffic(
        sct=sct,
        met=met,
        block_volumes=block_volumes,
        block_dependencies=block_dependencies,
    )

    # Eq.19: (Dep_C*H_C + Dep_S*H_S + Dep_D*H_D)/BW_NoC + Dep_D/BW_D
    h_c = max(0.0, float(noc_hops_compute))
    h_s = max(0.0, float(noc_hops_sram))
    h_d = max(0.0, float(dram_noc_hops))

    noc_traffic = dep_c * h_c + dep_s * h_s + dep_d * h_d
    noc_term = noc_latency(noc_traffic, max(1e-9, float(noc_bandwidth)))

    dram_bw = float(dram_bandwidth) if dram_bandwidth is not None else float(noc_bandwidth)
    dram_term = noc_latency(dep_d, max(1e-9, dram_bw))
    mem_latency = noc_term + dram_term

    # Eq.21: E_NoC*(Dep_C*H_C + Dep_S*H_S + Dep_D*H_D) + E_DRAM*Dep_D
    mem_energy = (
        communication_energy(noc_traffic, max(0.0, float(noc_energy_per_unit)))
        + communication_energy(dep_d, max(0.0, float(dram_energy_per_unit)))
    )
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
    hierarchy_level: int,
    hierarchy_notes: list[str] | None,
    trace_path: str | None,
) -> SearchResult:
    assert candidate.met_opt is not None

    total_latency = _combine_total_latency(
        candidate.compute_latency, candidate.memory_latency, latency_combine_mode
    )
    total_energy = candidate.compute_energy + candidate.memory_energy
    total_edp = edp(total_latency, total_energy)

    traces: list[dict[str, object]] = []
    if trace_path is not None:
        traces.append(
            {
                "path": str(trace_path),
                "level": int(hierarchy_level),
                "best_sub_batch": int(candidate.sub_batch),
                "scheduled_blocks": list(block_names),
                "state_order": list(state_order),
                "state_categories": list(categories),
                "state_batches": list(candidate.sct_opt.state_workloads),
                "sct": candidate.sct_opt.sct.table.tolist(),
                "met_s": candidate.met_opt.table.sram.tolist(),
                "met_d": candidate.met_opt.table.dram.tolist(),
            }
        )

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
        hierarchy_level=hierarchy_level,
        hierarchy_notes=list(hierarchy_notes or []),
        hierarchy_traces=traces,
    )


def _clone_block_tree(block: Block) -> Block:
    return Block(
        name=block.name,
        layers=list(block.layers),
        sub_blocks=[_clone_block_tree(sb) for sb in block.sub_blocks],
    )


def _child_blocks_from_block(block: Block) -> list[Block]:
    # Prefer explicit nested block hierarchy generated by merge.
    if len(block.sub_blocks) >= 2:
        return [_clone_block_tree(sb) for sb in block.sub_blocks]

    # Fallback: split a multi-layer block into layer-level children.
    if len(block.layers) >= 2:
        return [Block(name=ly.name, layers=[ly]) for ly in block.layers]

    return []


def _child_cfg(parent_cfg: SearchConfig, share: float, child_count: int) -> SearchConfig:
    share = max(1e-6, min(1.0, float(share)))
    child_pes = max(1, int(round(parent_cfg.num_pes * share)))

    if parent_cfg.num_pes > 0:
        sram_per_pe = float(parent_cfg.sram_capacity) / float(parent_cfg.num_pes)
    else:
        sram_per_pe = float(parent_cfg.sram_capacity)

    child_sram = max(1e-6, sram_per_pe * child_pes)
    child_max_layers = max(2, min(parent_cfg.max_layers_per_block, max(2, child_count)))

    return replace(
        parent_cfg,
        num_pes=child_pes,
        sram_capacity=child_sram,
        max_layers_per_block=child_max_layers,
    )


def _flat_search_prepared(
    work_blocks: Sequence[Block],
    block_deps: Sequence[tuple[int, int]],
    config: SearchConfig,
    block_flops_override: Sequence[float] | None = None,
    block_outputs_override: Sequence[float] | None = None,
    hierarchy_level: int = 0,
    hierarchy_notes: list[str] | None = None,
    trace_path: str | None = None,
) -> SearchResult:
    if not work_blocks:
        raise ValueError("blocks must not be empty")

    if block_flops_override is None:
        block_flops = [b.total_flops() for b in work_blocks]
    else:
        if len(block_flops_override) != len(work_blocks):
            raise ValueError("block_flops_override length mismatch")
        block_flops = [max(1e-9, float(v)) for v in block_flops_override]

    if block_outputs_override is None:
        block_outputs = [b.total_output_size() for b in work_blocks]
    else:
        if len(block_outputs_override) != len(work_blocks):
            raise ValueError("block_outputs_override length mismatch")
        block_outputs = [max(1e-9, float(v)) for v in block_outputs_override]

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
            block_dependencies=block_deps,
            noc_bandwidth=config.noc_bandwidth,
            dram_bandwidth=config.dram_bandwidth,
            noc_energy_per_unit=config.noc_energy_per_unit,
            dram_energy_per_unit=config.dram_energy_per_unit,
            dram_noc_hops=config.dram_noc_hops,
            noc_hops_compute=config.noc_hops_compute,
            noc_hops_sram=config.noc_hops_sram,
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
        block_deps=list(block_deps),
        state_order=state_order,
        categories=categories,
        active_pes=active_pes,
        state_active_blocks=state_active_blocks,
        latency_combine_mode=config.latency_combine_mode,
        hierarchy_level=hierarchy_level,
        hierarchy_notes=hierarchy_notes,
        trace_path=trace_path,
    )


def _hierarchical_search(
    blocks: Sequence[Block],
    config: SearchConfig,
    depth: int,
    lineage: list[str],
) -> SearchResult:
    work_blocks, block_deps = _prepare_blocks_and_dependencies(blocks, config)
    base_flops = [max(1e-9, b.total_flops()) for b in work_blocks]
    base_outputs = [max(1e-9, b.total_output_size()) for b in work_blocks]

    current_flops = list(base_flops)
    current_outputs = list(base_outputs)
    notes: list[str] = []
    trace_map: dict[str, dict[str, object]] = {}

    best_result: SearchResult | None = None

    max_iters = max(1, int(config.max_hierarchy_iters))
    smoothing = max(0.0, min(1.0, float(config.hierarchy_smoothing)))

    for iter_idx in range(max_iters):
        flat_now = _flat_search_prepared(
            work_blocks=work_blocks,
            block_deps=block_deps,
            config=config,
            block_flops_override=current_flops,
            block_outputs_override=current_outputs,
            hierarchy_level=len(lineage),
            hierarchy_notes=notes,
            trace_path="root" if not lineage else "root/" + "/".join(lineage),
        )

        for tr in flat_now.hierarchy_traces:
            trace_map[str(tr.get("path", f"level{len(lineage)}"))] = tr

        notes.append(
            f"level={len(lineage)} iter={iter_idx} parent_edp={flat_now.total_edp:.6e} blocks={len(work_blocks)}"
        )

        if best_result is None or flat_now.total_edp < best_result.total_edp:
            best_result = flat_now

        if depth <= 1:
            notes.append(f"level={len(lineage)} depth_limit_reached")
            break

        next_flops = list(current_flops)
        next_outputs = list(current_outputs)
        changed = False

        total_now_flops = max(1e-9, float(sum(current_flops)))

        for bi, blk in enumerate(work_blocks):
            child_blocks = _child_blocks_from_block(blk)
            if len(child_blocks) < int(config.hierarchy_min_child_blocks):
                continue

            share = max(1e-6, current_flops[bi] / total_now_flops)
            child_cfg = _child_cfg(config, share=share, child_count=len(child_blocks))
            child_result = _hierarchical_search(
                blocks=child_blocks,
                config=child_cfg,
                depth=depth - 1,
                lineage=lineage + [blk.name],
            )
            for tr in child_result.hierarchy_traces:
                trace_map[str(tr.get("path", f"child/{blk.name}"))] = tr

            child_sub_batches = max(1, int(child_cfg.batch_size // max(1, child_result.best_sub_batch)))
            flops_proxy = (
                child_result.total_latency
                * max(1e-9, float(child_cfg.compute_power_per_tile))
                * float(max(1, child_cfg.num_pes))
            ) / float(child_sub_batches)

            child_dram_bw = (
                float(child_cfg.dram_bandwidth)
                if child_cfg.dram_bandwidth is not None
                else float(child_cfg.noc_bandwidth)
            )
            denom = (
                max(0.0, float(child_cfg.dram_noc_hops)) / max(1e-9, float(child_cfg.noc_bandwidth))
                + 1.0 / max(1e-9, child_dram_bw)
            )
            total_traffic_volume = (
                float(child_result.memory_latency) / max(1e-12, denom)
                if child_result.memory_latency > 0
                else 0.0
            )
            output_proxy = max(1e-9, total_traffic_volume / float(child_sub_batches))

            new_flops = (1.0 - smoothing) * current_flops[bi] + smoothing * max(1e-9, flops_proxy)
            new_outputs = (1.0 - smoothing) * current_outputs[bi] + smoothing * output_proxy

            rel_f = abs(new_flops - current_flops[bi]) / max(1e-9, current_flops[bi])
            rel_o = abs(new_outputs - current_outputs[bi]) / max(1e-9, current_outputs[bi])
            if rel_f > 1e-3 or rel_o > 1e-3:
                changed = True

            next_flops[bi] = new_flops
            next_outputs[bi] = new_outputs

            notes.append(
                "level={} iter={} block={} child_blocks={} child_edp={:.6e} flops_proxy={:.6e} output_proxy={:.6e}".format(
                    len(lineage),
                    iter_idx,
                    blk.name,
                    len(child_blocks),
                    child_result.total_edp,
                    flops_proxy,
                    output_proxy,
                )
            )

        if not changed:
            notes.append(f"level={len(lineage)} iter={iter_idx} no_child_update")
            break

        flat_next = _flat_search_prepared(
            work_blocks=work_blocks,
            block_deps=block_deps,
            config=config,
            block_flops_override=next_flops,
            block_outputs_override=next_outputs,
            hierarchy_level=len(lineage),
            hierarchy_notes=notes,
            trace_path="root" if not lineage else "root/" + "/".join(lineage),
        )

        for tr in flat_next.hierarchy_traces:
            trace_map[str(tr.get("path", f"level{len(lineage)}"))] = tr

        if best_result is None or flat_next.total_edp < best_result.total_edp:
            best_result = flat_next

        improvement = (flat_now.total_edp - flat_next.total_edp) / max(1e-9, abs(flat_now.total_edp))
        notes.append(
            f"level={len(lineage)} iter={iter_idx} child_feedback_improvement={improvement:.6e}"
        )

        current_flops = next_flops
        current_outputs = next_outputs

        if improvement <= float(config.hierarchy_theta):
            notes.append(
                f"level={len(lineage)} iter={iter_idx} converge_by_theta theta={config.hierarchy_theta}"
            )
            break

    assert best_result is not None
    best_result.hierarchy_notes = notes
    best_result.hierarchy_level = len(lineage)
    best_result.hierarchy_traces = [trace_map[k] for k in sorted(trace_map.keys())]
    return best_result


def search_schedule(
    blocks: Sequence[Block],
    config: SearchConfig,
    solver: object | None = None,
) -> SearchResult:
    del solver  # kept for compatibility with older entry points

    if not blocks:
        raise ValueError("blocks must not be empty")

    use_hier = bool(config.enable_hierarchical_pipeline) and int(config.max_hierarchy_depth) > 1
    if use_hier:
        return _hierarchical_search(
            blocks=blocks,
            config=config,
            depth=max(1, int(config.max_hierarchy_depth)),
            lineage=[],
        )

    work_blocks, block_deps = _prepare_blocks_and_dependencies(blocks, config)
    return _flat_search_prepared(
        work_blocks=work_blocks,
        block_deps=block_deps,
        config=config,
        hierarchy_level=0,
        hierarchy_notes=[],
        trace_path="root",
    )
