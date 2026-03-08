from __future__ import annotations

import os
import time
from dataclasses import dataclass, field, replace
from math import ceil, floor
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
    top_k1_ratio: float = 0.5
    top_k2_ratio: float = 0.2
    use_all_sub_batch_factors: bool = False
    use_edp_objective: bool = True
    dependency_gap: int = 0
    allow_solver_fallback: bool = False
    latency_combine_mode: str = "max"
    verbose_progress: bool = False
    progress_prefix: str = ""

    # Hierarchical optimization (Section 6 style).
    enable_hierarchical_pipeline: bool = False
    derive_recursive_traces: bool = True
    max_hierarchy_depth: int = 2
    max_hierarchy_iters: int = 3
    hierarchy_theta: float = 0.02
    hierarchy_smoothing: float = 0.5
    hierarchy_min_child_blocks: int = 2

    # Structure refinement (cost-driven gradual partition).
    enable_structure_refinement: bool = True
    structure_refine_max_trials: int = 6
    structure_refine_min_improvement: float = 1e-4

    # Training co-support (Section 5.3).
    enable_training_recomputation: bool = False
    backward_compute_scale: float = 2.0
    backward_output_scale: float = 1.0
    recompute_compute_scale: float = 1.0
    recompute_output_scale: float = 1.0


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
    phase_results: dict[str, dict[str, object]] = field(default_factory=dict)


def _empty_state_block_override(
    num_states: int,
    num_blocks: int,
) -> list[list[float | None]]:
    return [[None for _ in range(num_blocks)] for _ in range(num_states)]



def _empty_state_int_bounds(
    num_states: int,
    num_blocks: int,
) -> list[list[int | None]]:
    return [[None for _ in range(num_blocks)] for _ in range(num_states)]



def _resample_monotone_cumulative(
    src: Sequence[float],
    out_len: int,
) -> list[int]:
    if out_len <= 0:
        return []
    if not src:
        return [0 for _ in range(out_len)]
    if out_len == 1:
        return [max(0, int(round(float(src[-1]))))]

    src_vals = [max(0.0, float(v)) for v in src]
    total = max(0, int(round(src_vals[-1])))
    out: list[int] = []
    src_last = max(1, len(src_vals) - 1)
    out_last = max(1, out_len - 1)
    for i in range(out_len):
        pos = float(i) * float(src_last) / float(out_last)
        lo = int(floor(pos))
        hi = min(src_last, lo + 1)
        alpha = max(0.0, min(1.0, pos - float(lo)))
        val = (1.0 - alpha) * src_vals[lo] + alpha * src_vals[hi]
        out.append(max(0, min(total, int(round(val)))))

    for i in range(1, len(out)):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    out[-1] = total
    return out


def _anchor_state_indices(num_states: int) -> list[int]:
    if num_states <= 0:
        return []
    if num_states == 1:
        return [0]

    anchors = {
        0,
        num_states - 1,
    }
    if num_states >= 3:
        anchors.add((num_states - 1) // 2)
    if num_states >= 5:
        anchors.add((num_states - 1) // 3)
        anchors.add(((num_states - 1) * 2) // 3)
    return sorted(anchors)


def _derive_child_completion_bounds_from_parent(
    parent_result: SearchResult,
    parent_block_index: int,
    child_num_states: int,
    child_num_blocks: int,
) -> tuple[list[list[int | None]], list[list[int | None]]]:
    if child_num_states <= 0 or child_num_blocks <= 0:
        raise ValueError('child_num_states and child_num_blocks must be positive')

    parent_cum = [
        max(0, int(round(parent_result.sct.get(i, parent_block_index))))
        for i in range(parent_result.sct.num_states)
    ]
    target = _resample_monotone_cumulative(parent_cum, child_num_states)
    total = max(0, target[-1] if target else 0)
    lb = _empty_state_int_bounds(child_num_states, child_num_blocks)
    ub = _empty_state_int_bounds(child_num_states, child_num_blocks)
    last_child = child_num_blocks - 1

    if total <= 0:
        lb[0][last_child] = 0
        ub[0][last_child] = 0
        lb[-1][last_child] = 0
        ub[-1][last_child] = 0
        return lb, ub

    anchors = _anchor_state_indices(child_num_states)
    # Keep only a few anchor states coupled to the parent curve. A dense
    # full-state boundary made long child chains infeasible in practice.
    base_slack = max(1, int(ceil(total * 0.25)))
    for idx, state in enumerate(anchors):
        val = int(target[state])
        if state == 0 or state == child_num_states - 1:
            lo = val
            hi = val
        else:
            anchor_slack = base_slack + max(0, idx - 1)
            lo = max(0, val - anchor_slack)
            hi = min(total, val + anchor_slack)
        lb[state][last_child] = lo
        ub[state][last_child] = hi

    lb[0][last_child] = 0
    ub[0][last_child] = 0
    lb[-1][last_child] = total
    ub[-1][last_child] = total
    return lb, ub

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




def _divisors(n: int) -> list[int]:
    out: set[int] = set()
    for d in range(1, int(n**0.5) + 1):
        if n % d == 0:
            out.add(d)
            out.add(n // d)
    return sorted(out)


def _effective_sub_batch_candidates(config: SearchConfig) -> list[int]:
    cfg = [int(x) for x in config.candidate_sub_batches if int(x) > 0]
    if config.strict_paper_mode and config.use_all_sub_batch_factors:
        cfg = sorted(set(cfg).union(_divisors(int(config.batch_size))))
    return cfg


def _topk_by_ratio_or_count(total: int, ratio: float, fallback_count: int) -> int:
    if total <= 0:
        return 0
    if ratio > 0.0:
        return max(1, min(total, int(ceil(total * ratio))))
    return max(1, min(total, int(fallback_count)))
def _progress_enabled(config: SearchConfig) -> bool:
    if bool(config.verbose_progress):
        return True
    env = os.environ.get("CRANE_LIVE_PROGRESS", "").strip().lower()
    return env in {"1", "true", "yes", "on"}


def _progress(config: SearchConfig, message: str) -> None:
    if not _progress_enabled(config):
        return
    ts = time.strftime("%H:%M:%S")
    prefix = (config.progress_prefix or "").strip()
    if prefix:
        print(f"[{ts}] [{prefix}] {message}", flush=True)
    else:
        print(f"[{ts}] {message}", flush=True)

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
                "state_active_blocks": [list(x) for x in state_active_blocks],
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




def _stamp_traces(
    traces: Sequence[dict[str, object]],
    *,
    parent_block: str,
    derived_mode: str,
    assumed_num_pes: int,
    assumed_pe_share: float,
) -> None:
    for trace in traces:
        trace["parent_block"] = parent_block
        trace["derived_mode"] = derived_mode
        trace["assumed_num_pes"] = int(assumed_num_pes)
        trace["assumed_pe_share"] = float(assumed_pe_share)
def _derive_or_default_dependencies(blocks: Sequence[Block]) -> list[tuple[int, int]]:
    deps = derive_block_dependencies(blocks)
    if not deps and len(blocks) > 1:
        deps = _default_linear_dependencies(len(blocks))
    return deps


def _replace_block_with_children(
    blocks: Sequence[Block],
    block_index: int,
    children: Sequence[Block],
) -> list[Block]:
    if block_index < 0 or block_index >= len(blocks):
        raise IndexError("block_index out of range")
    if not children:
        return [_clone_block_tree(b) for b in blocks]

    out: list[Block] = []
    for i, blk in enumerate(blocks):
        if i == block_index:
            out.extend(_clone_block_tree(ch) for ch in children)
        else:
            out.append(_clone_block_tree(blk))
    return out


def _candidate_child_expansions(
    blocks: Sequence[Block],
    min_children: int,
    max_trials: int,
) -> list[tuple[int, list[Block]]]:
    scored: list[tuple[tuple[float, float, float], int, list[Block]]] = []
    for i, blk in enumerate(blocks):
        children = _child_blocks_from_block(blk)
        if len(children) < max(2, int(min_children)):
            continue
        # Prioritize large/complex blocks first for gradual partition trials.
        score = (float(blk.layer_count()), float(len(children)), float(blk.total_flops()))
        scored.append((score, i, children))

    scored.sort(key=lambda x: x[0], reverse=True)
    if max_trials > 0:
        scored = scored[: max(1, int(max_trials))]
    return [(i, children) for _, i, children in scored]

def _derive_recursive_intra_block_traces(
    parent_blocks: Sequence[Block],
    parent_result: SearchResult,
    config: SearchConfig,
    depth_remaining: int,
    lineage: list[str],
) -> tuple[list[dict[str, object]], list[str]]:
    if depth_remaining <= 0:
        return [], []

    total_parent_flops = sum(max(1e-9, float(b.total_flops())) for b in parent_blocks)
    traces: list[dict[str, object]] = []
    notes: list[str] = []

    for bi, block in enumerate(parent_blocks):
        children = _child_blocks_from_block(block)
        if len(children) < 2:
            continue

        share = max(1e-6, float(block.total_flops()) / max(1e-9, total_parent_flops))
        child_cfg = _child_cfg(config, share=share, child_count=len(children))
        child_cfg = replace(
            child_cfg,
            candidate_sub_batches=[int(parent_result.best_sub_batch)],
            use_all_sub_batch_factors=False,
            strict_paper_mode=False,
            top_k1=1,
            top_k2=1,
            top_k1_ratio=0.0,
            top_k2_ratio=0.0,
            enable_hierarchical_pipeline=False,
            derive_recursive_traces=False,
            enable_structure_refinement=False,
            enable_chain_block_merge=False,
            max_hierarchy_depth=max(1, depth_remaining),
        )

        child_deps = _derive_or_default_dependencies(children)
        block_slug = f"{bi:02d}_{block.name}"
        child_path = "root/" + "/".join([*lineage, block_slug]) if lineage else f"root/{block_slug}"

        try:
            child_res = _flat_search_prepared(
                work_blocks=children,
                block_deps=child_deps,
                config=child_cfg,
                hierarchy_level=len(lineage) + 1,
                hierarchy_notes=[],
                trace_path=child_path,
                state_count_lower_bounds=child_lb,
                state_count_upper_bounds=child_ub,
            )
        except Exception as exc:
            notes.append(
                f"recursive_trace_failed parent={block.name} children={len(children)} reason={exc}"
            )
            continue

        _stamp_traces(
            child_res.hierarchy_traces,
            parent_block=block.name,
            derived_mode="intra_block",
            assumed_num_pes=int(child_cfg.num_pes),
            assumed_pe_share=float(share),
        )
        for trace in child_res.hierarchy_traces:
            traces.append(trace)

        notes.append(
            f"recursive_trace_ok parent={block.name} children={len(children)} "
            f"assumed_pes={child_cfg.num_pes} best_sub_batch={child_res.best_sub_batch}"
        )

        grand_traces, grand_notes = _derive_recursive_intra_block_traces(
            parent_blocks=children,
            parent_result=child_res,
            config=child_cfg,
            depth_remaining=depth_remaining - 1,
            lineage=[*lineage, block_slug],
        )
        traces.extend(grand_traces)
        notes.extend(grand_notes)

    return traces, notes


def _delta_matrix(sct: SchedulingTable) -> list[list[float]]:
    delta: list[list[float]] = []
    for i in range(sct.num_states):
        row: list[float] = []
        for j in range(sct.num_blocks):
            cur = float(sct.get(i, j))
            prev = float(sct.get(i - 1, j)) if i > 0 else 0.0
            row.append(max(0.0, cur - prev))
        delta.append(row)
    return delta


def _integer_alloc_for_active(
    active: Sequence[int],
    weights_by_block: Sequence[float],
    num_pes: int,
) -> list[int]:
    if not active:
        return []
    weights = [max(1e-9, float(weights_by_block[j])) for j in active]
    total = max(1e-9, sum(weights))
    alloc = [1 for _ in active]
    if num_pes > len(active):
        remaining = num_pes - len(active)
        raw = [remaining * w / total for w in weights]
        extra = [int(floor(v)) for v in raw]
        alloc = [a + e for a, e in zip(alloc, extra)]
        used = sum(extra)
        frac_order = sorted(
            range(len(active)),
            key=lambda idx: (raw[idx] - floor(raw[idx]), weights[idx]),
            reverse=True,
        )
        for idx in frac_order[: max(0, remaining - used)]:
            alloc[idx] += 1
    return alloc


def _estimate_block_pe_shares(
    work_blocks: Sequence[Block],
    res: SearchResult,
    num_pes: int,
) -> list[float]:
    n = len(work_blocks)
    if n == 0:
        return []

    weights = [max(1e-9, float(b.total_flops())) for b in work_blocks]
    delta = _delta_matrix(res.sct)
    assigned = [0.0 for _ in range(n)]
    totals = [0.0 for _ in range(n)]

    for i, row in enumerate(delta):
        active = [j for j, v in enumerate(row) if float(v) > 0.0]
        if not active:
            continue
        alloc = _integer_alloc_for_active(active, weights, num_pes)
        batch_weight = float(max(1, res.milp_solution.state_batches[i] if i < len(res.milp_solution.state_batches) else 1))
        for local_idx, j in enumerate(active):
            assigned[j] += float(alloc[local_idx]) * batch_weight
            totals[j] += float(max(1, num_pes)) * batch_weight

    total_weight = max(1e-9, sum(weights))
    out: list[float] = []
    for j in range(n):
        if totals[j] > 0.0:
            out.append(max(1e-6, min(1.0, assigned[j] / totals[j])))
        else:
            out.append(max(1e-6, min(1.0, weights[j] / total_weight)))
    return out



def _expand_per_subbatch_costs(
    state_batches: Sequence[int],
    state_coeffs: Sequence[float],
) -> list[float]:
    out: list[float] = []
    for batch_count, coeff in zip(state_batches, state_coeffs):
        cnt = max(0, int(batch_count))
        out.extend([max(0.0, float(coeff)) for _ in range(cnt)])
    return out



def _map_child_costs_to_parent_states(
    parent_result: SearchResult,
    parent_block_index: int,
    child_res: SearchResult,
) -> tuple[list[float | None], list[float | None]]:
    parent_delta = _delta_matrix(parent_result.sct)
    parent_counts = [
        int(round(parent_delta[i][parent_block_index]))
        for i in range(parent_result.sct.num_states)
    ]
    child_lat_seq = _expand_per_subbatch_costs(
        child_res.milp_solution.state_batches,
        child_res.state_unit_latency,
    )
    child_ene_seq = _expand_per_subbatch_costs(
        child_res.milp_solution.state_batches,
        child_res.state_unit_energy,
    )
    total_need = sum(max(0, v) for v in parent_counts)
    if len(child_lat_seq) < total_need or len(child_ene_seq) < total_need:
        raise RuntimeError('child schedule does not cover parent-assigned sub-batches')

    lat_override: list[float | None] = [None for _ in range(parent_result.sct.num_states)]
    ene_override: list[float | None] = [None for _ in range(parent_result.sct.num_states)]
    cursor = 0
    for i, cnt in enumerate(parent_counts):
        if cnt <= 0:
            continue
        lat_chunk = child_lat_seq[cursor : cursor + cnt]
        ene_chunk = child_ene_seq[cursor : cursor + cnt]
        if not lat_chunk or not ene_chunk:
            raise RuntimeError('failed to map child state costs to parent states')
        lat_override[i] = float(sum(lat_chunk) / len(lat_chunk))
        ene_override[i] = float(sum(ene_chunk) / len(ene_chunk))
        cursor += cnt
    return lat_override, ene_override

def _recursive_joint_optimize_prepared(
    work_blocks: Sequence[Block],
    block_deps: Sequence[tuple[int, int]],
    config: SearchConfig,
    depth_remaining: int,
    hierarchy_level: int,
    hierarchy_notes: list[str] | None,
    trace_path: str | None,
    final_counts_per_block: Sequence[int] | None = None,
    initial_counts_per_block: Sequence[int] | None = None,
    state_count_lower_bounds: Sequence[Sequence[int | None]] | None = None,
    state_count_upper_bounds: Sequence[Sequence[int | None]] | None = None,
) -> SearchResult:
    flat_cfg = replace(config, derive_recursive_traces=False)
    notes = list(hierarchy_notes or [])
    trace_map: dict[str, dict[str, object]] = {}

    current_result = _flat_search_prepared(
        work_blocks=work_blocks,
        block_deps=block_deps,
        config=flat_cfg,
        hierarchy_level=hierarchy_level,
        hierarchy_notes=notes,
        trace_path=trace_path,
        final_counts_per_block=final_counts_per_block,
        initial_counts_per_block=initial_counts_per_block,
        state_count_lower_bounds=state_count_lower_bounds,
        state_count_upper_bounds=state_count_upper_bounds,
    )
    _stamp_traces(
        current_result.hierarchy_traces,
        parent_block="ROOT",
        derived_mode="joint_root",
        assumed_num_pes=int(config.num_pes),
        assumed_pe_share=1.0,
    )
    for tr in current_result.hierarchy_traces:
        trace_map[str(tr.get("path", trace_path or "root"))] = tr

    if depth_remaining <= 1:
        current_result.hierarchy_notes = notes
        current_result.hierarchy_traces = [trace_map[k] for k in sorted(trace_map.keys())]
        return current_result

    rel_threshold = max(float(config.hierarchy_theta), float(config.structure_refine_min_improvement))
    max_joint_iters = max(1, min(3, int(config.max_hierarchy_iters)))

    for joint_iter in range(max_joint_iters):
        child_shares = _estimate_block_pe_shares(work_blocks, current_result, max(1, int(config.num_pes)))
        unit_lat_override: list[float | None] = [None for _ in work_blocks]
        unit_ene_override: list[float | None] = [None for _ in work_blocks]
        state_lat_override = _empty_state_block_override(current_result.sct.num_states, len(work_blocks))
        state_ene_override = _empty_state_block_override(current_result.sct.num_states, len(work_blocks))
        any_child = False

        for bi, block in enumerate(work_blocks):
            children = _child_blocks_from_block(block)
            if len(children) < 2:
                continue

            any_child = True
            child_cfg = _child_cfg(config, share=child_shares[bi], child_count=len(children))
            child_cfg = replace(
                child_cfg,
                candidate_sub_batches=[int(current_result.best_sub_batch)],
                use_all_sub_batch_factors=False,
                strict_paper_mode=False,
                top_k1=1,
                top_k2=1,
                top_k1_ratio=0.0,
                top_k2_ratio=0.0,
                enable_hierarchical_pipeline=False,
                derive_recursive_traces=False,
                enable_structure_refinement=False,
                enable_chain_block_merge=False,
            )
            child_deps = _derive_or_default_dependencies(children)
            child_num_states = int(child_cfg.num_states) if child_cfg.num_states is not None else (2 * len(children) - 1)
            child_lb, child_ub = _derive_child_completion_bounds_from_parent(
                current_result,
                parent_block_index=bi,
                child_num_states=child_num_states,
                child_num_blocks=len(children),
            )
            child_path = f"{trace_path or 'root'}/joint_{joint_iter + 1}_{bi:02d}_{block.name}"
            try:
                child_res = _recursive_joint_optimize_prepared(
                    work_blocks=children,
                    block_deps=child_deps,
                    config=child_cfg,
                    depth_remaining=depth_remaining - 1,
                    hierarchy_level=hierarchy_level + 1,
                    hierarchy_notes=[],
                    trace_path=child_path,
                    state_count_lower_bounds=child_lb,
                    state_count_upper_bounds=child_ub,
                )
                child_boundary_mode = "last_child_track"
            except Exception as exc:
                notes.append(
                    f"joint_iter={joint_iter} parent={block.name} boundary_retry=free reason={exc}"
                )
                child_res = _recursive_joint_optimize_prepared(
                    work_blocks=children,
                    block_deps=child_deps,
                    config=child_cfg,
                    depth_remaining=depth_remaining - 1,
                    hierarchy_level=hierarchy_level + 1,
                    hierarchy_notes=[],
                    trace_path=child_path,
                )
                child_boundary_mode = "free_fallback"
            total_sub_batches = max(1, int(config.batch_size // max(1, current_result.best_sub_batch)))
            unit_lat_override[bi] = float(child_res.total_latency) / float(total_sub_batches)
            unit_ene_override[bi] = float(child_res.total_energy) / float(total_sub_batches)
            parent_state_lat, parent_state_ene = _map_child_costs_to_parent_states(
                current_result,
                parent_block_index=bi,
                child_res=child_res,
            )
            for si in range(current_result.sct.num_states):
                state_lat_override[si][bi] = parent_state_lat[si]
                state_ene_override[si][bi] = parent_state_ene[si]
            notes.append(
                f"joint_iter={joint_iter} parent={block.name} children={len(children)} "
                f"share={child_shares[bi]:.4f} unit_lat={unit_lat_override[bi]:.6e} unit_ene={unit_ene_override[bi]:.6e} boundary={child_boundary_mode}"
            )
            _stamp_traces(
                child_res.hierarchy_traces,
                parent_block=block.name,
                derived_mode="joint_recursive",
                assumed_num_pes=int(child_cfg.num_pes),
                assumed_pe_share=float(child_shares[bi]),
            )
            for tr in child_res.hierarchy_traces:
                trace_map[str(tr.get('path', child_path))] = tr

        if not any_child:
            break

        refined = _flat_search_prepared(
            work_blocks=work_blocks,
            block_deps=block_deps,
            config=flat_cfg,
            hierarchy_level=hierarchy_level,
            hierarchy_notes=notes,
            trace_path=trace_path,
            final_counts_per_block=final_counts_per_block,
            initial_counts_per_block=initial_counts_per_block,
            state_count_lower_bounds=state_count_lower_bounds,
            state_count_upper_bounds=state_count_upper_bounds,
            block_unit_latency_override=unit_lat_override,
            block_unit_energy_override=unit_ene_override,
            state_block_latency_override=state_lat_override,
            state_block_energy_override=state_ene_override,
        )
        _stamp_traces(
            refined.hierarchy_traces,
            parent_block="ROOT",
            derived_mode="joint_refined",
            assumed_num_pes=int(config.num_pes),
            assumed_pe_share=1.0,
        )
        for tr in refined.hierarchy_traces:
            trace_map[str(tr.get("path", trace_path or "root"))] = tr

        rel = (current_result.total_edp - refined.total_edp) / max(1e-9, abs(current_result.total_edp))
        notes.append(
            f"joint_iter={joint_iter} parent_refine current_edp={current_result.total_edp:.6e} "
            f"refined_edp={refined.total_edp:.6e} rel_improve={rel:.6e}"
        )
        current_result = refined
        if rel <= rel_threshold:
            break

    current_result.hierarchy_notes = notes
    current_result.hierarchy_traces = [trace_map[k] for k in sorted(trace_map.keys())]
    return current_result

def _flat_search_prepared(
    work_blocks: Sequence[Block],
    block_deps: Sequence[tuple[int, int]],
    config: SearchConfig,
    block_flops_override: Sequence[float] | None = None,
    block_outputs_override: Sequence[float] | None = None,
    hierarchy_level: int = 0,
    hierarchy_notes: list[str] | None = None,
    trace_path: str | None = None,
    final_counts_per_block: Sequence[int] | None = None,
    initial_counts_per_block: Sequence[int] | None = None,
    state_count_lower_bounds: Sequence[Sequence[int | None]] | None = None,
    state_count_upper_bounds: Sequence[Sequence[int | None]] | None = None,
    block_unit_latency_override: Sequence[float | None] | None = None,
    block_unit_energy_override: Sequence[float | None] | None = None,
    state_block_latency_override: Sequence[Sequence[float | None]] | None = None,
    state_block_energy_override: Sequence[Sequence[float | None]] | None = None,
    enforce_end_dram_dependency: bool = False,
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

    if final_counts_per_block is not None and len(final_counts_per_block) != len(work_blocks):
        raise ValueError("final_counts_per_block length mismatch")
    if initial_counts_per_block is not None and len(initial_counts_per_block) != len(work_blocks):
        raise ValueError("initial_counts_per_block length mismatch")

    block_map_dims = [list(b.aggregate_map_dims()) for b in work_blocks]
    block_names = [b.name for b in work_blocks]

    n_states, state_order, categories, active_pes, state_active_blocks = _build_state_metadata(
        num_blocks=len(work_blocks),
        num_pes=config.num_pes,
        num_states=config.num_states,
    )

    stage1_candidates: list[_Candidate] = []

    sub_batch_candidates = _effective_sub_batch_candidates(config)
    dep_gap = int(config.dependency_gap)

    _progress(
        config,
        f"flat-search start level={hierarchy_level} path={trace_path or 'root'} blocks={len(work_blocks)} states={n_states} pes={config.num_pes} sub_batch_candidates={sub_batch_candidates}",
    )

    # Stage-1: enumerate BS_sub and optimize ScT (Eq.1-6, Eq.22).
    for sb_idx, sub_batch in enumerate(sub_batch_candidates, start=1):
        if sub_batch <= 0 or config.batch_size % sub_batch != 0:
            _progress(config, f"[Stage-1] skip invalid sub_batch={sub_batch} ({sb_idx}/{len(sub_batch_candidates)})")
            continue

        total_sub_batches = config.batch_size // sub_batch
        _progress(config, f"[Stage-1] solving ScT for sub_batch={sub_batch} ({sb_idx}/{len(sub_batch_candidates)}), total_sub_batches={total_sub_batches}")

        min_batch_if_active = int(config.min_batch_if_active)
        if final_counts_per_block is not None:
            if initial_counts_per_block is None:
                init_counts = [0 for _ in range(len(work_blocks))]
            else:
                init_counts = [int(v) for v in initial_counts_per_block]
            need_total = sum(
                max(0, int(final_counts_per_block[j]) - init_counts[j])
                for j in range(len(work_blocks))
            )
            if need_total <= 0:
                min_active = 0
                min_batch_if_active = 0
            else:
                min_active = max(1, min(int(config.min_active_states), n_states))
        else:
            min_active = max(1, min(int(config.min_active_states), n_states))

        caps = _state_caps(total_sub_batches, n_states, config.max_state_share)
        if caps is not None:
            if not _caps_cover_pipeline_windows(caps, total_sub_batches, len(work_blocks), n_states):
                caps = None

        try:
            sct_opt = optimize_sct_table(
                block_flops=block_flops,
                block_outputs=block_outputs,
                block_map_dims=block_map_dims,
                block_unit_latency_override=block_unit_latency_override,
                block_unit_energy_override=block_unit_energy_override,
                state_block_latency_override=state_block_latency_override,
                state_block_energy_override=state_block_energy_override,
                total_sub_batches=total_sub_batches,
                block_dependencies=block_deps,
                num_pes=config.num_pes,
                weight_latency=config.weight_latency,
                weight_energy=config.weight_energy,
                num_states=n_states,
                min_active_states=min_active,
                min_batch_if_active=min_batch_if_active,
                max_batches_per_state=caps,
                use_edp_objective=config.use_edp_objective,
                dependency_gap=dep_gap,
                compute_power_per_tile=config.compute_power_per_tile,
                energy_per_op=config.compute_energy_per_op,
                allow_fallback=config.allow_solver_fallback,
                final_counts_per_block=final_counts_per_block,
                initial_counts_per_block=initial_counts_per_block,
                state_count_lower_bounds=state_count_lower_bounds,
                state_count_upper_bounds=state_count_upper_bounds,
            )
        except Exception as exc:
            _progress(config, f"[Stage-1] ScT infeasible/failed for sub_batch={sub_batch}: {exc}")
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

    _progress(config, f"[Stage-1] feasible ScT candidates={len(stage1_candidates)}")
    if not stage1_candidates:
        raise RuntimeError("No feasible ScT candidate found. Check batch and constraints.")

    stage1_candidates.sort(key=lambda x: x.sct_opt.objective)
    if config.strict_paper_mode:
        k1 = _topk_by_ratio_or_count(
            total=len(stage1_candidates),
            ratio=float(config.top_k1_ratio),
            fallback_count=int(config.top_k1),
        )
    else:
        k1 = max(1, min(len(stage1_candidates), int(config.top_k1)))
    stage1_candidates = stage1_candidates[:k1]
    _progress(config, f"[Stage-1] keep top-K1={k1} candidates")

    # Stage-2: optimize MeT (Eq.7-12, Eq.23) on top-K1.
    stage2_candidates: list[_Candidate] = []
    for cidx, cand in enumerate(stage1_candidates, start=1):
        _progress(config, f"[Stage-2] solving MeT for candidate {cidx}/{len(stage1_candidates)} sub_batch={cand.sub_batch}")
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
                enforce_end_dram_dependency=enforce_end_dram_dependency,
            )
        except Exception as exc:
            _progress(config, f"[Stage-2] MeT infeasible/failed for sub_batch={cand.sub_batch}: {exc}")
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

    _progress(config, f"[Stage-2] feasible MeT candidates={len(stage2_candidates)}")
    if not stage2_candidates:
        raise RuntimeError("No feasible MeT candidate found after ScT optimization.")

    stage2_candidates.sort(key=lambda x: x.met_opt.objective if x.met_opt is not None else float("inf"))
    if config.strict_paper_mode:
        k2 = _topk_by_ratio_or_count(
            total=len(stage2_candidates),
            ratio=float(config.top_k2_ratio),
            fallback_count=int(config.top_k2),
        )
    else:
        k2 = max(1, min(len(stage2_candidates), int(config.top_k2)))
    stage2_candidates = stage2_candidates[:k2]
    _progress(config, f"[Stage-2] keep top-K2={k2} candidates")

    # Final choose by total EDP.
    best_cand = min(
        stage2_candidates,
        key=lambda x: edp(
            _combine_total_latency(x.compute_latency, x.memory_latency, config.latency_combine_mode),
            x.compute_energy + x.memory_energy,
        ),
    )

    best_total_latency = _combine_total_latency(best_cand.compute_latency, best_cand.memory_latency, config.latency_combine_mode)
    best_total_energy = best_cand.compute_energy + best_cand.memory_energy
    _progress(config, f"[Final] best sub_batch={best_cand.sub_batch} total_latency={best_total_latency:.6e} total_energy={best_total_energy:.6e} total_edp={edp(best_total_latency, best_total_energy):.6e}")

    result = _build_result_from_candidate(
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

    depth_budget = max(0, int(config.max_hierarchy_depth) - int(hierarchy_level) - 1)
    if bool(config.derive_recursive_traces) and depth_budget > 0:
        trace_lineage = [] if not trace_path or trace_path == "root" else [p for p in str(trace_path).split("/") if p and p != "root"]
        extra_traces, extra_notes = _derive_recursive_intra_block_traces(
            parent_blocks=work_blocks,
            parent_result=result,
            config=config,
            depth_remaining=depth_budget,
            lineage=trace_lineage,
        )
        if extra_traces:
            result.hierarchy_traces = list(result.hierarchy_traces) + extra_traces
        if extra_notes:
            result.hierarchy_notes = list(result.hierarchy_notes) + extra_notes

    return result

def _hierarchical_search(
    blocks: Sequence[Block],
    config: SearchConfig,
    depth: int,
    lineage: list[str],
) -> SearchResult:
    work_blocks, block_deps = _prepare_blocks_and_dependencies(blocks, config)
    notes: list[str] = []
    trace_map: dict[str, dict[str, object]] = {}

    base_path = "root" if not lineage else "root/" + "/".join(lineage)
    _progress(
        config,
        f"[Hier] start path={base_path} depth={depth} blocks={len(work_blocks)} deps={len(block_deps)}",
    )

    current_result = _recursive_joint_optimize_prepared(
        work_blocks=work_blocks,
        block_deps=block_deps,
        config=config,
        depth_remaining=depth,
        hierarchy_level=len(lineage),
        hierarchy_notes=notes,
        trace_path=base_path,
    )
    for tr in current_result.hierarchy_traces:
        trace_map[str(tr.get("path", base_path))] = tr

    max_iters = max(1, int(config.max_hierarchy_iters))
    rel_threshold = max(float(config.hierarchy_theta), float(config.structure_refine_min_improvement))

    if depth <= 1 or not bool(config.enable_structure_refinement):
        notes.append(f"level={len(lineage)} structure_refinement_disabled_or_depth_limit")
    else:
        for iter_idx in range(max_iters):
            expansions = _candidate_child_expansions(
                blocks=work_blocks,
                min_children=int(config.hierarchy_min_child_blocks),
                max_trials=int(config.structure_refine_max_trials),
            )
            if not expansions:
                notes.append(f"level={len(lineage)} iter={iter_idx} no_expandable_block")
                break

            _progress(
                config,
                f"[Hier] iter={iter_idx + 1}/{max_iters} evaluating {len(expansions)} structure candidates",
            )

            best_trial_result: SearchResult | None = None
            best_trial_blocks: list[Block] | None = None
            best_trial_deps: list[tuple[int, int]] | None = None
            best_trial_block_name = ""

            for trial_idx, (bi, children) in enumerate(expansions, start=1):
                parent_name = work_blocks[bi].name
                cand_blocks = _replace_block_with_children(work_blocks, bi, children)
                cand_deps = _derive_or_default_dependencies(cand_blocks)
                cand_path = f"{base_path}/iter{iter_idx + 1}/expand_{trial_idx}_{parent_name}"

                _progress(
                    config,
                    f"[Hier] trial={trial_idx}/{len(expansions)} expand block={parent_name} -> {len(children)} children",
                )

                try:
                    cand_result = _recursive_joint_optimize_prepared(
                        work_blocks=cand_blocks,
                        block_deps=cand_deps,
                        config=config,
                        depth_remaining=depth,
                        hierarchy_level=len(lineage),
                        hierarchy_notes=notes,
                        trace_path=cand_path,
                    )
                except Exception as exc:
                    notes.append(
                        f"level={len(lineage)} iter={iter_idx} trial={trial_idx} block={parent_name} infeasible={exc}"
                    )
                    _progress(config, f"[Hier] trial failed block={parent_name}: {exc}")
                    continue

                for tr in cand_result.hierarchy_traces:
                    trace_map[str(tr.get("path", cand_path))] = tr

                rel = (current_result.total_edp - cand_result.total_edp) / max(1e-9, abs(current_result.total_edp))
                notes.append(
                    f"level={len(lineage)} iter={iter_idx} trial={trial_idx} block={parent_name} "
                    f"current_edp={current_result.total_edp:.6e} cand_edp={cand_result.total_edp:.6e} rel_improve={rel:.6e}"
                )

                if best_trial_result is None or cand_result.total_edp < best_trial_result.total_edp:
                    best_trial_result = cand_result
                    best_trial_blocks = cand_blocks
                    best_trial_deps = cand_deps
                    best_trial_block_name = parent_name

            if best_trial_result is None or best_trial_blocks is None or best_trial_deps is None:
                notes.append(f"level={len(lineage)} iter={iter_idx} all_structure_trials_failed")
                break

            best_rel = (current_result.total_edp - best_trial_result.total_edp) / max(
                1e-9, abs(current_result.total_edp)
            )
            if best_rel <= rel_threshold:
                notes.append(
                    f"level={len(lineage)} iter={iter_idx} no_significant_improvement "
                    f"best_rel={best_rel:.6e} threshold={rel_threshold:.6e}"
                )
                _progress(
                    config,
                    f"[Hier] stop iter={iter_idx + 1}, best_rel={best_rel:.6e} <= threshold={rel_threshold:.6e}",
                )
                break

            work_blocks = best_trial_blocks
            block_deps = best_trial_deps
            current_result = best_trial_result
            notes.append(
                f"level={len(lineage)} iter={iter_idx} accept_expand block={best_trial_block_name} "
                f"new_blocks={len(work_blocks)} new_edp={current_result.total_edp:.6e}"
            )
            _progress(
                config,
                f"[Hier] accept iter={iter_idx + 1} expand={best_trial_block_name} "
                f"new_blocks={len(work_blocks)} edp={current_result.total_edp:.6e}",
            )

    current_result.hierarchy_notes = notes
    current_result.hierarchy_level = len(lineage)
    current_result.hierarchy_traces = [trace_map[k] for k in sorted(trace_map.keys())]
    return current_result


def _reverse_dependencies_for_backward(
    num_blocks: int,
    forward_dependencies: Sequence[tuple[int, int]],
) -> list[tuple[int, int]]:
    # Forward dep p->c becomes backward dep c->p under reversed block order.
    out: set[tuple[int, int]] = set()
    for p, c in forward_dependencies:
        if p < 0 or c < 0 or p >= num_blocks or c >= num_blocks or p == c:
            continue
        bp = num_blocks - 1 - c
        bc = num_blocks - 1 - p
        if bp != bc:
            out.add((bp, bc))
    if not out and num_blocks > 1:
        return _default_linear_dependencies(num_blocks)
    return sorted(out)


def _phase_payload(res: SearchResult) -> dict[str, object]:
    return {
        "best_sub_batch": int(res.best_sub_batch),
        "scheduled_blocks": list(res.scheduled_blocks),
        "block_dependencies": list(res.block_dependencies),
        "state_order": list(res.state_order),
        "state_categories": list(res.state_categories),
        "state_batches": list(res.milp_solution.state_batches),
        "sct": res.sct.table.tolist(),
        "met_s": res.met.sram.tolist(),
        "met_d": res.met.dram.tolist(),
        "compute_latency": float(res.compute_latency),
        "compute_energy": float(res.compute_energy),
        "memory_latency": float(res.memory_latency),
        "memory_energy": float(res.memory_energy),
        "total_latency": float(res.total_latency),
        "total_energy": float(res.total_energy),
        "total_edp": float(res.total_edp),
    }



def _training_retention_candidates(total_sub_batches: int) -> list[int]:
    total = max(0, int(total_sub_batches))
    if total <= 8:
        return list(range(0, total + 1))

    vals = {
        0,
        1,
        max(0, total // 4),
        max(0, total // 2),
        max(0, (3 * total) // 4),
        max(0, total - 1),
        total,
    }
    return sorted(v for v in vals if 0 <= v <= total)


def _with_memory_override(
    res: SearchResult,
    met_opt: MemoryOptimizationResult,
    memory_latency: float,
    memory_energy: float,
    latency_combine_mode: str,
) -> SearchResult:
    total_latency = _combine_total_latency(res.compute_latency, memory_latency, latency_combine_mode)
    total_energy = res.compute_energy + memory_energy
    return replace(
        res,
        met=met_opt.table,
        met_solver_name=met_opt.solver_name,
        memory_latency=float(memory_latency),
        memory_energy=float(memory_energy),
        total_latency=float(total_latency),
        total_energy=float(total_energy),
        total_edp=float(edp(total_latency, total_energy)),
    )

def _bw1_initial_counts_from_forward_end_md(end_md: Sequence[int]) -> list[int]:
    """Eq.14-style prefix counts for backward-only preprocessing."""

    rev = [int(v) for v in reversed(end_md)]
    init = [0 for _ in rev]
    for j in range(1, len(rev)):
        init[j] = max(0, rev[j - 1] - rev[j])
    return init


def _training_end_md_profiles(total_sub_batches: int, n_blocks: int) -> list[tuple[str, list[int]]]:
    base = sorted({max(0, total_sub_batches - keep) for keep in _training_retention_candidates(total_sub_batches)}, reverse=True)
    profiles: list[tuple[str, list[int]]] = []
    seen: set[tuple[int, ...]] = set()

    def add_profile(tag: str, vals: list[int]) -> None:
        key = tuple(max(0, min(total_sub_batches, int(v))) for v in vals)
        if key in seen:
            return
        seen.add(key)
        profiles.append((tag, list(key)))

    half = max(1, n_blocks // 2)
    q3 = max(1, (3 * n_blocks) // 4)
    for md in base:
        add_profile(f"uniform_md{md}", [md for _ in range(n_blocks)])
        if md > 0:
            add_profile(f"halfsplit_md{md}_{max(0, md - 1)}", [md for _ in range(half)] + [max(0, md - 1) for _ in range(n_blocks - half)])
        if md > 1:
            add_profile(f"tailkeep_md{md}_{max(0, md - 2)}", [md for _ in range(q3)] + [max(0, md - 2) for _ in range(n_blocks - q3)])

    return profiles


def _empty_state_bounds(num_states: int, num_blocks: int) -> list[list[int | None]]:
    return [[None for _ in range(num_blocks)] for _ in range(num_states)]


def _bw1_eq14_state_bounds_from_forward_end_md(
    end_md: Sequence[int],
    total_sub_batches: int,
) -> tuple[list[list[int | None]], list[list[int | None]], list[int], list[int]]:
    n = len(end_md)
    num_states = 2 * n - 1
    lower = _empty_state_bounds(num_states, n)
    upper = _empty_state_bounds(num_states, n)

    stored_counts = [max(0, int(total_sub_batches) - int(v)) for v in end_md]
    final_counts = [stored_counts[n - 1 - j] for j in range(n)]
    init_counts = _bw1_initial_counts_from_forward_end_md(end_md)

    for j in range(n):
        init_j = int(init_counts[j])
        final_j = int(final_counts[j])
        for i in range(0, j):
            lower[i][j] = init_j
            upper[i][j] = init_j
        finish = n + j - 1
        for i in range(finish, num_states):
            lower[i][j] = final_j
            upper[i][j] = final_j

    return lower, upper, init_counts, final_counts


def _bw2_eq15_state_bounds_from_forward_end_md(
    end_md: Sequence[int],
) -> tuple[list[list[int | None]], list[list[int | None]], list[int]]:
    n = len(end_md)
    num_blocks = 2 * n
    num_states = 2 * num_blocks - 1
    lower = _empty_state_bounds(num_states, num_blocks)
    upper = _empty_state_bounds(num_states, num_blocks)

    discarded_counts = [max(0, int(v)) for v in end_md]
    final_counts = list(discarded_counts) + [discarded_counts[n - 1 - j] for j in range(n)]

    for j in range(num_blocks):
        final_j = int(final_counts[j])
        finish = num_blocks + j - 1
        for i in range(finish, num_states):
            lower[i][j] = final_j
            upper[i][j] = final_j

    return lower, upper, final_counts


def _search_training_with_recomputation(
    blocks: Sequence[Block],
    config: SearchConfig,
) -> SearchResult:
    work_blocks, block_deps = _prepare_blocks_and_dependencies(blocks, config)
    n = len(work_blocks)
    if n == 0:
        raise ValueError("blocks must not be empty")

    base_flops = [max(1e-9, float(b.total_flops())) for b in work_blocks]
    base_outputs = [max(1e-9, float(b.total_output_size())) for b in work_blocks]

    bw_blocks = [_clone_block_tree(b) for b in reversed(work_blocks)]
    for i, b in enumerate(bw_blocks):
        b.name = f"bw::{i:03d}::{b.name}"
    bw_deps = _reverse_dependencies_for_backward(n, block_deps)
    bw_flops = [base_flops[n - 1 - i] * max(1e-9, float(config.backward_compute_scale)) for i in range(n)]
    bw_outputs = [base_outputs[n - 1 - i] * max(1e-9, float(config.backward_output_scale)) for i in range(n)]

    rc_blocks = [_clone_block_tree(b) for b in work_blocks]
    for i, b in enumerate(rc_blocks):
        b.name = f"rc::{i:03d}::{b.name}"

    bw2_back_blocks = [_clone_block_tree(b) for b in reversed(work_blocks)]
    for i, b in enumerate(bw2_back_blocks):
        b.name = f"bw2::{i:03d}::{b.name}"

    bw2_blocks = rc_blocks + bw2_back_blocks
    bw2_deps: set[tuple[int, int]] = set()
    for p, c in block_deps:
        if 0 <= p < n and 0 <= c < n and p != c:
            bw2_deps.add((p, c))
    for p, c in bw_deps:
        bw2_deps.add((n + p, n + c))
    for j in range(n):
        bw_j = n + (n - 1 - j)
        bw2_deps.add((j, bw_j))
    bw2_dep_list = sorted(bw2_deps)
    if not bw2_dep_list and len(bw2_blocks) > 1:
        bw2_dep_list = _default_linear_dependencies(len(bw2_blocks))

    rc_flops = [f * max(1e-9, float(config.recompute_compute_scale)) for f in base_flops]
    rc_outputs = [v * max(1e-9, float(config.recompute_output_scale)) for v in base_outputs]
    bw2_flops_base = rc_flops + bw_flops
    bw2_outputs_base = rc_outputs + bw_outputs

    best_result: SearchResult | None = None
    sub_batch_candidates = _effective_sub_batch_candidates(config)

    _progress(
        config,
        f"[Training] start blocks={n} deps={len(block_deps)} candidates={sub_batch_candidates}",
    )

    for sb_idx, sub_batch in enumerate(sub_batch_candidates, start=1):
        if sub_batch <= 0 or config.batch_size % sub_batch != 0:
            _progress(
                config,
                f"[Training] skip invalid sub_batch={sub_batch} ({sb_idx}/{len(sub_batch_candidates)})",
            )
            continue

        total_sub = int(config.batch_size // sub_batch)
        phase_cfg = replace(
            config,
            candidate_sub_batches=[int(sub_batch)],
            use_all_sub_batch_factors=False,
            strict_paper_mode=False,
            top_k1=1,
            top_k2=1,
            top_k1_ratio=0.0,
            top_k2_ratio=0.0,
            enable_hierarchical_pipeline=False,
        )

        _progress(
            config,
            f"[Training] sub_batch={sub_batch} ({sb_idx}/{len(sub_batch_candidates)}): solve FW ScT",
        )

        try:
            fw_res_base = _flat_search_prepared(
                work_blocks=work_blocks,
                block_deps=block_deps,
                config=phase_cfg,
                block_flops_override=base_flops,
                block_outputs_override=base_outputs,
                hierarchy_level=0,
                hierarchy_notes=[],
                trace_path=f"train/sb{sub_batch}/fw_base",
                enforce_end_dram_dependency=True,
            )
        except Exception as exc:
            _progress(config, f"[Training] FW failed for sub_batch={sub_batch}: {exc}")
            continue

        end_md_profiles = _training_end_md_profiles(total_sub, n)
        seen_end_md: set[tuple[int, ...]] = set()
        candidate_logs: list[str] = []

        for retain_idx, (retain_tag, desired_md_ub) in enumerate(end_md_profiles, start=1):
            _progress(
                config,
                f"[Training] sub_batch={sub_batch}: profile {retain_idx}/{len(end_md_profiles)} {retain_tag} target_end_md={desired_md_ub}",
            )

            try:
                fw_met_opt = optimize_memory_table(
                    sct=fw_res_base.sct,
                    block_volumes=base_outputs,
                    block_dependencies=block_deps,
                    sram_capacity=phase_cfg.sram_capacity,
                    dram_capacity=phase_cfg.dram_capacity,
                    heuristic_sram_keep_ratio=0.6,
                    noc_bandwidth=phase_cfg.noc_bandwidth,
                    dram_bandwidth=phase_cfg.dram_bandwidth,
                    noc_energy_per_unit=phase_cfg.noc_energy_per_unit,
                    dram_energy_per_unit=phase_cfg.dram_energy_per_unit,
                    dram_noc_hops=phase_cfg.dram_noc_hops,
                    weight_latency=phase_cfg.weight_latency,
                    weight_energy=phase_cfg.weight_energy,
                    use_edp_objective=phase_cfg.use_edp_objective,
                    allow_fallback=False,
                    enforce_end_dram_dependency=True,
                    end_dram_upper_bounds=desired_md_ub,
                    force_final_sram_empty=True,
                )
            except Exception as exc:
                candidate_logs.append(
                    f"candidate sub_batch={sub_batch} profile={retain_tag} status=fw_met_failed reason={exc}"
                )
                continue

            fw_memory_latency, fw_memory_energy = _estimate_memory_cost(
                sct=fw_res_base.sct,
                met=fw_met_opt.table,
                block_volumes=base_outputs,
                block_dependencies=block_deps,
                noc_bandwidth=phase_cfg.noc_bandwidth,
                dram_bandwidth=phase_cfg.dram_bandwidth,
                noc_energy_per_unit=phase_cfg.noc_energy_per_unit,
                dram_energy_per_unit=phase_cfg.dram_energy_per_unit,
                dram_noc_hops=phase_cfg.dram_noc_hops,
                noc_hops_compute=phase_cfg.noc_hops_compute,
                noc_hops_sram=phase_cfg.noc_hops_sram,
            )
            fw_res = _with_memory_override(
                res=fw_res_base,
                met_opt=fw_met_opt,
                memory_latency=fw_memory_latency,
                memory_energy=fw_memory_energy,
                latency_combine_mode=phase_cfg.latency_combine_mode,
            )

            end_md_raw = list(fw_res.met.dram[-1, :]) if fw_res.met.dram.size > 0 else [0.0 for _ in range(n)]
            end_md = tuple(max(0, min(total_sub, int(round(v)))) for v in end_md_raw)
            if end_md in seen_end_md:
                continue
            seen_end_md.add(end_md)
            bw1_lb, bw1_ub, bw1_init, bw1_final = _bw1_eq14_state_bounds_from_forward_end_md(
                end_md=end_md,
                total_sub_batches=total_sub,
            )
            bw2_lb, bw2_ub, bw2_final = _bw2_eq15_state_bounds_from_forward_end_md(end_md)

            _progress(config, f"[Training] sub_batch={sub_batch}: solve BW1 profile={retain_tag} end_md={list(end_md)}")
            try:
                bw1_res = _flat_search_prepared(
                    work_blocks=bw_blocks,
                    block_deps=bw_deps,
                    config=phase_cfg,
                    block_flops_override=bw_flops,
                    block_outputs_override=bw_outputs,
                    hierarchy_level=0,
                    hierarchy_notes=[],
                    trace_path=f"train/sb{sub_batch}/bw1_{retain_tag}",
                    final_counts_per_block=bw1_final,
                    initial_counts_per_block=bw1_init,
                    state_count_lower_bounds=bw1_lb,
                    state_count_upper_bounds=bw1_ub,
                )
            except Exception as exc:
                candidate_logs.append(
                    f"candidate sub_batch={sub_batch} profile={retain_tag} status=bw1_failed end_md={list(end_md)} reason={exc}"
                )
                continue

            _progress(config, f"[Training] sub_batch={sub_batch}: solve BW2 profile={retain_tag} end_md={list(end_md)}")
            try:
                bw2_res = _flat_search_prepared(
                    work_blocks=bw2_blocks,
                    block_deps=bw2_dep_list,
                    config=phase_cfg,
                    block_flops_override=bw2_flops_base,
                    block_outputs_override=bw2_outputs_base,
                    hierarchy_level=0,
                    hierarchy_notes=[],
                    trace_path=f"train/sb{sub_batch}/bw2_{retain_tag}",
                    final_counts_per_block=bw2_final,
                    initial_counts_per_block=[0 for _ in range(len(bw2_blocks))],
                    state_count_lower_bounds=bw2_lb,
                    state_count_upper_bounds=bw2_ub,
                )
            except Exception as exc:
                candidate_logs.append(
                    f"candidate sub_batch={sub_batch} profile={retain_tag} status=bw2_failed end_md={list(end_md)} reason={exc}"
                )
                continue

            total_compute_latency = fw_res.compute_latency + bw1_res.compute_latency + bw2_res.compute_latency
            total_compute_energy = fw_res.compute_energy + bw1_res.compute_energy + bw2_res.compute_energy
            total_memory_latency = fw_res.memory_latency + bw1_res.memory_latency + bw2_res.memory_latency
            total_memory_energy = fw_res.memory_energy + bw1_res.memory_energy + bw2_res.memory_energy

            total_latency = fw_res.total_latency + bw1_res.total_latency + bw2_res.total_latency
            total_energy = fw_res.total_energy + bw1_res.total_energy + bw2_res.total_energy
            total_edp = edp(total_latency, total_energy)

            notes = [
                f"training_mode=True sub_batch={sub_batch}",
                f"retain_profile={retain_tag}",
                "explicit_bw_constraints=eq14_eq15",
                f"fw_end_met_d={list(end_md)}",
                f"bw1_final_counts={bw1_final}",
                f"bw2_final_counts={bw2_final}",
            ] + list(candidate_logs)

            phase_results = {
                "fw": _phase_payload(fw_res),
                "bw1": _phase_payload(bw1_res),
                "bw2": _phase_payload(bw2_res),
            }

            merged = replace(
                fw_res,
                best_sub_batch=int(sub_batch),
                compute_latency=float(total_compute_latency),
                compute_energy=float(total_compute_energy),
                memory_latency=float(total_memory_latency),
                memory_energy=float(total_memory_energy),
                total_latency=float(total_latency),
                total_energy=float(total_energy),
                total_edp=float(total_edp),
                hierarchy_notes=notes,
                phase_results=phase_results,
            )

            candidate_logs.append(
                f"candidate sub_batch={sub_batch} profile={retain_tag} status=ok end_md={list(end_md)} total_edp={total_edp:.6e}"
            )
            _progress(
                config,
                f"[Training] sub_batch={sub_batch} profile={retain_tag} total_latency={total_latency:.6e} total_energy={total_energy:.6e} total_edp={total_edp:.6e}",
            )

            if best_result is None or merged.total_edp < best_result.total_edp:
                best_result = merged

    if best_result is None:
        raise RuntimeError("No feasible training candidate found.")

    _progress(
        config,
        f"[Training] best sub_batch={best_result.best_sub_batch} total_edp={best_result.total_edp:.6e}",
    )
    return best_result


def search_schedule(
    blocks: Sequence[Block],
    config: SearchConfig,
    solver: object | None = None,
) -> SearchResult:
    del solver  # kept for compatibility with older entry points

    if not blocks:
        raise ValueError("blocks must not be empty")

    if bool(config.enable_training_recomputation):
        return _search_training_with_recomputation(blocks=blocks, config=config)

    use_hier = bool(config.enable_hierarchical_pipeline) and int(config.max_hierarchy_depth) > 1
    if use_hier:
        return _hierarchical_search(
            blocks=blocks,
            config=config,
            depth=max(1, int(config.max_hierarchy_depth)),
            lineage=[],
        )

    work_blocks, block_deps = _prepare_blocks_and_dependencies(blocks, config)
    use_joint = bool(config.derive_recursive_traces) and int(config.max_hierarchy_depth) > 1
    if use_joint:
        return _recursive_joint_optimize_prepared(
            work_blocks=work_blocks,
            block_deps=block_deps,
            config=config,
            depth_remaining=max(1, int(config.max_hierarchy_depth)),
            hierarchy_level=0,
            hierarchy_notes=[],
            trace_path="root",
        )
    return _flat_search_prepared(
        work_blocks=work_blocks,
        block_deps=block_deps,
        config=config,
        hierarchy_level=0,
        hierarchy_notes=[],
        trace_path="root",
    )






















