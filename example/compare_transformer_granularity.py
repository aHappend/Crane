from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from example.run_transformer_min_layer_block_experiment import build_transformer_min_layers
from scheduler.block import Block
from scheduler.hardware_profile import paper_7_2_search_params
from search.scheduler_search import SearchConfig, SearchResult, search_schedule


@dataclass
class CaseResult:
    mode: str
    input_blocks: list[Block]
    result: SearchResult


def _partition_sizes(total: int, parts: int) -> list[int]:
    if total <= 0 or parts <= 0:
        raise ValueError("total and parts must be > 0")
    base = total // parts
    rem = total % parts
    sizes = [base for _ in range(parts)]
    for i in range(rem):
        sizes[i] += 1
    return sizes


def build_stage_blocks_from_min_layers(min_layer_blocks: Sequence[Block], num_stages: int = 8) -> list[Block]:
    """Build stage-level blocks by contiguous partition over the same 471 minimal layers."""

    layers = []
    for blk in min_layer_blocks:
        if len(blk.layers) != 1:
            raise RuntimeError("expected one layer per minimal block")
        layers.append(blk.layers[0])

    sizes = _partition_sizes(len(layers), num_stages)
    blocks: list[Block] = []
    cursor = 0
    for stage_idx, size in enumerate(sizes):
        start = cursor
        end = cursor + size
        part = layers[start:end]
        if not part:
            continue
        bname = f"stage_{stage_idx:02d}_{part[0].name}_to_{part[-1].name}"
        blocks.append(Block(name=bname, layers=list(part)))
        cursor = end
    return blocks


def _cfg(
    num_pes: int,
    use_paper_hw_7_2: bool,
    top_k1_ratio: float,
    top_k2_ratio: float,
    all_sub_batch_factors: bool,
    verbose_progress: bool,
    progress_prefix: str,
    hierarchical: bool,
    hier_depth: int,
    hier_iters: int,
    hier_theta: float,
    structure_max_trials: int,
) -> SearchConfig:
    hw_kwargs: dict[str, float] = {}
    if use_paper_hw_7_2:
        hw_kwargs = paper_7_2_search_params(num_pes=num_pes, dram_capacity_mb=30000.0)

    return SearchConfig(
        batch_size=128,
        candidate_sub_batches=[4, 8, 16, 32],
        sram_capacity=float(hw_kwargs.get("sram_capacity", 15000.0)),
        dram_capacity=float(hw_kwargs.get("dram_capacity", 30000.0)),
        num_pes=num_pes,
        enable_chain_block_merge=True,
        max_layers_per_block=16,
        min_layers_per_block=2,
        min_active_states=1,
        min_batch_if_active=1,
        max_state_share=1.0,
        strict_paper_mode=True,
        top_k1=4,
        top_k2=2,
        top_k1_ratio=top_k1_ratio,
        top_k2_ratio=top_k2_ratio,
        use_all_sub_batch_factors=all_sub_batch_factors,
        use_edp_objective=True,
        dependency_gap=0,
        allow_solver_fallback=False,
        verbose_progress=verbose_progress,
        progress_prefix=progress_prefix,
        noc_bandwidth=float(hw_kwargs.get("noc_bandwidth", 4096.0)),
        dram_bandwidth=float(hw_kwargs.get("dram_bandwidth", 4096.0)),
        noc_energy_per_unit=float(hw_kwargs.get("noc_energy_per_unit", 0.0)),
        dram_energy_per_unit=float(hw_kwargs.get("dram_energy_per_unit", 0.0075)),
        dram_noc_hops=float(hw_kwargs.get("dram_noc_hops", 1.0)),
        compute_power_per_tile=float(hw_kwargs.get("compute_power_per_tile", 1.0)),
        compute_energy_per_op=float(hw_kwargs.get("compute_energy_per_op", 1e-12)),
        enable_hierarchical_pipeline=bool(hierarchical),
        max_hierarchy_depth=max(1, int(hier_depth)),
        max_hierarchy_iters=max(1, int(hier_iters)),
        hierarchy_theta=max(0.0, float(hier_theta)),
        enable_structure_refinement=bool(hierarchical),
        structure_refine_max_trials=max(1, int(structure_max_trials)),
    )


def _delta_from_cumulative(cum):
    rows = []
    for i in range(len(cum)):
        if i == 0:
            rows.append(cum[i].copy())
        else:
            rows.append(cum[i] - cum[i - 1])
    return rows


def _fmt_matrix(rows) -> str:
    return "\n".join(",".join(f"{v:.0f}" for v in r) for r in rows)


def _write_detail(path: Path, case: CaseResult) -> None:
    res = case.result
    sct = res.sct.table
    delta = _delta_from_cumulative(sct)

    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("network=transformer_official_same_source\n")
        f.write("source_ref=src/nns/transformer.cpp\n")
        f.write(f"mode={case.mode}\n")
        f.write(f"input_blocks={len(case.input_blocks)}\n")
        f.write(f"scheduled_blocks={len(res.scheduled_blocks)}\n")
        f.write(f"num_states={len(res.state_order)}\n")
        f.write(f"best_sub_batch={res.best_sub_batch}\n")
        f.write(f"sct_solver={res.sct_solver_name}\n")
        f.write(f"met_solver={res.met_solver_name}\n")
        f.write(f"compute_latency={res.compute_latency:.6f}\n")
        f.write(f"compute_energy={res.compute_energy:.6f}\n")
        f.write(f"memory_latency={res.memory_latency:.6f}\n")
        f.write(f"memory_energy={res.memory_energy:.6f}\n")
        f.write(f"latency={res.total_latency:.6f}\n")
        f.write(f"energy={res.total_energy:.6f}\n")
        f.write(f"edp={res.total_edp:.6f}\n\n")

        f.write("states\n")
        for i, (sname, cat, ape, sb, act) in enumerate(
            zip(
                res.state_order,
                res.state_categories,
                res.active_pes,
                res.milp_solution.state_batches,
                res.state_active_blocks,
            )
        ):
            if len(act) <= 10:
                act_view = act
            else:
                act_view = act[:4] + ["..."] + act[-3:]
            f.write(
                f"  idx={i} name={sname} category={cat} active_pe={ape} "
                f"assigned_sub_batches={sb} active_blocks={act_view}\n"
            )

        f.write("\nScT_cumulative(rows=state, cols=block)\n")
        f.write(_fmt_matrix(sct) + "\n")
        f.write("\nScT_delta_per_state(rows=state, cols=block)\n")
        f.write(_fmt_matrix(delta) + "\n")
        f.write("\nMeT_S(rows=state, cols=block)\n")
        f.write(_fmt_matrix(res.met.sram) + "\n")
        f.write("\nMeT_D(rows=state, cols=block)\n")
        f.write(_fmt_matrix(res.met.dram) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare transformer stage vs layer granularity under same constraints")
    parser.add_argument("--num-pes", type=int, default=4)
    parser.add_argument("--paper-hw-7-2", action="store_true", help="use paper Section 7.2 hardware profile")
    parser.add_argument("--top-k1-ratio", type=float, default=0.1)
    parser.add_argument("--top-k2-ratio", type=float, default=0.05)
    parser.add_argument("--all-sub-batch-factors", action="store_true", help="enumerate all factors of batch size for BS_sub")
    parser.add_argument("--verbose-progress", action="store_true", help="print live solver progress from scheduler")
    parser.add_argument("--hierarchical", action="store_true", help="enable hierarchical block-structure refinement")
    parser.add_argument("--hier-depth", type=int, default=2)
    parser.add_argument("--hier-iters", type=int, default=2)
    parser.add_argument("--hier-theta", type=float, default=0.01)
    parser.add_argument("--structure-max-trials", type=int, default=2)
    args = parser.parse_args()

    if args.num_pes <= 0:
        raise ValueError("--num-pes must be > 0")

    min_layer_blocks = build_transformer_min_layers()
    stage_blocks = build_stage_blocks_from_min_layers(min_layer_blocks, num_stages=8)

    stage_cfg = _cfg(
        num_pes=args.num_pes,
        use_paper_hw_7_2=bool(args.paper_hw_7_2),
        top_k1_ratio=float(args.top_k1_ratio),
        top_k2_ratio=float(args.top_k2_ratio),
        all_sub_batch_factors=bool(args.all_sub_batch_factors),
        verbose_progress=bool(args.verbose_progress),
        progress_prefix="stage",
        hierarchical=bool(args.hierarchical),
        hier_depth=int(args.hier_depth),
        hier_iters=int(args.hier_iters),
        hier_theta=float(args.hier_theta),
        structure_max_trials=int(args.structure_max_trials),
    )
    layer_cfg = _cfg(
        num_pes=args.num_pes,
        use_paper_hw_7_2=bool(args.paper_hw_7_2),
        top_k1_ratio=float(args.top_k1_ratio),
        top_k2_ratio=float(args.top_k2_ratio),
        all_sub_batch_factors=bool(args.all_sub_batch_factors),
        verbose_progress=bool(args.verbose_progress),
        progress_prefix="layer",
        hierarchical=bool(args.hierarchical),
        hier_depth=int(args.hier_depth),
        hier_iters=int(args.hier_iters),
        hier_theta=float(args.hier_theta),
        structure_max_trials=int(args.structure_max_trials),
    )

    print(
        "run_config:",
        {
            "num_pes": args.num_pes,
            "paper_hw_7_2": bool(args.paper_hw_7_2),
            "top_k1_ratio": float(args.top_k1_ratio),
            "top_k2_ratio": float(args.top_k2_ratio),
            "all_sub_batch_factors": bool(args.all_sub_batch_factors),
            "verbose_progress": bool(args.verbose_progress),
            "hierarchical": bool(args.hierarchical),
            "hier_depth": int(args.hier_depth),
            "hier_iters": int(args.hier_iters),
            "hier_theta": float(args.hier_theta),
            "structure_max_trials": int(args.structure_max_trials),
        },
    )

    stage_result = search_schedule(stage_blocks, stage_cfg)
    layer_result = search_schedule(min_layer_blocks, layer_cfg)

    cases = [
        CaseResult(mode="stage_level", input_blocks=stage_blocks, result=stage_result),
        CaseResult(mode="layer_level", input_blocks=min_layer_blocks, result=layer_result),
    ]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "outputs" / "experiments" / f"transformer_granularity_compare_pe{args.num_pes}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_txt = out_dir / "summary.txt"
    summary_csv = out_dir / "summary.csv"

    for c in cases:
        _write_detail(out_dir / f"{c.mode}_detail.txt", c)

    rows = []
    for c in cases:
        r = c.result
        rows.append(
            {
                "mode": c.mode,
                "input_blocks": len(c.input_blocks),
                "scheduled_blocks": len(r.scheduled_blocks),
                "num_states": len(r.state_order),
                "best_sub_batch": r.best_sub_batch,
                "state_batches": ",".join(str(x) for x in r.milp_solution.state_batches),
                "latency": f"{r.total_latency:.6f}",
                "energy": f"{r.total_energy:.6f}",
                "edp": f"{r.total_edp:.6f}",
                "sct_solver": r.sct_solver_name,
                "met_solver": r.met_solver_name,
            }
        )

    with summary_csv.open("w", encoding="utf-8", newline="\n") as f:
        fields = [
            "mode",
            "input_blocks",
            "scheduled_blocks",
            "num_states",
            "best_sub_batch",
            "state_batches",
            "latency",
            "energy",
            "edp",
            "sct_solver",
            "met_solver",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    stage_edp = stage_result.total_edp
    layer_edp = layer_result.total_edp
    better = "layer_level" if layer_edp < stage_edp else "stage_level"
    ratio = (stage_edp / layer_edp) if layer_edp > 0 else float("inf")

    with summary_txt.open("w", encoding="utf-8", newline="\n") as f:
        f.write("experiment=transformer_granularity_compare_same_constraints\n")
        f.write("source_ref=src/nns/transformer.cpp\n")
        f.write("same_constraints=True\n")
        f.write(f"config.batch_size={stage_cfg.batch_size}\n")
        f.write(f"config.candidate_sub_batches={list(stage_cfg.candidate_sub_batches)}\n")
        f.write(f"config.sram_capacity={stage_cfg.sram_capacity}\n")
        f.write(f"config.dram_capacity={stage_cfg.dram_capacity}\n")
        f.write(f"config.num_pes={stage_cfg.num_pes}\n")
        f.write(f"config.enable_chain_block_merge={stage_cfg.enable_chain_block_merge}\n")
        f.write(f"config.max_layers_per_block={stage_cfg.max_layers_per_block}\n")
        f.write(f"config.min_layers_per_block={stage_cfg.min_layers_per_block}\n")
        f.write(f"config.min_active_states={stage_cfg.min_active_states}\n")
        f.write(f"config.min_batch_if_active={stage_cfg.min_batch_if_active}\n")
        f.write(f"config.max_state_share={stage_cfg.max_state_share}\n")
        f.write(f"config.strict_paper_mode={stage_cfg.strict_paper_mode}\n")
        f.write(f"config.top_k1={stage_cfg.top_k1}\n")
        f.write(f"config.top_k2={stage_cfg.top_k2}\n")
        f.write(f"config.top_k1_ratio={stage_cfg.top_k1_ratio}\n")
        f.write(f"config.top_k2_ratio={stage_cfg.top_k2_ratio}\n")
        f.write(f"config.use_all_sub_batch_factors={stage_cfg.use_all_sub_batch_factors}\n")
        f.write(f"config.use_edp_objective={stage_cfg.use_edp_objective}\n")
        f.write(f"config.dependency_gap={stage_cfg.dependency_gap}\n")
        f.write(f"config.allow_solver_fallback={stage_cfg.allow_solver_fallback}\n")
        f.write(f"config.paper_hw_7_2={bool(args.paper_hw_7_2)}\n")
        f.write(f"config.hierarchical={bool(args.hierarchical)}\n")
        f.write(f"config.hier_depth={int(args.hier_depth)}\n")
        f.write(f"config.hier_iters={int(args.hier_iters)}\n")
        f.write(f"config.hier_theta={float(args.hier_theta)}\n")
        f.write(f"config.structure_max_trials={int(args.structure_max_trials)}\n")
        f.write(f"config.noc_bandwidth={stage_cfg.noc_bandwidth}\n")
        f.write(f"config.dram_bandwidth={stage_cfg.dram_bandwidth}\n")
        f.write(f"config.noc_energy_per_unit={stage_cfg.noc_energy_per_unit}\n")
        f.write(f"config.dram_energy_per_unit={stage_cfg.dram_energy_per_unit}\n")
        f.write(f"config.compute_power_per_tile={stage_cfg.compute_power_per_tile}\n")
        f.write(f"config.compute_energy_per_op={stage_cfg.compute_energy_per_op}\n")
        f.write(f"raw_layer_count={len(min_layer_blocks)}\n")
        f.write(f"stage_input_blocks={len(stage_blocks)}\n")
        f.write(f"layer_input_blocks={len(min_layer_blocks)}\n")
        f.write(f"stage_edp={stage_edp:.6f}\n")
        f.write(f"layer_edp={layer_edp:.6f}\n")
        f.write(f"better_mode={better}\n")
        f.write(f"edp_ratio_stage_over_layer={ratio:.6f}\n")
        f.write("stage_detail=stage_level_detail.txt\n")
        f.write("layer_detail=layer_level_detail.txt\n")

    print(f"out_dir={out_dir}")
    print(f"summary_txt={summary_txt}")
    print(f"summary_csv={summary_csv}")
    print(f"stage_detail={out_dir / 'stage_level_detail.txt'}")
    print(f"layer_detail={out_dir / 'layer_level_detail.txt'}")


if __name__ == "__main__":
    main()
