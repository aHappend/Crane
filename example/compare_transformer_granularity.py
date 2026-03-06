from __future__ import annotations

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


def _cfg() -> SearchConfig:
    return SearchConfig(
        batch_size=128,
        candidate_sub_batches=[4, 8, 16, 32],
        sram_capacity=15000.0,
        dram_capacity=30000.0,
        num_pes=4,
        enable_chain_block_merge=True,
        max_layers_per_block=16,
        min_layers_per_block=2,
        min_active_states=6,
        min_batch_if_active=1,
        max_state_share=0.40,
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
    min_layer_blocks = build_transformer_min_layers()
    stage_blocks = build_stage_blocks_from_min_layers(min_layer_blocks, num_stages=8)

    stage_cfg = _cfg()
    layer_cfg = _cfg()

    stage_result = search_schedule(stage_blocks, stage_cfg)
    layer_result = search_schedule(min_layer_blocks, layer_cfg)

    cases = [
        CaseResult(mode="stage_level", input_blocks=stage_blocks, result=stage_result),
        CaseResult(mode="layer_level", input_blocks=min_layer_blocks, result=layer_result),
    ]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "outputs" / "experiments" / f"transformer_granularity_compare_{ts}"
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
        f.write("config.batch_size=128\n")
        f.write("config.candidate_sub_batches=[4,8,16,32]\n")
        f.write("config.sram_capacity=15000.0\n")
        f.write("config.dram_capacity=30000.0\n")
        f.write("config.num_pes=4\n")
        f.write("config.enable_chain_block_merge=True\n")
        f.write("config.max_layers_per_block=16\n")
        f.write("config.min_layers_per_block=2\n")
        f.write("config.min_active_states=6\n")
        f.write("config.min_batch_if_active=1\n")
        f.write("config.max_state_share=0.40\n")
        f.write(f"raw_layer_count={len(min_layer_blocks)}\n")
        f.write(f"stage_input_blocks={len(stage_blocks)}\n")
        f.write(f"layer_input_blocks={len(min_layer_blocks)}\n")
        f.write(f"stage_edp={stage_edp:.6f}\n")
        f.write(f"layer_edp={layer_edp:.6f}\n")
        f.write(f"better_mode={better}\n")
        f.write(f"edp_ratio_stage_over_layer={ratio:.6f}\n")
        f.write(f"stage_detail={out_dir / 'stage_level_detail.txt'}\n")
        f.write(f"layer_detail={out_dir / 'layer_level_detail.txt'}\n")

    print(f"out_dir={out_dir}")
    print(f"summary_txt={summary_txt}")
    print(f"summary_csv={summary_csv}")
    print(f"stage_detail={out_dir / 'stage_level_detail.txt'}")
    print(f"layer_detail={out_dir / 'layer_level_detail.txt'}")


if __name__ == "__main__":
    main()
