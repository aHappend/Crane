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
from search.scheduler_search import SearchConfig, SearchResult, search_schedule


# Aligned with outputs/visualization/official_nns_structure.html.
TRANSFORMER_STAGE_LAYOUT: list[tuple[str, int]] = [
    ("word_embed_enc", 1),
    ("6x encoder", 162),
    ("word_embed_dec", 1),
    ("6x decoder", 306),
    ("proj", 1),
]


@dataclass
class Candidate:
    mode: str
    max_layers_per_block: int
    result: SearchResult
    input_blocks: int


def _build_stage_blocks_from_min_layers(min_layer_blocks: Sequence[Block]) -> list[Block]:
    layers = []
    for blk in min_layer_blocks:
        if len(blk.layers) != 1:
            raise RuntimeError("expected one layer per minimal block")
        layers.append(blk.layers[0])

    blocks: list[Block] = []
    cursor = 0
    for stage_name, cnt in TRANSFORMER_STAGE_LAYOUT:
        part = layers[cursor : cursor + cnt]
        if len(part) != cnt:
            raise RuntimeError(f"unexpected layer count when building stage {stage_name}")
        cursor += cnt
        blocks.append(Block(name=stage_name, layers=list(part)))

    if cursor != len(layers):
        raise RuntimeError("stage layout does not consume all minimal layers")
    return blocks


def _cfg(num_pes: int, max_layers_per_block: int) -> SearchConfig:
    return SearchConfig(
        batch_size=128,
        candidate_sub_batches=[4, 8, 16, 32],
        sram_capacity=15000.0,
        dram_capacity=30000.0,
        num_pes=num_pes,
        enable_chain_block_merge=True,
        max_layers_per_block=max_layers_per_block,
        min_layers_per_block=2,
        min_active_states=1,
        min_batch_if_active=1,
        max_state_share=1.0,
        strict_paper_mode=True,
        top_k1=4,
        top_k2=2,
        use_edp_objective=True,
        dependency_gap=0,
        allow_solver_fallback=False,
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


def _write_detail(path: Path, cand: Candidate) -> None:
    res = cand.result
    sct = res.sct.table
    delta = _delta_from_cumulative(sct)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("network=transformer\n")
        f.write("source_ref=src/nns/transformer.cpp\n")
        f.write(f"mode={cand.mode}\n")
        f.write("stage_definition=official_nns_structure_html\n")
        f.write("block_merge_enabled=True\n")
        f.write(f"max_layers_per_block={cand.max_layers_per_block}\n")
        f.write(f"input_blocks={cand.input_blocks}\n")
        f.write(f"scheduled_blocks={len(res.scheduled_blocks)}\n")
        f.write(f"num_states={len(res.state_order)}\n")
        f.write(f"best_sub_batch={res.best_sub_batch}\n")
        f.write(f"sct_solver={res.sct_solver_name}\n")
        f.write(f"met_solver={res.met_solver_name}\n")
        f.write(f"latency={res.total_latency:.6f}\n")
        f.write(f"energy={res.total_energy:.6f}\n")
        f.write(f"edp={res.total_edp:.6f}\n\n")

        f.write("scheduled_blocks\n")
        for i, name in enumerate(res.scheduled_blocks):
            f.write(f"  {i}: {name}\n")

        f.write("\nstates\n")
        for i, (sname, cat, ape, sb, act) in enumerate(
            zip(
                res.state_order,
                res.state_categories,
                res.active_pes,
                res.milp_solution.state_batches,
                res.state_active_blocks,
            )
        ):
            act_names = [res.scheduled_blocks[j] for j in act if 0 <= j < len(res.scheduled_blocks)]
            if len(act_names) > 8:
                act_view = act_names[:3] + ["..."] + act_names[-2:]
            else:
                act_view = act_names
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


def _run_candidates(mode: str, blocks: list[Block], num_pes: int, caps: Sequence[int]) -> list[Candidate]:
    out: list[Candidate] = []
    for cap in caps:
        cfg = _cfg(num_pes=num_pes, max_layers_per_block=int(cap))
        res = search_schedule(blocks, cfg)
        out.append(
            Candidate(
                mode=mode,
                max_layers_per_block=int(cap),
                result=res,
                input_blocks=len(blocks),
            )
        )
    out.sort(key=lambda x: x.result.total_edp)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Transformer stage-vs-layer compare with block merge enabled")
    parser.add_argument("--num-pes", type=int, default=16)
    args = parser.parse_args()

    if args.num_pes <= 0:
        raise ValueError("--num-pes must be > 0")

    min_layers = build_transformer_min_layers()
    stage_blocks = _build_stage_blocks_from_min_layers(min_layers)

    # Sweep merge granularity to select best result for each mode.
    candidate_caps = [4, 8, 12, 16, 24, 32, 48, 64]

    stage_cands = _run_candidates("stage_level", stage_blocks, args.num_pes, candidate_caps)
    layer_cands = _run_candidates("layer_level", min_layers, args.num_pes, candidate_caps)

    best_stage = stage_cands[0]
    best_layer = layer_cands[0]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "outputs" / "experiments" / f"transformer_stage_layer_merge_pe{args.num_pes}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = out_dir / "candidates.csv"
    summary_txt = out_dir / "summary.txt"
    stage_detail = out_dir / "best_stage_detail.txt"
    layer_detail = out_dir / "best_layer_detail.txt"

    with summary_csv.open("w", encoding="utf-8", newline="\n") as f:
        fields = [
            "mode",
            "max_layers_per_block",
            "input_blocks",
            "scheduled_blocks",
            "num_states",
            "best_sub_batch",
            "latency",
            "energy",
            "edp",
            "sct_solver",
            "met_solver",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for cand in stage_cands + layer_cands:
            r = cand.result
            w.writerow(
                {
                    "mode": cand.mode,
                    "max_layers_per_block": cand.max_layers_per_block,
                    "input_blocks": cand.input_blocks,
                    "scheduled_blocks": len(r.scheduled_blocks),
                    "num_states": len(r.state_order),
                    "best_sub_batch": r.best_sub_batch,
                    "latency": f"{r.total_latency:.6f}",
                    "energy": f"{r.total_energy:.6f}",
                    "edp": f"{r.total_edp:.6f}",
                    "sct_solver": r.sct_solver_name,
                    "met_solver": r.met_solver_name,
                }
            )

    _write_detail(stage_detail, best_stage)
    _write_detail(layer_detail, best_layer)

    ratio = (best_stage.result.total_edp / best_layer.result.total_edp) if best_layer.result.total_edp > 0 else float("inf")
    better = "layer_level" if best_layer.result.total_edp < best_stage.result.total_edp else "stage_level"

    with summary_txt.open("w", encoding="utf-8", newline="\n") as f:
        f.write("experiment=transformer_stage_vs_layer_with_block_merge\n")
        f.write("source_ref=src/nns/transformer.cpp\n")
        f.write("stage_definition=official_nns_structure_html\n")
        f.write("layer_definition=official_min_layers_471\n")
        f.write(f"num_pes={args.num_pes}\n")
        f.write(f"candidate_max_layers_per_block={candidate_caps}\n")
        f.write("block_merge_enabled=True\n")
        f.write("same_constraints=True\n\n")
        f.write("[best_stage]\n")
        f.write(f"  input_blocks={best_stage.input_blocks}\n")
        f.write(f"  max_layers_per_block={best_stage.max_layers_per_block}\n")
        f.write(f"  scheduled_blocks={len(best_stage.result.scheduled_blocks)}\n")
        f.write(f"  num_states={len(best_stage.result.state_order)}\n")
        f.write(f"  best_sub_batch={best_stage.result.best_sub_batch}\n")
        f.write(f"  edp={best_stage.result.total_edp:.6f}\n")
        f.write(f"  detail={stage_detail}\n\n")
        f.write("[best_layer]\n")
        f.write(f"  input_blocks={best_layer.input_blocks}\n")
        f.write(f"  max_layers_per_block={best_layer.max_layers_per_block}\n")
        f.write(f"  scheduled_blocks={len(best_layer.result.scheduled_blocks)}\n")
        f.write(f"  num_states={len(best_layer.result.state_order)}\n")
        f.write(f"  best_sub_batch={best_layer.result.best_sub_batch}\n")
        f.write(f"  edp={best_layer.result.total_edp:.6f}\n")
        f.write(f"  detail={layer_detail}\n\n")
        f.write("[compare]\n")
        f.write(f"  better_mode={better}\n")
        f.write(f"  edp_ratio_stage_over_layer={ratio:.6f}\n")

    print(f"out_dir={out_dir}")
    print(f"summary_txt={summary_txt}")
    print(f"summary_csv={summary_csv}")
    print(f"stage_detail={stage_detail}")
    print(f"layer_detail={layer_detail}")


if __name__ == "__main__":
    main()
