from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.layer import Layer
from scheduler.block import Block
from search.scheduler_search import SearchConfig, SearchResult, search_schedule
from example.schedule_html import write_schedule_html


@dataclass
class CandidateResult:
    max_layers_per_block: int
    result: SearchResult


def _layer_profile(name: str) -> tuple[float, float, tuple[float, float, float, float], str]:
    lname = name.lower()
    if "groupconv" in lname or "_qk" in lname or "_qkv" in lname:
        return 3.2e9, 18.0, (512.0, 512.0, 8.0, 64.0), "groupconv"
    if "transpose" in lname or "_kt" in lname:
        return 0.6e9, 10.0, (64.0, 512.0, 1.0, 1.0), "transpose"
    if "elt" in lname:
        return 0.4e9, 9.0, (512.0, 512.0, 2.0, 1.0), "eltwise"
    if "feedfwd1" in lname:
        return 6.0e9, 24.0, (512.0, 2048.0, 1.0, 1.0), "conv"
    if "feedfwd2" in lname:
        return 6.0e9, 24.0, (2048.0, 512.0, 1.0, 1.0), "conv"
    if lname.endswith("_q") or re.search(r"_k\d+$", lname) or lname.endswith("_v") or lname.endswith("_fc"):
        return 2.5e9, 16.0, (512.0, 512.0, 1.0, 1.0), "conv"
    if lname.endswith("_k"):
        return 1.0e9, 12.0, (4096.0, 64.0, 1.0, 1.0), "ptp"
    if "embed" in lname:
        return 1.3e9, 20.0, (512.0, 512.0, 1.0, 1.0), "ptp"
    if "proj" in lname:
        return 2.5e9, 16.0, (512.0, 1000.0, 1.0, 1.0), "conv"
    return 1.0e9, 12.0, (128.0, 128.0, 1.0, 1.0), "generic"


def _append_layer(chain: list[Layer], name: str) -> None:
    flops, out, dims, op_type = _layer_profile(name)
    layer = Layer.with_map_dims(name=name, flops=flops, output_size=out, dims=dims, op_type=op_type)
    if chain:
        chain[-1].connect_to(layer)
    chain.append(layer)


def build_transformer_min_layers() -> list[Block]:
    """Expand official transformer.cpp into minimal basic layers (471 total)."""

    layers: list[Layer] = []

    def add_attention(prefix: str) -> None:
        _append_layer(layers, f"{prefix}_Q")
        for g in range(8):
            _append_layer(layers, f"{prefix}_K{g}")
            _append_layer(layers, f"{prefix}_Kt{g}")
        _append_layer(layers, f"{prefix}_K")
        _append_layer(layers, f"{prefix}_V")
        _append_layer(layers, f"{prefix}_QK_GroupConv")
        _append_layer(layers, f"{prefix}_QK_elt")
        _append_layer(layers, f"{prefix}_QKV_GroupConv")
        _append_layer(layers, f"{prefix}_FC")

    def add_encoder(prefix: str) -> None:
        add_attention(prefix)
        _append_layer(layers, f"{prefix}_elt1")
        _append_layer(layers, f"{prefix}_feedfwd1")
        _append_layer(layers, f"{prefix}_feedfwd2")
        _append_layer(layers, f"{prefix}_elt2")

    def add_decoder(prefix: str) -> None:
        add_attention(prefix + "_1")
        _append_layer(layers, f"{prefix}_elt1")
        add_attention(prefix + "_2")
        _append_layer(layers, f"{prefix}_elt2")
        _append_layer(layers, f"{prefix}_feedfwd1")
        _append_layer(layers, f"{prefix}_feedfwd2")
        _append_layer(layers, f"{prefix}_elt3")

    _append_layer(layers, "word_embed_enc")
    for i in range(1, 7):
        add_encoder(f"encoder{i}")

    _append_layer(layers, "word_embed_dec")
    for i in range(1, 7):
        add_decoder(f"decoder{i}")

    _append_layer(layers, "proj")

    blocks: list[Block] = []
    for i, layer in enumerate(layers):
        blocks.append(Block(name=f"L{i:04d}_{layer.name}", layers=[layer]))

    if len(blocks) != 471:
        raise RuntimeError(f"unexpected layer count {len(blocks)} (expected 471)")

    return blocks
def fmt_matrix(rows):
    return "\n".join(",".join(f"{v:.0f}" for v in r) for r in rows)


def run_candidate(blocks: list[Block], max_layers_per_block: int) -> CandidateResult:
    cfg = SearchConfig(
        batch_size=128,
        candidate_sub_batches=[4, 8, 16, 32],
        sram_capacity=15000.0,
        dram_capacity=30000.0,
        num_pes=4,
        enable_chain_block_merge=True,
        max_layers_per_block=max_layers_per_block,
        min_layers_per_block=2,
        min_active_states=6,
        min_batch_if_active=1,
        max_state_share=0.40,
        derive_recursive_traces=True,
        max_hierarchy_depth=2,
    )
    res = search_schedule(blocks, cfg)
    return CandidateResult(max_layers_per_block=max_layers_per_block, result=res)


def write_detail(path: Path, blocks: list[Block], cand: CandidateResult) -> None:
    res = cand.result
    sct = res.sct.table
    delta = []
    for i in range(sct.shape[0]):
        if i == 0:
            delta.append(sct[i].copy())
        else:
            delta.append(sct[i] - sct[i - 1])

    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("network=transformer_official_min_layer\n")
        f.write("source_ref=src/nns/transformer.cpp\n")
        f.write("mode=min_layer_plus_block_merge\n")
        f.write(f"raw_layer_count={len(blocks)}\n")
        f.write(f"max_layers_per_block={cand.max_layers_per_block}\n")
        f.write(f"merged_blocks={len(res.scheduled_blocks)}\n")
        f.write(f"num_states={len(res.state_order)}\n")
        f.write(f"best_sub_batch={res.best_sub_batch}\n")
        f.write(f"sct_solver={res.sct_solver_name}\n")
        f.write(f"met_solver={res.met_solver_name}\n")
        f.write(f"latency={res.total_latency:.6f}\n")
        f.write(f"energy={res.total_energy:.6f}\n")
        f.write(f"edp={res.total_edp:.6f}\n\n")

        f.write("states\n")
        for i, (sname, cat, ape, sb, act) in enumerate(
            zip(res.state_order, res.state_categories, res.active_pes, res.milp_solution.state_batches, res.state_active_blocks)
        ):
            act_names = [res.scheduled_blocks[j] for j in act if 0 <= j < len(res.scheduled_blocks)]
            if len(act_names) > 6:
                act_view = act_names[:3] + ["..."] + act_names[-2:]
            else:
                act_view = act_names
            f.write(
                f"  idx={i} name={sname} category={cat} active_pe={ape} assigned_sub_batches={sb} active_blocks={act_view}\n"
            )

        f.write("\nScT_cumulative(rows=state, cols=merged_block)\n")
        f.write(fmt_matrix(sct) + "\n")

        f.write("\nScT_delta_per_state(rows=state, cols=merged_block)\n")
        f.write(fmt_matrix(delta) + "\n")

        f.write("\nMeT_S(rows=state, cols=merged_block)\n")
        f.write(fmt_matrix(res.met.sram) + "\n")

        f.write("\nMeT_D(rows=state, cols=merged_block)\n")
        f.write(fmt_matrix(res.met.dram) + "\n")


def write_html(path: Path, cand: CandidateResult) -> None:
    res = cand.result
    write_schedule_html(
        path,
        title="Transformer Minimal-Layer Block Merge",
        meta={
            "network": "transformer_official_min_layer",
            "source_ref": "src/nns/transformer.cpp",
            "raw_layer_count": 471,
            "max_layers_per_block": cand.max_layers_per_block,
            "merged_blocks": len(res.scheduled_blocks),
            "num_states": len(res.state_order),
            "best_sub_batch": res.best_sub_batch,
            "latency": f"{res.total_latency:.6f}",
            "energy": f"{res.total_energy:.6f}",
            "edp": f"{res.total_edp:.6f}",
        },
        scheduled_blocks=list(res.scheduled_blocks),
        state_order=list(res.state_order),
        state_categories=list(res.state_categories),
        state_batches=list(res.milp_solution.state_batches),
        state_active_blocks=list(res.state_active_blocks),
        sct=res.sct.table.tolist(),
        met_s=res.met.sram.tolist(),
        met_d=res.met.dram.tolist(),
        hierarchy_notes=list(res.hierarchy_notes),
        hierarchy_traces=list(res.hierarchy_traces),
    )


def main() -> None:
    blocks = build_transformer_min_layers()

    candidate_caps = [12, 16, 24, 32]
    candidates: list[CandidateResult] = []
    for cap in candidate_caps:
        candidates.append(run_candidate(blocks, cap))

    candidates.sort(key=lambda x: x.result.total_edp)
    best = candidates[0]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "outputs" / "experiments" / f"transformer_min_layer_block_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "candidates.csv"
    txt_path = out_dir / "summary.txt"
    best_detail = out_dir / "best_detail.txt"
    best_html = out_dir / "best_detail.html"

    with csv_path.open("w", encoding="utf-8", newline="\n") as f:
        fields = [
            "max_layers_per_block",
            "raw_layer_count",
            "merged_blocks",
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
        for c in candidates:
            w.writerow(
                {
                    "max_layers_per_block": c.max_layers_per_block,
                    "raw_layer_count": len(blocks),
                    "merged_blocks": len(c.result.scheduled_blocks),
                    "num_states": len(c.result.state_order),
                    "best_sub_batch": c.result.best_sub_batch,
                    "state_batches": ",".join(str(x) for x in c.result.milp_solution.state_batches),
                    "latency": f"{c.result.total_latency:.6f}",
                    "energy": f"{c.result.total_energy:.6f}",
                    "edp": f"{c.result.total_edp:.6f}",
                    "sct_solver": c.result.sct_solver_name,
                    "met_solver": c.result.met_solver_name,
                }
            )

    with txt_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("experiment=transformer_min_layer_plus_block_merge\n")
        f.write("source_ref=src/nns/transformer.cpp\n")
        f.write(f"raw_layer_count={len(blocks)}\n")
        f.write(f"candidate_max_layers_per_block={candidate_caps}\n")
        f.write(f"best_max_layers_per_block={best.max_layers_per_block}\n")
        f.write(f"best_merged_blocks={len(best.result.scheduled_blocks)}\n")
        f.write(f"best_num_states={len(best.result.state_order)}\n")
        f.write(f"best_sub_batch={best.result.best_sub_batch}\n")
        f.write(f"best_latency={best.result.total_latency:.6f}\n")
        f.write(f"best_energy={best.result.total_energy:.6f}\n")
        f.write(f"best_edp={best.result.total_edp:.6f}\n")
        f.write(f"best_detail={best_detail}\n")
        f.write(f"best_html={best_html}\n")

    write_detail(best_detail, blocks, best)
    write_html(best_html, best)

    print(f"out_dir={out_dir}")
    print(f"summary={txt_path}")
    print(f"candidates={csv_path}")
    print(f"best_detail={best_detail}")
    print(f"best_html={best_html}")


if __name__ == "__main__":
    main()


