from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from example.run_transformer_min_layer_block_experiment import build_transformer_min_layers
from example.schedule_html import write_schedule_html
from scheduler.hardware_profile import paper_7_3_search_params
from search.scheduler_search import SearchConfig, SearchResult, search_schedule


def _fmt_matrix(rows: list[list[float]]) -> str:
    return "\n".join(",".join(f"{float(v):.0f}" for v in row) for row in rows)


def _delta_from_cumulative(sct: list[list[float]]) -> list[list[float]]:
    out: list[list[float]] = []
    prev = [0.0 for _ in sct[0]] if sct else []
    for row in sct:
        out.append([max(0.0, float(v) - float(p)) for v, p in zip(row, prev)])
        prev = list(row)
    return out


def _active_blocks_from_delta(delta: list[list[float]]) -> list[list[int]]:
    out: list[list[int]] = []
    for row in delta:
        out.append([i for i, v in enumerate(row) if float(v) > 0.0])
    return out


def _write_phase_detail(path: Path, phase_name: str, phase: dict[str, Any]) -> None:
    sct = [[float(v) for v in row] for row in phase["sct"]]
    met_s = [[float(v) for v in row] for row in phase["met_s"]]
    met_d = [[float(v) for v in row] for row in phase["met_d"]]
    delta = _delta_from_cumulative(sct)

    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"phase={phase_name}\n")
        f.write(f"best_sub_batch={phase['best_sub_batch']}\n")
        f.write(f"blocks={len(phase['scheduled_blocks'])}\n")
        f.write(f"states={len(phase['state_order'])}\n")
        f.write(f"compute_latency={float(phase['compute_latency']):.6f}\n")
        f.write(f"compute_energy={float(phase['compute_energy']):.6f}\n")
        f.write(f"memory_latency={float(phase['memory_latency']):.6f}\n")
        f.write(f"memory_energy={float(phase['memory_energy']):.6f}\n")
        f.write(f"total_latency={float(phase['total_latency']):.6f}\n")
        f.write(f"total_energy={float(phase['total_energy']):.6f}\n")
        f.write(f"total_edp={float(phase['total_edp']):.6f}\n\n")

        f.write("state_batches\n")
        f.write(",".join(str(int(x)) for x in phase["state_batches"]) + "\n\n")

        f.write("ScT_cumulative(rows=state, cols=block)\n")
        f.write(_fmt_matrix(sct) + "\n\n")
        f.write("ScT_delta_per_state(rows=state, cols=block)\n")
        f.write(_fmt_matrix(delta) + "\n\n")
        f.write("MeT_S(rows=state, cols=block)\n")
        f.write(_fmt_matrix(met_s) + "\n\n")
        f.write("MeT_D(rows=state, cols=block)\n")
        f.write(_fmt_matrix(met_d) + "\n")


def _write_phase_html(path: Path, phase_name: str, phase: dict[str, Any], sub_batch: int) -> None:
    sct = [[float(v) for v in row] for row in phase["sct"]]
    met_s = [[float(v) for v in row] for row in phase["met_s"]]
    met_d = [[float(v) for v in row] for row in phase["met_d"]]
    delta = _delta_from_cumulative(sct)

    write_schedule_html(
        path,
        title=f"Transformer Training {phase_name.upper()} Schedule",
        meta={
            "phase": phase_name,
            "sub_batch": sub_batch,
            "states": len(phase["state_order"]),
            "blocks": len(phase["scheduled_blocks"]),
            "latency": f"{float(phase['total_latency']):.6f}",
            "energy": f"{float(phase['total_energy']):.6f}",
            "edp": f"{float(phase['total_edp']):.6f}",
        },
        scheduled_blocks=list(phase["scheduled_blocks"]),
        state_order=list(phase["state_order"]),
        state_categories=list(phase["state_categories"]),
        state_batches=[int(x) for x in phase["state_batches"]],
        state_active_blocks=_active_blocks_from_delta(delta),
        sct=sct,
        met_s=met_s,
        met_d=met_d,
        hierarchy_notes=None,
        hierarchy_traces=None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict training reproduction for transformer (FW/BW1/BW2)")
    parser.add_argument("--num-pes", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--verbose-progress", action="store_true")
    parser.add_argument("--all-sub-batch-factors", action="store_true")
    parser.add_argument("--candidate-sub-batches", type=str, default="4,8,16,32")
    args = parser.parse_args()

    candidates = [int(x) for x in args.candidate_sub_batches.split(",") if x.strip()]
    if not candidates:
        raise ValueError("empty --candidate-sub-batches")

    hw = paper_7_3_search_params(num_pes=args.num_pes, dram_capacity_mb=32768.0)

    cfg = SearchConfig(
        batch_size=int(args.batch_size),
        candidate_sub_batches=candidates,
        sram_capacity=float(hw["sram_capacity"]),
        dram_capacity=float(hw["dram_capacity"]),
        num_pes=int(args.num_pes),
        strict_paper_mode=True,
        use_all_sub_batch_factors=bool(args.all_sub_batch_factors),
        top_k1=4,
        top_k2=2,
        top_k1_ratio=0.1,
        top_k2_ratio=0.05,
        use_edp_objective=True,
        allow_solver_fallback=False,
        dependency_gap=0,
        noc_bandwidth=float(hw["noc_bandwidth"]),
        dram_bandwidth=float(hw["dram_bandwidth"]),
        noc_energy_per_unit=float(hw["noc_energy_per_unit"]),
        dram_energy_per_unit=float(hw["dram_energy_per_unit"]),
        dram_noc_hops=float(hw["dram_noc_hops"]),
        compute_power_per_tile=float(hw["compute_power_per_tile"]),
        compute_energy_per_op=float(hw["compute_energy_per_op"]),
        enable_chain_block_merge=True,
        max_layers_per_block=16,
        min_layers_per_block=2,
        min_active_states=1,
        min_batch_if_active=1,
        max_state_share=1.0,
        enable_training_recomputation=True,
        backward_compute_scale=2.0,
        backward_output_scale=1.0,
        recompute_compute_scale=1.0,
        recompute_output_scale=1.0,
        verbose_progress=bool(args.verbose_progress),
        progress_prefix="train",
    )

    blocks = build_transformer_min_layers()
    res: SearchResult = search_schedule(blocks, cfg)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "outputs" / "experiments" / f"transformer_training_repro_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_txt = out_dir / "summary.txt"
    summary_csv = out_dir / "summary.csv"
    fw_html = out_dir / "fw_schedule.html"

    write_schedule_html(
        fw_html,
        title="Transformer Training Reproduction - FW Overview",
        meta={
            "mode": "training_recompute",
            "best_sub_batch": res.best_sub_batch,
            "num_pes": args.num_pes,
            "batch_size": args.batch_size,
            "total_latency_all_phases": f"{res.total_latency:.6f}",
            "total_energy_all_phases": f"{res.total_energy:.6f}",
            "total_edp_all_phases": f"{res.total_edp:.6f}",
        },
        scheduled_blocks=res.scheduled_blocks,
        state_order=res.state_order,
        state_categories=res.state_categories,
        state_batches=res.milp_solution.state_batches,
        state_active_blocks=res.state_active_blocks,
        sct=res.sct.table.tolist(),
        met_s=res.met.sram.tolist(),
        met_d=res.met.dram.tolist(),
        hierarchy_notes=res.hierarchy_notes,
        hierarchy_traces=res.hierarchy_traces,
    )

    phase_rows = []
    for phase_name in ["fw", "bw1", "bw2"]:
        phase = res.phase_results.get(phase_name)
        if phase is None:
            continue
        detail_path = out_dir / f"{phase_name}_detail.txt"
        html_path = out_dir / f"{phase_name}_schedule.html"
        _write_phase_detail(detail_path, phase_name, phase)
        _write_phase_html(html_path, phase_name, phase, res.best_sub_batch)

        phase_rows.append(
            {
                "phase": phase_name,
                "blocks": len(phase["scheduled_blocks"]),
                "states": len(phase["state_order"]),
                "sub_batch": int(phase["best_sub_batch"]),
                "latency": f"{float(phase['total_latency']):.6f}",
                "energy": f"{float(phase['total_energy']):.6f}",
                "edp": f"{float(phase['total_edp']):.6f}",
            }
        )

    with summary_csv.open("w", encoding="utf-8", newline="\n") as f:
        fields = ["phase", "blocks", "states", "sub_batch", "latency", "energy", "edp"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(phase_rows)

    with summary_txt.open("w", encoding="utf-8", newline="\n") as f:
        f.write("experiment=transformer_training_reproduction\n")
        f.write("source_ref=src/nns/transformer.cpp\n")
        f.write("mode=FW_BW1_BW2_recomputation\n")
        f.write(f"raw_layer_count={len(blocks)}\n")
        f.write(f"num_pes={args.num_pes}\n")
        f.write(f"batch_size={args.batch_size}\n")
        f.write(f"candidate_sub_batches={candidates}\n")
        f.write(f"use_all_sub_batch_factors={bool(args.all_sub_batch_factors)}\n")
        f.write("hardware_profile=paper_7_3\n")
        f.write(f"best_sub_batch={res.best_sub_batch}\n")
        f.write(f"total_compute_latency={res.compute_latency:.6f}\n")
        f.write(f"total_compute_energy={res.compute_energy:.6f}\n")
        f.write(f"total_memory_latency={res.memory_latency:.6f}\n")
        f.write(f"total_memory_energy={res.memory_energy:.6f}\n")
        f.write(f"total_latency={res.total_latency:.6f}\n")
        f.write(f"total_energy={res.total_energy:.6f}\n")
        f.write(f"total_edp={res.total_edp:.6f}\n")
        for note in res.hierarchy_notes:
            f.write(f"note={note}\n")

    print(f"out_dir={out_dir}")
    print(f"summary={summary_txt}")
    print(f"phase_csv={summary_csv}")
    print(f"fw_html={fw_html}")


if __name__ == "__main__":
    main()
