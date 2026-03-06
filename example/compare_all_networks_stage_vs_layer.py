from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.layer import Layer
from scheduler.block import Block
from search.scheduler_search import SearchConfig, search_schedule
from example.run_official_nns_suite import official_specs


@dataclass
class RunOutcome:
    network: str
    mode: str
    source_ref: str
    batch_size: int
    input_blocks: int
    scheduled_blocks: int
    num_states: int
    best_sub_batch: int
    latency: float
    energy: float
    edp: float
    sct_solver: str
    met_solver: str
    state_batches: str
    ok: bool
    error: str


def _candidates(batch_size: int) -> list[int]:
    base = [2, 4, 8, 16, 32]
    return [x for x in base if x <= batch_size and batch_size % x == 0]


def _build_layer_blocks(layers_spec) -> list[Block]:
    blocks: list[Block] = []
    prev_layer: Layer | None = None
    for lname, gflops, out_mb in layers_spec:
        layer = Layer(name=lname, flops=float(gflops) * 1e9, output_size=float(out_mb))
        if prev_layer is not None:
            prev_layer.connect_to(layer)
        prev_layer = layer
        blocks.append(Block(name=lname, layers=[layer]))
    return blocks


def _partition_sizes(total: int, parts: int) -> list[int]:
    base = total // parts
    rem = total % parts
    out = [base] * parts
    for i in range(rem):
        out[i] += 1
    return out


def _build_stage_blocks_from_layer_blocks(layer_blocks: list[Block], stage_count: int) -> list[Block]:
    n = len(layer_blocks)
    if n == 0:
        return []

    s = max(1, min(stage_count, n))
    sizes = _partition_sizes(n, s)

    out: list[Block] = []
    cursor = 0
    for idx, sz in enumerate(sizes):
        part = layer_blocks[cursor : cursor + sz]
        cursor += sz
        if not part:
            continue

        layers = []
        names = []
        for b in part:
            layers.extend(b.layers)
            names.append(b.name)

        name = f"stage_{idx:02d}_{names[0]}_to_{names[-1]}"
        out.append(Block(name=name, layers=layers, sub_blocks=part))

    return out


def _cfg(batch_size: int, num_pes: int) -> SearchConfig:
    return SearchConfig(
        batch_size=batch_size,
        candidate_sub_batches=_candidates(batch_size),
        sram_capacity=15000.0,
        dram_capacity=30000.0,
        num_pes=num_pes,
        enable_chain_block_merge=False,
        max_layers_per_block=16,
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


def _run_one(network: str, source_ref: str, batch_size: int, blocks: list[Block], mode: str, details_dir: Path, num_pes: int) -> RunOutcome:
    cfg = _cfg(batch_size=batch_size, num_pes=num_pes)

    try:
        result = search_schedule(blocks, cfg)
        detail = details_dir / f"{network}__{mode}.txt"
        with detail.open("w", encoding="utf-8", newline="\n") as f:
            f.write(f"network={network}\n")
            f.write(f"mode={mode}\n")
            f.write(f"source_ref={source_ref}\n")
            f.write(f"batch_size={batch_size}\n")
            f.write(f"num_pes={num_pes}\n")
            f.write(f"input_blocks={len(blocks)}\n")
            f.write(f"scheduled_blocks={len(result.scheduled_blocks)}\n")
            f.write(f"num_states={len(result.state_order)}\n")
            f.write(f"best_sub_batch={result.best_sub_batch}\n")
            f.write(f"sct_solver={result.sct_solver_name}\n")
            f.write(f"met_solver={result.met_solver_name}\n")
            f.write(f"latency={result.total_latency:.6f}\n")
            f.write(f"energy={result.total_energy:.6f}\n")
            f.write(f"edp={result.total_edp:.6f}\n")
            f.write(f"state_batches={','.join(str(x) for x in result.milp_solution.state_batches)}\n")

        return RunOutcome(
            network=network,
            mode=mode,
            source_ref=source_ref,
            batch_size=batch_size,
            input_blocks=len(blocks),
            scheduled_blocks=len(result.scheduled_blocks),
            num_states=len(result.state_order),
            best_sub_batch=result.best_sub_batch,
            latency=result.total_latency,
            energy=result.total_energy,
            edp=result.total_edp,
            sct_solver=result.sct_solver_name,
            met_solver=result.met_solver_name,
            state_batches=",".join(str(x) for x in result.milp_solution.state_batches),
            ok=True,
            error="",
        )
    except Exception as exc:
        return RunOutcome(
            network=network,
            mode=mode,
            source_ref=source_ref,
            batch_size=batch_size,
            input_blocks=len(blocks),
            scheduled_blocks=0,
            num_states=0,
            best_sub_batch=0,
            latency=0.0,
            energy=0.0,
            edp=0.0,
            sct_solver="",
            met_solver="",
            state_batches="",
            ok=False,
            error=str(exc),
        )


def main() -> None:
    num_pes = 16
    stage_count = 4

    specs = official_specs()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "outputs" / "experiments" / f"all_networks_stage_vs_layer_pe{num_pes}_{ts}"
    details_dir = out_dir / "details"
    details_dir.mkdir(parents=True, exist_ok=True)

    outcomes: list[RunOutcome] = []

    for name in sorted(specs.keys()):
        spec = specs[name]
        layer_blocks = _build_layer_blocks(spec["layers"])
        stage_blocks = _build_stage_blocks_from_layer_blocks(layer_blocks, stage_count=stage_count)

        outcomes.append(
            _run_one(
                network=name,
                source_ref=str(spec["source_ref"]),
                batch_size=int(spec["batch_size"]),
                blocks=stage_blocks,
                mode="stage_level",
                details_dir=details_dir,
                num_pes=num_pes,
            )
        )
        outcomes.append(
            _run_one(
                network=name,
                source_ref=str(spec["source_ref"]),
                batch_size=int(spec["batch_size"]),
                blocks=layer_blocks,
                mode="layer_level",
                details_dir=details_dir,
                num_pes=num_pes,
            )
        )

    csv_path = out_dir / "summary.csv"
    txt_path = out_dir / "summary.txt"

    with csv_path.open("w", encoding="utf-8", newline="\n") as f:
        fields = [
            "network",
            "mode",
            "source_ref",
            "batch_size",
            "num_pes",
            "input_blocks",
            "scheduled_blocks",
            "num_states",
            "best_sub_batch",
            "latency",
            "energy",
            "edp",
            "sct_solver",
            "met_solver",
            "state_batches",
            "ok",
            "error",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for o in outcomes:
            w.writerow(
                {
                    "network": o.network,
                    "mode": o.mode,
                    "source_ref": o.source_ref,
                    "batch_size": o.batch_size,
                    "num_pes": num_pes,
                    "input_blocks": o.input_blocks,
                    "scheduled_blocks": o.scheduled_blocks,
                    "num_states": o.num_states,
                    "best_sub_batch": o.best_sub_batch,
                    "latency": f"{o.latency:.6f}" if o.ok else "",
                    "energy": f"{o.energy:.6f}" if o.ok else "",
                    "edp": f"{o.edp:.6f}" if o.ok else "",
                    "sct_solver": o.sct_solver,
                    "met_solver": o.met_solver,
                    "state_batches": o.state_batches,
                    "ok": o.ok,
                    "error": o.error,
                }
            )

    # Pairwise compare: layer vs stage per network.
    by_net: dict[str, dict[str, RunOutcome]] = {}
    for o in outcomes:
        by_net.setdefault(o.network, {})[o.mode] = o

    with txt_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("experiment=all_networks_stage_vs_layer\n")
        f.write(f"num_pes={num_pes}\n")
        f.write(f"stage_count={stage_count}\n")
        f.write("strict_paper_mode=True\n")
        f.write("allow_solver_fallback=False\n\n")

        better_layer = 0
        better_stage = 0
        tied = 0

        for name in sorted(by_net.keys()):
            stage = by_net[name].get("stage_level")
            layer = by_net[name].get("layer_level")

            f.write(f"[{name}]\n")
            if stage is None or layer is None or (not stage.ok) or (not layer.ok):
                f.write("  status: failed\n")
                if stage is not None and not stage.ok:
                    f.write(f"  stage_error: {stage.error}\n")
                if layer is not None and not layer.ok:
                    f.write(f"  layer_error: {layer.error}\n")
                f.write("\n")
                continue

            ratio = layer.edp / stage.edp if stage.edp > 0 else float("inf")
            if abs(ratio - 1.0) < 1e-9:
                winner = "tie"
                tied += 1
            elif ratio < 1.0:
                winner = "layer"
                better_layer += 1
            else:
                winner = "stage"
                better_stage += 1

            f.write(f"  stage_edp: {stage.edp:.6f}\n")
            f.write(f"  layer_edp: {layer.edp:.6f}\n")
            f.write(f"  layer_over_stage: {ratio:.6f}\n")
            f.write(f"  winner: {winner}\n\n")

        f.write("[overall]\n")
        f.write(f"  better_stage: {better_stage}\n")
        f.write(f"  better_layer: {better_layer}\n")
        f.write(f"  tied: {tied}\n")
        f.write(f"  total_networks: {len(by_net)}\n")

    print(f"out_dir={out_dir}")
    print(f"summary_txt={txt_path}")
    print(f"summary_csv={csv_path}")
    print(f"details_dir={details_dir}")


if __name__ == "__main__":
    main()
