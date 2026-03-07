from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import multiprocessing as mp
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.layer import Layer
from scheduler.block import Block
from search.scheduler_search import SearchConfig, search_schedule
from example.run_official_nns_suite import official_specs
from scheduler.hardware_profile import paper_7_2_search_params


# Stage layouts from outputs/visualization/official_nns_structure.html
OFFICIAL_STAGE_LAYOUTS: dict[str, list[tuple[str, int]]] = {
    "alexnet": [
        ("conv1 split", 2),
        ("pool1", 2),
        ("conv2", 2),
        ("pool2", 2),
        ("conv3", 2),
        ("conv4", 2),
        ("conv5", 2),
        ("pool3", 2),
        ("fc", 3),
    ],
    "darknet19": [("conv+pool stack", 24), ("pool_avg", 1)],
    "densenet": [
        ("stem(conv0+pool0)", 2),
        ("dense_block1", 12),
        ("transition1", 2),
        ("dense_block2", 24),
        ("transition2", 2),
        ("dense_block3", 48),
        ("transition3", 2),
        ("dense_block4", 32),
        ("head(pool_avg+fc)", 2),
    ],
    "gnmt": [("word_embed", 1), ("8x GNMT block", 72), ("8x residual add", 8), ("Wd", 1)],
    "googlenet": [
        ("stem", 5),
        ("inception 3a/3b", 14),
        ("inception 4a..4e", 35),
        ("inception 5a/5b", 14),
        ("head", 4),
    ],
    "incep_resnet": [
        ("Stem", 7),
        ("5x Inception-ResNet-A", 40),
        ("Reduction-A", 5),
        ("10x Inception-ResNet-B", 60),
        ("Reduction-B", 8),
        ("5x Inception-ResNet-C", 30),
        ("head", 3),
    ],
    # llm.cpp official list in visualization is block variants; use BERT_block view.
    "llm": [("word_embed", 1), ("1x transformer block", 42), ("proj", 1)],
    "pnasnet": [("conv0", 1), ("stem1", 43), ("stem2", 45), ("12x cells", 511), ("head", 3)],
    # official_specs "resnet" mapped to resnet50 in visualization.
    "resnet": [("stem", 2), ("stage2", 13), ("stage3", 17), ("stage4", 25), ("stage5", 13), ("head", 2)],
    "transformer": [("word_embed_enc", 1), ("6x encoder", 162), ("word_embed_dec", 1), ("6x decoder", 306), ("proj", 1)],
    # official_specs "vgg" mapped to vgg19 in visualization.
    "vgg": [("conv1..conv16", 16), ("pool1..pool5", 5)],
    "zfnet": [("conv+pool", 8), ("fc1..fc3", 3)],
}


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
    ok: bool
    status: str
    error: str


def _candidate_sub_batches(batch_size: int) -> list[int]:
    base = [2, 4, 8, 16, 32]
    valid = [x for x in base if x <= batch_size and batch_size % x == 0]
    return valid if valid else [batch_size]


def _layout_for_network(network: str) -> list[tuple[str, int]]:
    layout = OFFICIAL_STAGE_LAYOUTS.get(network)
    if layout is None:
        raise KeyError(f"missing stage layout for network={network}")
    return layout


def _raw_layers(layout: list[tuple[str, int]]) -> int:
    return int(sum(int(cnt) for _, cnt in layout))


def _spec_totals(spec_layers) -> tuple[float, float]:
    total_flops = sum(float(gflops) * 1e9 for _, gflops, _ in spec_layers)
    total_out = sum(float(out_mb) for _, _, out_mb in spec_layers)
    return float(total_flops), float(total_out)


def _build_min_layers(spec_layers, layout: list[tuple[str, int]]) -> list[Block]:
    total_layers = _raw_layers(layout)
    total_flops, total_out = _spec_totals(spec_layers)

    layers: list[Layer] = []
    prev: Layer | None = None

    for stage_name, stage_layers in layout:
        share = float(stage_layers) / float(total_layers)
        stage_flops = total_flops * share
        stage_out = total_out * share
        per_flops = stage_flops / float(stage_layers)
        per_out = stage_out / float(stage_layers)

        for i in range(int(stage_layers)):
            layer = Layer(name=f"{stage_name}__l{i + 1:04d}", flops=per_flops, output_size=per_out)
            if prev is not None:
                prev.connect_to(layer)
            prev = layer
            layers.append(layer)

    return [Block(name=ly.name, layers=[ly]) for ly in layers]


def _build_stage_blocks_from_min_layers(min_layer_blocks: list[Block], layout: list[tuple[str, int]]) -> list[Block]:
    layers = [b.layers[0] for b in min_layer_blocks]

    out: list[Block] = []
    cursor = 0
    for stage_name, count in layout:
        part = layers[cursor : cursor + int(count)]
        if len(part) != int(count):
            raise RuntimeError(f"layout mismatch at stage={stage_name}")
        out.append(Block(name=stage_name, layers=list(part)))
        cursor += int(count)

    if cursor != len(layers):
        raise RuntimeError("layout does not consume all minimal layers")
    return out


def _cfg(batch_size: int, num_pes: int, max_layers_per_block: int, use_paper_hw_7_2: bool) -> SearchConfig:
    if use_paper_hw_7_2:
        hw = paper_7_2_search_params(num_pes=num_pes)
    else:
        hw = {
            "sram_capacity": 15000.0,
            "dram_capacity": 30000.0,
            "noc_bandwidth": 4096.0,
            "dram_bandwidth": 4096.0,
            "noc_energy_per_unit": 0.0,
            "dram_energy_per_unit": 0.0075,
            "dram_noc_hops": 1.0,
            "compute_power_per_tile": 1.0,
            "compute_energy_per_op": 1e-12,
        }

    return SearchConfig(
        batch_size=batch_size,
        candidate_sub_batches=_candidate_sub_batches(batch_size),
        sram_capacity=float(hw["sram_capacity"]),
        dram_capacity=float(hw["dram_capacity"]),
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
        noc_bandwidth=float(hw["noc_bandwidth"]),
        dram_bandwidth=float(hw["dram_bandwidth"]),
        noc_energy_per_unit=float(hw["noc_energy_per_unit"]),
        dram_energy_per_unit=float(hw["dram_energy_per_unit"]),
        dram_noc_hops=float(hw["dram_noc_hops"]),
        latency_combine_mode="max",
        compute_power_per_tile=float(hw["compute_power_per_tile"]),
        compute_energy_per_op=float(hw["compute_energy_per_op"]),
        allow_solver_fallback=False,
    )


def _worker_run(queue: mp.Queue, network: str, mode: str, spec: dict, num_pes: int, max_layers_per_block: int, use_paper_hw_7_2: bool) -> None:
    try:
        layout = _layout_for_network(network)
        min_blocks = _build_min_layers(spec["layers"], layout)
        if mode == "stage_level":
            blocks = _build_stage_blocks_from_min_layers(min_blocks, layout)
        elif mode == "layer_level":
            blocks = min_blocks
        else:
            raise ValueError(f"unknown mode={mode}")

        cfg = _cfg(batch_size=int(spec["batch_size"]), num_pes=num_pes, max_layers_per_block=max_layers_per_block, use_paper_hw_7_2=use_paper_hw_7_2)
        res = search_schedule(blocks, cfg)

        queue.put(
            {
                "ok": True,
                "status": "ok",
                "error": "",
                "input_blocks": len(blocks),
                "scheduled_blocks": len(res.scheduled_blocks),
                "num_states": len(res.state_order),
                "best_sub_batch": res.best_sub_batch,
                "latency": res.total_latency,
                "energy": res.total_energy,
                "edp": res.total_edp,
                "sct_solver": res.sct_solver_name,
                "met_solver": res.met_solver_name,
            }
        )
    except Exception as exc:
        queue.put(
            {
                "ok": False,
                "status": "error",
                "error": str(exc),
                "input_blocks": 0,
                "scheduled_blocks": 0,
                "num_states": 0,
                "best_sub_batch": 0,
                "latency": 0.0,
                "energy": 0.0,
                "edp": 0.0,
                "sct_solver": "",
                "met_solver": "",
            }
        )


def _run_one_with_timeout(
    network: str,
    source_ref: str,
    batch_size: int,
    spec: dict,
    mode: str,
    details_dir: Path,
    num_pes: int,
    max_layers_per_block: int,
    timeout_sec: int,
    use_paper_hw_7_2: bool,
) -> RunOutcome:
    q: mp.Queue = mp.Queue()
    p = mp.Process(
        target=_worker_run,
        args=(q, network, mode, spec, num_pes, max_layers_per_block, use_paper_hw_7_2),
    )
    p.start()
    p.join(timeout=timeout_sec)

    if p.is_alive():
        p.terminate()
        p.join(timeout=5)
        detail = details_dir / f"{network}__{mode}.txt"
        with detail.open("w", encoding="utf-8", newline="\n") as f:
            f.write(f"network={network}\n")
            f.write(f"mode={mode}\n")
            f.write(f"source_ref={source_ref}\n")
            f.write(f"batch_size={batch_size}\n")
            f.write(f"num_pes={num_pes}\n")
            f.write(f"max_layers_per_block={max_layers_per_block}\n")
            f.write(f"status=timeout\n")
            f.write(f"timeout_sec={timeout_sec}\n")

        return RunOutcome(
            network=network,
            mode=mode,
            source_ref=source_ref,
            batch_size=batch_size,
            input_blocks=0,
            scheduled_blocks=0,
            num_states=0,
            best_sub_batch=0,
            latency=0.0,
            energy=0.0,
            edp=0.0,
            sct_solver="",
            met_solver="",
            ok=False,
            status="timeout",
            error=f"timeout after {timeout_sec}s",
        )

    if q.empty():
        return RunOutcome(
            network=network,
            mode=mode,
            source_ref=source_ref,
            batch_size=batch_size,
            input_blocks=0,
            scheduled_blocks=0,
            num_states=0,
            best_sub_batch=0,
            latency=0.0,
            energy=0.0,
            edp=0.0,
            sct_solver="",
            met_solver="",
            ok=False,
            status="error",
            error="worker exited without result",
        )

    data = q.get()
    detail = details_dir / f"{network}__{mode}.txt"
    with detail.open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"network={network}\n")
        f.write(f"mode={mode}\n")
        f.write(f"source_ref={source_ref}\n")
        f.write(f"batch_size={batch_size}\n")
        f.write(f"num_pes={num_pes}\n")
        f.write(f"max_layers_per_block={max_layers_per_block}\n")
        f.write(f"status={data['status']}\n")
        if data["ok"]:
            f.write(f"input_blocks={data['input_blocks']}\n")
            f.write(f"scheduled_blocks={data['scheduled_blocks']}\n")
            f.write(f"num_states={data['num_states']}\n")
            f.write(f"best_sub_batch={data['best_sub_batch']}\n")
            f.write(f"sct_solver={data['sct_solver']}\n")
            f.write(f"met_solver={data['met_solver']}\n")
            f.write(f"latency={data['latency']:.6f}\n")
            f.write(f"energy={data['energy']:.6f}\n")
            f.write(f"edp={data['edp']:.6f}\n")
        else:
            f.write(f"error={data['error']}\n")

    return RunOutcome(
        network=network,
        mode=mode,
        source_ref=source_ref,
        batch_size=batch_size,
        input_blocks=int(data["input_blocks"]),
        scheduled_blocks=int(data["scheduled_blocks"]),
        num_states=int(data["num_states"]),
        best_sub_batch=int(data["best_sub_batch"]),
        latency=float(data["latency"]),
        energy=float(data["energy"]),
        edp=float(data["edp"]),
        sct_solver=str(data["sct_solver"]),
        met_solver=str(data["met_solver"]),
        ok=bool(data["ok"]),
        status=str(data["status"]),
        error=str(data["error"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="All-network stage vs layer compare with same-source construction and block merge enabled",
    )
    parser.add_argument("--num-pes", type=int, default=16)
    parser.add_argument("--timeout-sec", type=int, default=120)
    parser.add_argument("--max-layers-per-block", type=int, default=32)
    parser.add_argument("--paper-hw-7-2", action="store_true", help="use paper Section 7.2 hardware parameters")
    args = parser.parse_args()

    num_pes = int(args.num_pes)
    timeout_sec = max(10, int(args.timeout_sec))
    max_layers_per_block = max(2, int(args.max_layers_per_block))
    use_paper_hw_7_2 = bool(args.paper_hw_7_2)

    specs = official_specs()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "outputs" / "experiments" / f"all_networks_same_source_merge_pe{num_pes}_{ts}"
    details_dir = out_dir / "details"
    details_dir.mkdir(parents=True, exist_ok=True)

    outcomes: list[RunOutcome] = []

    for name in sorted(specs.keys()):
        spec = specs[name]
        outcomes.append(
            _run_one_with_timeout(
                network=name,
                source_ref=str(spec["source_ref"]),
                batch_size=int(spec["batch_size"]),
                spec=spec,
                mode="stage_level",
                details_dir=details_dir,
                num_pes=num_pes,
                max_layers_per_block=max_layers_per_block,
                timeout_sec=timeout_sec,
                use_paper_hw_7_2=use_paper_hw_7_2,
            )
        )
        outcomes.append(
            _run_one_with_timeout(
                network=name,
                source_ref=str(spec["source_ref"]),
                batch_size=int(spec["batch_size"]),
                spec=spec,
                mode="layer_level",
                details_dir=details_dir,
                num_pes=num_pes,
                max_layers_per_block=max_layers_per_block,
                timeout_sec=timeout_sec,
                use_paper_hw_7_2=use_paper_hw_7_2,
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
            "timeout_sec",
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
            "ok",
            "status",
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
                    "timeout_sec": timeout_sec,
                    "max_layers_per_block": max_layers_per_block,
                    "input_blocks": o.input_blocks,
                    "scheduled_blocks": o.scheduled_blocks,
                    "num_states": o.num_states,
                    "best_sub_batch": o.best_sub_batch,
                    "latency": f"{o.latency:.6f}" if o.ok else "",
                    "energy": f"{o.energy:.6f}" if o.ok else "",
                    "edp": f"{o.edp:.6f}" if o.ok else "",
                    "sct_solver": o.sct_solver,
                    "met_solver": o.met_solver,
                    "ok": o.ok,
                    "status": o.status,
                    "error": o.error,
                }
            )

    by_net: dict[str, dict[str, RunOutcome]] = {}
    for o in outcomes:
        by_net.setdefault(o.network, {})[o.mode] = o

    comparable = 0
    holds = 0
    violations = 0
    stage_timeout = 0
    layer_timeout = 0

    with txt_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("experiment=all_networks_same_source_merge_stage_vs_layer\n")
        f.write(f"num_pes={num_pes}\n")
        f.write(f"timeout_sec={timeout_sec}\n")
        f.write(f"max_layers_per_block={max_layers_per_block}\n")
        f.write("same_source=True\n")
        f.write("same_constraints=True\n")
        f.write("block_merge_enabled=True\n")
        f.write("criterion_check=layer_edp<=stage_edp\n\n")

        for name in sorted(by_net.keys()):
            stage = by_net[name].get("stage_level")
            layer = by_net[name].get("layer_level")
            f.write(f"[{name}]\n")

            if stage is None or layer is None:
                f.write("  status=missing\n\n")
                continue

            f.write(f"  stage_status={stage.status}\n")
            f.write(f"  layer_status={layer.status}\n")

            if stage.status == "timeout":
                stage_timeout += 1
            if layer.status == "timeout":
                layer_timeout += 1

            if (not stage.ok) or (not layer.ok):
                if not stage.ok:
                    f.write(f"  stage_error={stage.error}\n")
                if not layer.ok:
                    f.write(f"  layer_error={layer.error}\n")
                f.write("\n")
                continue

            comparable += 1
            cond = layer.edp <= stage.edp + 1e-12
            if cond:
                holds += 1
                verdict = "holds"
            else:
                violations += 1
                verdict = "violated"

            ratio = layer.edp / stage.edp if stage.edp > 0 else float("inf")

            f.write(f"  stage_input_blocks={stage.input_blocks}\n")
            f.write(f"  layer_input_blocks={layer.input_blocks}\n")
            f.write(f"  stage_scheduled_blocks={stage.scheduled_blocks}\n")
            f.write(f"  layer_scheduled_blocks={layer.scheduled_blocks}\n")
            f.write(f"  stage_edp={stage.edp:.6f}\n")
            f.write(f"  layer_edp={layer.edp:.6f}\n")
            f.write(f"  layer_over_stage={ratio:.6f}\n")
            f.write(f"  inequality={verdict}\n\n")

        f.write("[overall]\n")
        f.write(f"  total_networks={len(by_net)}\n")
        f.write(f"  comparable_networks={comparable}\n")
        f.write(f"  inequality_holds_count={holds}\n")
        f.write(f"  inequality_violations_count={violations}\n")
        f.write(f"  stage_timeout_count={stage_timeout}\n")
        f.write(f"  layer_timeout_count={layer_timeout}\n")

    print(f"out_dir={out_dir}")
    print(f"summary_txt={txt_path}")
    print(f"summary_csv={csv_path}")
    print(f"details_dir={details_dir}")


if __name__ == "__main__":
    main()







