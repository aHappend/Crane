from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import multiprocessing as mp
from pathlib import Path
import sys

_BOOTSTRAP_ROOT = Path(__file__).absolute().parents[1]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))


ROOT = project_root_from(__file__, 1)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.layer import Layer
from scheduler.block import Block
from search.scheduler_search import SearchConfig, search_schedule
from example.run_official_nns_suite import official_specs


# Stage layouts from outputs/visualization/official_nns_structure.html
# (origin: tools/generate_official_nns_visualization.py)
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
    # llm.cpp official list in the visualization is block variants; use BERT_block view.
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
    target_raw_layers: int
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
    status: str
    error: str


def _candidates(batch_size: int) -> list[int]:
    base = [2, 4, 8, 16, 32]
    valid = [x for x in base if x <= batch_size and batch_size % x == 0]
    if not valid:
        return [batch_size]
    # Small search budget to keep all-network run tractable.
    return [max(valid)]


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


def _build_stage_blocks_from_layout(spec_layers, layout: list[tuple[str, int]]) -> list[Block]:
    total_layers = _raw_layers(layout)
    total_flops, total_out = _spec_totals(spec_layers)

    blocks: list[Block] = []
    prev_layer: Layer | None = None
    for stage_name, stage_layers in layout:
        share = float(stage_layers) / float(total_layers)
        layer = Layer(
            name=stage_name,
            flops=total_flops * share,
            output_size=total_out * share,
        )
        if prev_layer is not None:
            prev_layer.connect_to(layer)
        prev_layer = layer
        blocks.append(Block(name=stage_name, layers=[layer]))
    return blocks


def _build_min_layers_from_layout(spec_layers, layout: list[tuple[str, int]]) -> list[Block]:
    total_layers = _raw_layers(layout)
    total_flops, total_out = _spec_totals(spec_layers)

    blocks: list[Block] = []
    prev_layer: Layer | None = None

    for stage_name, stage_layers in layout:
        share = float(stage_layers) / float(total_layers)
        stage_flops = total_flops * share
        stage_out = total_out * share
        per_flops = stage_flops / float(stage_layers)
        per_out = stage_out / float(stage_layers)

        for i in range(int(stage_layers)):
            lname = f"{stage_name}__l{i + 1:04d}"
            layer = Layer(name=lname, flops=per_flops, output_size=per_out)
            if prev_layer is not None:
                prev_layer.connect_to(layer)
            prev_layer = layer
            blocks.append(Block(name=lname, layers=[layer]))

    return blocks


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
        top_k1=1,
        top_k2=1,
        use_edp_objective=True,
        dependency_gap=0,
        allow_solver_fallback=False,
    )


def _short_state_batches(state_batches: list[int]) -> str:
    if len(state_batches) <= 80:
        return ",".join(str(x) for x in state_batches)
    head = ",".join(str(x) for x in state_batches[:30])
    tail = ",".join(str(x) for x in state_batches[-30:])
    return f"{head},...,{tail}"


def _worker_run(queue: mp.Queue, network: str, mode: str, spec: dict, num_pes: int) -> None:
    try:
        layout = _layout_for_network(network)
        target_raw_layers = _raw_layers(layout)

        if mode == "stage_level":
            blocks = _build_stage_blocks_from_layout(spec["layers"], layout)
        elif mode == "layer_level":
            blocks = _build_min_layers_from_layout(spec["layers"], layout)
        else:
            raise ValueError(f"unknown mode={mode}")

        cfg = _cfg(batch_size=int(spec["batch_size"]), num_pes=num_pes)
        result = search_schedule(blocks, cfg)

        queue.put(
            {
                "ok": True,
                "status": "ok",
                "error": "",
                "target_raw_layers": target_raw_layers,
                "input_blocks": len(blocks),
                "scheduled_blocks": len(result.scheduled_blocks),
                "num_states": len(result.state_order),
                "best_sub_batch": result.best_sub_batch,
                "latency": result.total_latency,
                "energy": result.total_energy,
                "edp": result.total_edp,
                "sct_solver": result.sct_solver_name,
                "met_solver": result.met_solver_name,
                "state_batches": _short_state_batches(result.milp_solution.state_batches),
            }
        )
    except Exception as exc:
        queue.put(
            {
                "ok": False,
                "status": "error",
                "error": str(exc),
                "target_raw_layers": 0,
                "input_blocks": 0,
                "scheduled_blocks": 0,
                "num_states": 0,
                "best_sub_batch": 0,
                "latency": 0.0,
                "energy": 0.0,
                "edp": 0.0,
                "sct_solver": "",
                "met_solver": "",
                "state_batches": "",
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
    timeout_sec: int,
) -> RunOutcome:
    q: mp.Queue = mp.Queue()
    p = mp.Process(target=_worker_run, args=(q, network, mode, spec, num_pes))
    p.start()
    p.join(timeout=timeout_sec)

    if p.is_alive():
        p.terminate()
        p.join(timeout=5)
        target_layers = _raw_layers(_layout_for_network(network))
        detail = details_dir / f"{network}__{mode}.txt"
        with detail.open("w", encoding="utf-8", newline="\n") as f:
            f.write(f"network={network}\n")
            f.write(f"mode={mode}\n")
            f.write(f"source_ref={source_ref}\n")
            f.write(f"batch_size={batch_size}\n")
            f.write(f"num_pes={num_pes}\n")
            f.write(f"target_raw_layers={target_layers}\n")
            f.write(f"status=timeout\n")
            f.write(f"timeout_sec={timeout_sec}\n")

        return RunOutcome(
            network=network,
            mode=mode,
            source_ref=source_ref,
            batch_size=batch_size,
            target_raw_layers=target_layers,
            input_blocks=0,
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
            status="timeout",
            error=f"timeout after {timeout_sec}s",
        )

    if q.empty():
        target_layers = _raw_layers(_layout_for_network(network))
        return RunOutcome(
            network=network,
            mode=mode,
            source_ref=source_ref,
            batch_size=batch_size,
            target_raw_layers=target_layers,
            input_blocks=0,
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
        f.write(f"target_raw_layers={data['target_raw_layers']}\n")
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
            f.write(f"state_batches={data['state_batches']}\n")
        else:
            f.write(f"error={data['error']}\n")

    return RunOutcome(
        network=network,
        mode=mode,
        source_ref=source_ref,
        batch_size=batch_size,
        target_raw_layers=int(data["target_raw_layers"]),
        input_blocks=int(data["input_blocks"]),
        scheduled_blocks=int(data["scheduled_blocks"]),
        num_states=int(data["num_states"]),
        best_sub_batch=int(data["best_sub_batch"]),
        latency=float(data["latency"]),
        energy=float(data["energy"]),
        edp=float(data["edp"]),
        sct_solver=str(data["sct_solver"]),
        met_solver=str(data["met_solver"]),
        state_batches=str(data["state_batches"]),
        ok=bool(data["ok"]),
        status=str(data["status"]),
        error=str(data["error"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare all official networks: stage-level(from official_nns_structure) vs layer-level(raw layers)",
    )
    parser.add_argument("--num-pes", type=int, default=16)
    parser.add_argument("--timeout-sec", type=int, default=120)
    args = parser.parse_args()

    num_pes = int(args.num_pes)
    timeout_sec = max(10, int(args.timeout_sec))
    specs = official_specs()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "outputs" / "experiments" / f"all_networks_stage_vs_layer_pe{num_pes}_{ts}"
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
                timeout_sec=timeout_sec,
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
                timeout_sec=timeout_sec,
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
            "target_raw_layers",
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
                    "target_raw_layers": o.target_raw_layers,
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
                    "status": o.status,
                    "error": o.error,
                }
            )

    by_net: dict[str, dict[str, RunOutcome]] = {}
    for o in outcomes:
        by_net.setdefault(o.network, {})[o.mode] = o

    with txt_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("experiment=all_networks_stage_vs_layer\n")
        f.write(f"num_pes={num_pes}\n")
        f.write(f"timeout_sec={timeout_sec}\n")
        f.write("stage_definition=official_nns_structure_html\n")
        f.write("layer_definition=raw_layers_from_same_stage_layout\n")
        f.write("strict_paper_mode=True\n")
        f.write("allow_solver_fallback=False\n")
        f.write("candidate_sub_batches=max_divisor_in_[2,4,8,16,32]\n")
        f.write("llm_layout=BERT_block_proxy_from_official_visualization\n\n")

        better_layer = 0
        better_stage = 0
        tied = 0
        comparable = 0

        for name in sorted(by_net.keys()):
            stage = by_net[name].get("stage_level")
            layer = by_net[name].get("layer_level")

            f.write(f"[{name}]\n")
            if stage is None or layer is None:
                f.write("  status: missing\n\n")
                continue

            f.write(f"  stage_status: {stage.status}\n")
            f.write(f"  layer_status: {layer.status}\n")
            if not stage.ok:
                f.write(f"  stage_error: {stage.error}\n")
            if not layer.ok:
                f.write(f"  layer_error: {layer.error}\n")

            if (not stage.ok) or (not layer.ok):
                f.write("\n")
                continue

            comparable += 1
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

            f.write(f"  target_raw_layers: {layer.target_raw_layers}\n")
            f.write(f"  stage_input_blocks: {stage.input_blocks}\n")
            f.write(f"  layer_input_blocks: {layer.input_blocks}\n")
            f.write(f"  stage_edp: {stage.edp:.6f}\n")
            f.write(f"  layer_edp: {layer.edp:.6f}\n")
            f.write(f"  layer_over_stage: {ratio:.6f}\n")
            f.write(f"  winner: {winner}\n\n")

        f.write("[overall]\n")
        f.write(f"  better_stage: {better_stage}\n")
        f.write(f"  better_layer: {better_layer}\n")
        f.write(f"  tied: {tied}\n")
        f.write(f"  comparable_networks: {comparable}\n")
        f.write(f"  total_networks: {len(by_net)}\n")

    print(f"out_dir={repo_rel(out_dir, ROOT)}")
    print(f"summary_txt={repo_rel(txt_path, ROOT)}")
    print(f"summary_csv={repo_rel(csv_path, ROOT)}")
    print(f"details_dir={repo_rel(details_dir, ROOT)}")


if __name__ == "__main__":
    main()





