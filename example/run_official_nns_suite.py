import csv
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.layer import Layer
from scheduler.block import Block
from search.scheduler_search import SearchConfig, search_schedule


def official_specs():
    # Proxy layer metadata derived from official src/nns network families.
    return {
        "alexnet": {
            "source_ref": "src/nns/alexnet.cpp",
            "batch_size": 64,
            "layers": [
                ("conv1", 0.72, 18.0), ("pool1", 0.10, 9.0),
                ("conv2", 1.10, 12.0), ("conv3", 1.35, 10.0),
                ("conv4", 1.30, 8.5), ("conv5", 1.20, 8.0),
                ("fc6", 0.45, 4.0), ("fc7", 0.45, 4.0), ("fc8", 0.08, 1.2),
            ],
        },
        "darknet19": {
            "source_ref": "src/nns/darknet19.cpp",
            "batch_size": 64,
            "layers": [
                ("conv1", 0.6, 20.0), ("conv2", 0.9, 16.0),
                ("conv3", 1.4, 14.0), ("conv4", 1.8, 12.0),
                ("conv5", 2.0, 10.0), ("conv6", 2.3, 9.0),
                ("conv7", 1.2, 6.0), ("pred", 0.5, 3.5),
            ],
        },
        "densenet": {
            "source_ref": "src/nns/densenet.cpp",
            "batch_size": 64,
            "layers": [
                ("db1_l1", 1.1, 24.0), ("db1_l2", 1.2, 24.0), ("trans1", 0.4, 18.0),
                ("db2_l1", 1.5, 20.0), ("db2_l2", 1.6, 20.0), ("trans2", 0.5, 15.0),
                ("db3_l1", 1.8, 16.0), ("db3_l2", 2.0, 16.0), ("trans3", 0.6, 12.0),
                ("classifier", 0.4, 2.0),
            ],
        },
        "gnmt": {
            "source_ref": "src/nns/gnmt.cpp",
            "batch_size": 128,
            "layers": [
                ("src_embed", 0.8, 28.0),
                ("enc_lstm1", 3.8, 30.0), ("enc_lstm2", 3.6, 30.0), ("enc_lstm3", 3.5, 30.0),
                ("dec_lstm1", 4.0, 32.0), ("dec_lstm2", 3.7, 32.0),
                ("attn", 1.6, 24.0), ("softmax", 0.7, 8.0),
            ],
        },
        "googlenet": {
            "source_ref": "src/nns/googlenet.cpp",
            "batch_size": 64,
            "layers": [
                ("conv1", 0.9, 18.0), ("pool1", 0.2, 10.0),
                ("inc3a", 2.0, 14.0), ("inc3b", 2.1, 14.0),
                ("inc4a", 2.5, 12.0), ("inc4b", 2.6, 12.0), ("inc4c", 2.6, 12.0),
                ("inc5", 2.1, 9.0), ("fc", 0.5, 2.0),
            ],
        },
        "incep_resnet": {
            "source_ref": "src/nns/incep_resnet.cpp",
            "batch_size": 64,
            "layers": [
                ("conv_stem", 1.1, 20.0),
                ("ir_a1", 3.0, 16.0), ("ir_a2", 3.0, 16.0),
                ("ir_b1", 3.5, 14.0), ("ir_b2", 3.6, 14.0),
                ("ir_c1", 4.0, 12.0), ("ir_c2", 4.1, 12.0),
                ("fc", 0.7, 2.4),
            ],
        },
        "llm": {
            "source_ref": "src/nns/llm.cpp",
            "batch_size": 128,
            "layers": [
                ("token_embed", 2.5, 50.0), ("rope", 0.4, 50.0),
                ("attn1", 8.0, 56.0), ("mlp1", 12.5, 58.0),
                ("attn2", 7.8, 56.0), ("mlp2", 12.0, 58.0),
                ("attn3", 7.6, 55.0), ("mlp3", 11.8, 57.0),
                ("norm", 0.6, 30.0), ("head", 2.0, 12.0),
            ],
        },
        "pnasnet": {
            "source_ref": "src/nns/pnasnet.cpp",
            "batch_size": 64,
            "layers": [
                ("conv1", 0.9, 18.0),
                ("sepconv1", 2.4, 14.0), ("sepconv2", 2.5, 14.0),
                ("sepconv3", 2.8, 12.0), ("sepconv4", 2.9, 12.0),
                ("sepconv5", 3.1, 10.0), ("pool_proj", 0.8, 8.0),
                ("fc", 0.5, 2.0),
            ],
        },
        "resnet": {
            "source_ref": "src/nns/resnet.cpp",
            "batch_size": 64,
            "layers": [
                ("conv1", 1.0, 20.0), ("pool1", 0.2, 12.0),
                ("res2_1", 3.0, 16.0), ("res2_2", 2.8, 16.0),
                ("res3_1", 4.2, 14.0), ("res3_2", 4.0, 14.0),
                ("res4_1", 5.0, 12.0), ("res4_2", 4.8, 12.0),
                ("fc", 0.6, 2.2),
            ],
        },
        "transformer": {
            "source_ref": "src/nns/transformer.cpp",
            "batch_size": 128,
            "layers": [
                ("embed", 1.7, 38.0),
                ("attn1", 4.0, 40.0), ("mlp1", 6.4, 43.0),
                ("attn2", 3.9, 40.0), ("mlp2", 6.2, 43.0),
                ("attn3", 3.8, 40.0), ("mlp3", 6.0, 42.0),
                ("proj", 1.9, 14.0),
            ],
        },
        "vgg": {
            "source_ref": "src/nns/vgg.cpp",
            "batch_size": 64,
            "layers": [
                ("conv1", 0.8, 22.0), ("conv2", 1.2, 20.0),
                ("conv3", 1.8, 16.0), ("conv4", 2.1, 16.0),
                ("conv5", 2.5, 13.0), ("conv6", 2.8, 13.0),
                ("fc6", 0.9, 6.0), ("fc7", 0.9, 6.0), ("fc8", 0.2, 2.0),
            ],
        },
        "zfnet": {
            "source_ref": "src/nns/zfnet.cpp",
            "batch_size": 64,
            "layers": [
                ("conv1", 0.6, 16.0), ("pool1", 0.1, 8.0),
                ("conv2", 1.0, 12.0), ("conv3", 1.2, 10.0),
                ("conv4", 1.2, 8.0), ("conv5", 1.1, 7.0),
                ("fc6", 0.4, 3.0), ("fc7", 0.4, 3.0), ("fc8", 0.1, 1.0),
            ],
        },
    }


def build_layer_blocks(layers):
    blocks = []
    prev = None
    for lname, gflops, out_mb in layers:
        layer = Layer(name=lname, flops=float(gflops) * 1e9, output_size=float(out_mb))
        if prev is not None:
            prev.connect_to(layer)
        prev = layer
        blocks.append(Block(name=lname, layers=[layer]))
    return blocks


def candidates(batch_size):
    base = [2, 4, 8, 16, 32]
    return [x for x in base if x <= batch_size and batch_size % x == 0]


def fmt_matrix(rows):
    return "\n".join(",".join(f"{v:.0f}" for v in r) for r in rows)


def run_one(name, spec):
    blocks = build_layer_blocks(spec["layers"])
    cfg = SearchConfig(
        batch_size=int(spec["batch_size"]),
        candidate_sub_batches=candidates(int(spec["batch_size"])),
        sram_capacity=15000.0,
        dram_capacity=30000.0,
        num_pes=4,
        min_active_states=4,
        min_batch_if_active=1,
        max_state_share=0.45,
    )
    result = search_schedule(blocks, cfg)
    sct = result.sct.table

    # delta[state, layer] = processed amount in this state only
    delta = []
    for i in range(sct.shape[0]):
        if i == 0:
            delta.append(sct[i].copy())
        else:
            delta.append(sct[i] - sct[i - 1])

    layer_state_map = []
    layer_names = [b.name for b in blocks]
    for j, lname in enumerate(layer_names):
        pairs = []
        for i, d in enumerate(delta):
            if d[j] > 0:
                pairs.append(f"{result.state_order[i]}:+{int(d[j])}")
        layer_state_map.append((lname, " ".join(pairs) if pairs else "none"))

    return result, sct, delta, layer_names, layer_state_map


def main():
    specs = official_specs()
    out_dir = ROOT / "outputs" / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    detail_dir = out_dir / f"official_nns_details_{ts}"
    detail_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in sorted(specs.keys()):
        result, sct, delta, layer_names, layer_state_map = run_one(name, specs[name])

        detail = detail_dir / f"{name}.txt"
        with detail.open("w", encoding="utf-8", newline="\n") as f:
            f.write(f"network={name}\n")
            f.write(f"source_ref={specs[name]['source_ref']}\n")
            f.write(f"batch_size={specs[name]['batch_size']}\n")
            f.write(f"best_sub_batch={result.best_sub_batch}\n")
            f.write(f"solver={result.milp_solution.solver_name}\n")
            f.write(f"latency={result.total_latency:.6f}\n")
            f.write(f"energy={result.total_energy:.6f}\n")
            f.write(f"edp={result.total_edp:.6f}\n\n")

            f.write("states\n")
            for i, (sname, cat, ape, sb) in enumerate(
                zip(result.state_order, result.state_categories, result.active_pes, result.milp_solution.state_batches)
            ):
                f.write(f"  idx={i} name={sname} category={cat} active_pe={ape} assigned_sub_batches={sb}\n")

            f.write("\nlayers\n")
            for j, (lname, gflops, out_mb) in enumerate(specs[name]["layers"]):
                f.write(f"  col={j} layer={lname} flops_g={gflops} out_mb={out_mb}\n")

            f.write("\nScT_cumulative(rows=state, cols=layer)\n")
            f.write(fmt_matrix(sct) + "\n")

            f.write("\nScT_delta_per_state(rows=state, cols=layer)\n")
            f.write(fmt_matrix(delta) + "\n")

            f.write("\nlayer_to_state_mapping\n")
            for lname, mapping in layer_state_map:
                f.write(f"  {lname}: {mapping}\n")

        rows.append(
            {
                "network": name,
                "source_ref": specs[name]["source_ref"],
                "batch_size": specs[name]["batch_size"],
                "best_sub_batch": result.best_sub_batch,
                "state_order": " -> ".join(result.state_order),
                "state_batches": ",".join(str(x) for x in result.milp_solution.state_batches),
                "active_states": sum(1 for x in result.milp_solution.state_batches if x > 0),
                "latency": f"{result.total_latency:.6f}",
                "energy": f"{result.total_energy:.6f}",
                "edp": f"{result.total_edp:.6f}",
                "detail_file": str(detail),
            }
        )

    csv_file = out_dir / f"official_nns_suite_{ts}.csv"
    txt_file = out_dir / f"official_nns_suite_{ts}.txt"

    fields = [
        "network", "source_ref", "batch_size", "best_sub_batch", "state_order",
        "state_batches", "active_states", "latency", "energy", "edp", "detail_file",
    ]
    with csv_file.open("w", encoding="utf-8", newline="\n") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    with txt_file.open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"official_nns_suite run: {ts}\n")
        f.write(f"count={len(rows)}\n")
        f.write(f"details_dir={detail_dir}\n\n")
        for r in rows:
            f.write(f"[{r['network']}]\n")
            f.write(f"  source_ref: {r['source_ref']}\n")
            f.write(f"  batch_size: {r['batch_size']}\n")
            f.write(f"  best_sub_batch: {r['best_sub_batch']}\n")
            f.write(f"  state_batches: {r['state_batches']}\n")
            f.write(f"  active_states: {r['active_states']}\n")
            f.write(f"  latency: {r['latency']}\n")
            f.write(f"  energy: {r['energy']}\n")
            f.write(f"  edp: {r['edp']}\n")
            f.write(f"  detail_file: {r['detail_file']}\n\n")

    print(f"txt={txt_file}")
    print(f"csv={csv_file}")
    print(f"details={detail_dir}")


if __name__ == "__main__":
    main()
