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


def profile_specs() -> dict:
    # flops in GFLOPs proxy, output_size in MB proxy
    return {
        # official src/nns networks
        "alexnet": {
            "batch_size": 64,
            "blocks": [
                ("stem", [("conv1", 0.72, 18.0), ("pool1", 0.10, 9.0)]),
                ("mid", [("conv2", 1.10, 12.0), ("conv3", 1.35, 10.0)]),
                ("deep", [("conv4", 1.30, 8.5), ("conv5", 1.20, 8.0)]),
                ("head", [("fc6", 0.45, 4.0), ("fc7", 0.45, 4.0), ("fc8", 0.08, 1.2)]),
            ],
        },
        "darknet19": {
            "batch_size": 64,
            "blocks": [
                ("stage1", [("conv1", 0.6, 20.0), ("conv2", 0.9, 16.0)]),
                ("stage2", [("conv3", 1.4, 14.0), ("conv4", 1.8, 12.0)]),
                ("stage3", [("conv5", 2.0, 10.0), ("conv6", 2.3, 9.0)]),
                ("head", [("conv7", 1.2, 6.0), ("pred", 0.5, 3.5)]),
            ],
        },
        "densenet": {
            "batch_size": 64,
            "blocks": [
                ("dense1", [("db1_l1", 1.1, 24.0), ("db1_l2", 1.2, 24.0), ("trans1", 0.4, 18.0)]),
                ("dense2", [("db2_l1", 1.5, 20.0), ("db2_l2", 1.6, 20.0), ("trans2", 0.5, 15.0)]),
                ("dense3", [("db3_l1", 1.8, 16.0), ("db3_l2", 2.0, 16.0), ("trans3", 0.6, 12.0)]),
                ("head", [("classifier", 0.4, 2.0)]),
            ],
        },
        "gnmt": {
            "batch_size": 128,
            "blocks": [
                ("enc_embed", [("src_embed", 0.8, 28.0)]),
                ("encoder", [("enc_lstm1", 3.8, 30.0), ("enc_lstm2", 3.6, 30.0), ("enc_lstm3", 3.5, 30.0)]),
                ("decoder", [("dec_lstm1", 4.0, 32.0), ("dec_lstm2", 3.7, 32.0)]),
                ("attn_head", [("attn", 1.6, 24.0), ("softmax", 0.7, 8.0)]),
            ],
        },
        "googlenet": {
            "batch_size": 64,
            "blocks": [
                ("stem", [("conv1", 0.9, 18.0), ("pool1", 0.2, 10.0)]),
                ("inception3", [("inc3a", 2.0, 14.0), ("inc3b", 2.1, 14.0)]),
                ("inception4", [("inc4a", 2.5, 12.0), ("inc4b", 2.6, 12.0), ("inc4c", 2.6, 12.0)]),
                ("head", [("inc5", 2.1, 9.0), ("fc", 0.5, 2.0)]),
            ],
        },
        "incep_resnet": {
            "batch_size": 64,
            "blocks": [
                ("stem", [("conv_stem", 1.1, 20.0)]),
                ("block_a", [("ir_a1", 3.0, 16.0), ("ir_a2", 3.0, 16.0)]),
                ("block_b", [("ir_b1", 3.5, 14.0), ("ir_b2", 3.6, 14.0)]),
                ("block_c", [("ir_c1", 4.0, 12.0), ("ir_c2", 4.1, 12.0), ("fc", 0.7, 2.4)]),
            ],
        },
        "llm": {
            "batch_size": 128,
            "blocks": [
                ("embedding", [("token_embed", 2.5, 50.0), ("rope", 0.4, 50.0)]),
                ("decoder_stack_1", [("attn1", 8.0, 56.0), ("mlp1", 12.5, 58.0)]),
                ("decoder_stack_2", [("attn2", 7.8, 56.0), ("mlp2", 12.0, 58.0)]),
                ("decoder_stack_3", [("attn3", 7.6, 55.0), ("mlp3", 11.8, 57.0)]),
                ("lm_head", [("norm", 0.6, 30.0), ("head", 2.0, 12.0)]),
            ],
        },
        "pnasnet": {
            "batch_size": 64,
            "blocks": [
                ("stem", [("conv1", 0.9, 18.0)]),
                ("cell1", [("sepconv1", 2.4, 14.0), ("sepconv2", 2.5, 14.0)]),
                ("cell2", [("sepconv3", 2.8, 12.0), ("sepconv4", 2.9, 12.0)]),
                ("cell3", [("sepconv5", 3.1, 10.0), ("pool_proj", 0.8, 8.0)]),
                ("head", [("fc", 0.5, 2.0)]),
            ],
        },
        "resnet": {
            "batch_size": 64,
            "blocks": [
                ("stem", [("conv1", 1.0, 20.0), ("pool1", 0.2, 12.0)]),
                ("stage2", [("res2_1", 3.0, 16.0), ("res2_2", 2.8, 16.0)]),
                ("stage3", [("res3_1", 4.2, 14.0), ("res3_2", 4.0, 14.0)]),
                ("stage4", [("res4_1", 5.0, 12.0), ("res4_2", 4.8, 12.0)]),
                ("head", [("fc", 0.6, 2.2)]),
            ],
        },
        "transformer": {
            "batch_size": 128,
            "blocks": [
                ("embed", [("embed", 1.7, 38.0)]),
                ("enc1", [("attn1", 4.0, 40.0), ("mlp1", 6.4, 43.0)]),
                ("enc2", [("attn2", 3.9, 40.0), ("mlp2", 6.2, 43.0)]),
                ("enc3", [("attn3", 3.8, 40.0), ("mlp3", 6.0, 42.0)]),
                ("head", [("proj", 1.9, 14.0)]),
            ],
        },
        "vgg": {
            "batch_size": 64,
            "blocks": [
                ("conv12", [("conv1", 0.8, 22.0), ("conv2", 1.2, 20.0)]),
                ("conv34", [("conv3", 1.8, 16.0), ("conv4", 2.1, 16.0)]),
                ("conv56", [("conv5", 2.5, 13.0), ("conv6", 2.8, 13.0)]),
                ("head", [("fc6", 0.9, 6.0), ("fc7", 0.9, 6.0), ("fc8", 0.2, 2.0)]),
            ],
        },
        "zfnet": {
            "batch_size": 64,
            "blocks": [
                ("stem", [("conv1", 0.6, 16.0), ("pool1", 0.1, 8.0)]),
                ("mid", [("conv2", 1.0, 12.0), ("conv3", 1.2, 10.0)]),
                ("deep", [("conv4", 1.2, 8.0), ("conv5", 1.1, 7.0)]),
                ("head", [("fc6", 0.4, 3.0), ("fc7", 0.4, 3.0), ("fc8", 0.1, 1.0)]),
            ],
        },
        # supplemented variants requested by user
        "attention_group_builder": {
            "batch_size": 96,
            "blocks": [
                ("embed", [("token_embed", 1.2, 36.0), ("pos_embed", 0.3, 36.0)]),
                ("attn_group_a", [("qkv_proj_a", 2.2, 34.0), ("mh_attn_a", 3.4, 38.0)]),
                ("attn_group_b", [("qkv_proj_b", 2.2, 34.0), ("mh_attn_b", 3.4, 38.0)]),
                ("mlp_group", [("mlp_up", 4.0, 40.0), ("mlp_down", 3.8, 36.0)]),
            ],
        },
        "transformer_fused": {
            "batch_size": 128,
            "blocks": [
                ("embed", [("embed", 1.9, 40.0)]),
                ("enc_fused_1", [("fused_attn_mlp1", 9.5, 46.0)]),
                ("enc_fused_2", [("fused_attn_mlp2", 9.1, 46.0)]),
                ("head", [("proj", 2.1, 14.0)]),
            ],
        },
        "transformer_grouped": {
            "batch_size": 128,
            "blocks": [
                ("embed", [("embed", 1.8, 39.0)]),
                ("enc_group_1", [("attn_g1", 4.6, 44.0), ("mlp_g1", 6.9, 45.0)]),
                ("enc_group_2", [("attn_g2", 4.5, 44.0), ("mlp_g2", 6.8, 45.0)]),
                ("head", [("proj", 2.0, 14.0)]),
            ],
        },
        "transformer_semifused": {
            "batch_size": 128,
            "blocks": [
                ("embed", [("embed", 1.8, 39.0)]),
                ("enc_sf_1", [("attn1", 4.4, 43.0), ("fused_mlp1", 7.0, 45.0)]),
                ("enc_sf_2", [("attn2", 4.3, 43.0), ("fused_mlp2", 6.9, 45.0)]),
                ("head", [("proj", 2.0, 14.0)]),
            ],
        },
        "transformer_semigrouped": {
            "batch_size": 128,
            "blocks": [
                ("embed", [("embed", 1.8, 39.0)]),
                ("enc_sg_1", [("grouped_attn1", 4.2, 42.0), ("mlp1", 6.7, 44.0)]),
                ("enc_sg_2", [("grouped_attn2", 4.1, 42.0), ("mlp2", 6.6, 44.0)]),
                ("head", [("proj", 2.0, 14.0)]),
            ],
        },
    }


def build_blocks(block_specs):
    blocks = []
    prev_last = None
    for block_name, layers_spec in block_specs:
        layers = []
        for name, gflops, out_mb in layers_spec:
            layer = Layer(name=name, flops=float(gflops) * 1e9, output_size=float(out_mb))
            if prev_last is not None:
                prev_last.connect_to(layer)
            prev_last = layer
            layers.append(layer)
        blocks.append(Block(name=block_name, layers=layers))
    return blocks


def candidate_sub_batches(batch_size: int):
    vals = [2, 4, 8, 16, 32]
    return [v for v in vals if v <= batch_size and batch_size % v == 0]


def run_one(name: str, spec: dict, official_set: set[str]):
    blocks = build_blocks(spec["blocks"])
    batch_size = int(spec["batch_size"])
    candidates = candidate_sub_batches(batch_size)

    cfg = SearchConfig(
        batch_size=batch_size,
        candidate_sub_batches=candidates,
        sram_capacity=15000.0,
        dram_capacity=30000.0,
        num_pes=4,
        min_active_states=4,
        min_batch_if_active=1,
        max_state_share=0.45,
    )
    result = search_schedule(blocks, cfg)
    active_states = sum(1 for v in result.milp_solution.state_batches if v > 0)

    source_kind = "official_src_nns" if name in official_set else "supplemented_variant"
    source_ref = f"src/nns/{name}.cpp" if name in official_set else "supplemented_profile"

    return {
        "network": name,
        "source_kind": source_kind,
        "source_ref": source_ref,
        "batch_size": batch_size,
        "best_sub_batch": result.best_sub_batch,
        "state_order": " -> ".join(result.state_order),
        "state_batches": ",".join(str(v) for v in result.milp_solution.state_batches),
        "active_states": active_states,
        "latency": f"{result.total_latency:.6f}",
        "energy": f"{result.total_energy:.6f}",
        "edp": f"{result.total_edp:.6f}",
        "solver": result.milp_solution.solver_name,
    }


def main():
    specs = profile_specs()
    official_cpp = {p.stem for p in (ROOT / "src" / "nns").glob("*.cpp")}

    rows = []
    for name in sorted(specs.keys()):
        rows.append(run_one(name, specs[name], official_cpp))

    out_dir = ROOT / "outputs" / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt = out_dir / f"supplemented_nns_suite_{ts}.txt"
    csvf = out_dir / f"supplemented_nns_suite_{ts}.csv"

    with txt.open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"supplemented_nns_suite run: {ts}\n")
        f.write(f"count={len(rows)}\n\n")
        for row in rows:
            f.write(f"[{row['network']}]\n")
            f.write(f"  source_kind: {row['source_kind']}\n")
            f.write(f"  source_ref: {row['source_ref']}\n")
            f.write(f"  best_sub_batch: {row['best_sub_batch']}\n")
            f.write(f"  state_batches: {row['state_batches']}\n")
            f.write(f"  active_states: {row['active_states']}\n")
            f.write(f"  latency: {row['latency']}\n")
            f.write(f"  energy: {row['energy']}\n")
            f.write(f"  edp: {row['edp']}\n")
            f.write(f"  solver: {row['solver']}\n\n")

    fields = [
        "network", "source_kind", "source_ref", "batch_size", "best_sub_batch",
        "state_order", "state_batches", "active_states", "latency", "energy",
        "edp", "solver",
    ]
    with csvf.open("w", encoding="utf-8", newline="\n") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"txt={txt}")
    print(f"csv={csvf}")


if __name__ == "__main__":
    main()
