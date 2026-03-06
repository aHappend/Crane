import csv
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from example.run_official_nns_suite import (
    official_specs,
    build_layer_blocks,
    candidates,
    fmt_matrix,
    _delta_from_cumulative,
    _block_layers_from_name,
)
from search.scheduler_search import SearchConfig, search_schedule


def run_one(name, spec):
    blocks = build_layer_blocks(spec["layers"])
    cfg = SearchConfig(
        batch_size=int(spec["batch_size"]),
        candidate_sub_batches=candidates(int(spec["batch_size"])),
        sram_capacity=15000.0,
        dram_capacity=30000.0,
        num_pes=4,
        enable_chain_block_merge=False,
        max_layers_per_block=1,
        min_layers_per_block=1,
        min_active_states=4,
        min_batch_if_active=1,
        max_state_share=0.45,
    )
    result = search_schedule(blocks, cfg)

    sct = result.sct.table
    met_s = result.met.sram
    met_d = result.met.dram
    delta = _delta_from_cumulative(sct)

    block_names = list(result.scheduled_blocks)
    block_to_state_map = []
    for j, bname in enumerate(block_names):
        pairs = []
        for i, d in enumerate(delta):
            if d[j] > 0:
                pairs.append(f"{result.state_order[i]}:+{int(d[j])}")
        block_to_state_map.append((bname, " ".join(pairs) if pairs else "none"))

    state_lines = []
    for i, (sname, cat, ape, sb, active_blocks) in enumerate(
        zip(
            result.state_order,
            result.state_categories,
            result.active_pes,
            result.milp_solution.state_batches,
            result.state_active_blocks,
        )
    ):
        active_names = [block_names[idx] for idx in active_blocks if 0 <= idx < len(block_names)]
        pe_span = f"PE0..PE{max(0, ape - 1)}"
        state_lines.append(
            f"  idx={i} name={sname} category={cat} active_pe={ape} pe_span={pe_span} "
            f"active_blocks={active_names} assigned_sub_batches={sb}"
        )

    return result, sct, met_s, met_d, delta, block_names, block_to_state_map, state_lines


def main():
    specs = official_specs()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    base_dir = ROOT / "outputs" / "experiments"
    run_dir = base_dir / f"official_nns_layer_level_{ts}"
    details_dir = run_dir / "details"
    details_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in sorted(specs.keys()):
        result, sct, met_s, met_d, delta, block_names, block_to_state_map, state_lines = run_one(name, specs[name])

        detail = details_dir / f"{name}.txt"
        with detail.open("w", encoding="utf-8", newline="\n") as f:
            f.write(f"network={name}\n")
            f.write(f"mode=layer_level\n")
            f.write(f"source_ref={specs[name]['source_ref']}\n")
            f.write(f"batch_size={specs[name]['batch_size']}\n")
            f.write(f"best_sub_batch={result.best_sub_batch}\n")
            f.write(f"num_blocks={len(block_names)}\n")
            f.write(f"num_states={len(result.state_order)}\n")
            f.write(f"sct_solver={result.sct_solver_name}\n")
            f.write(f"met_solver={result.met_solver_name}\n")
            f.write(f"compute_latency={result.compute_latency:.6f}\n")
            f.write(f"compute_energy={result.compute_energy:.6f}\n")
            f.write(f"memory_latency={result.memory_latency:.6f}\n")
            f.write(f"memory_energy={result.memory_energy:.6f}\n")
            f.write(f"latency={result.total_latency:.6f}\n")
            f.write(f"energy={result.total_energy:.6f}\n")
            f.write(f"edp={result.total_edp:.6f}\n\n")

            f.write("states\n")
            for line in state_lines:
                f.write(line + "\n")

            f.write("\nblocks(columns in ScT/MeT)\n")
            for j, bname in enumerate(block_names):
                f.write(f"  col={j} block={bname}\n")

            f.write("\nblock_dependencies(parent->child)\n")
            for p, c in result.block_dependencies:
                p_name = block_names[p] if 0 <= p < len(block_names) else str(p)
                c_name = block_names[c] if 0 <= c < len(block_names) else str(c)
                f.write(f"  ({p},{c}) {p_name} -> {c_name}\n")

            f.write("\nScT_cumulative(rows=state, cols=block)\n")
            f.write(fmt_matrix(sct) + "\n")

            f.write("\nScT_delta_per_state(rows=state, cols=block)\n")
            f.write(fmt_matrix(delta) + "\n")

            f.write("\nMeT_S(rows=state, cols=block)\n")
            f.write(fmt_matrix(met_s) + "\n")

            f.write("\nMeT_D(rows=state, cols=block)\n")
            f.write(fmt_matrix(met_d) + "\n")

            f.write("\nblock_to_state_mapping\n")
            for bname, mapping in block_to_state_map:
                f.write(f"  {bname}: {mapping}\n")

        rows.append(
            {
                "network": name,
                "mode": "layer_level",
                "source_ref": specs[name]["source_ref"],
                "batch_size": specs[name]["batch_size"],
                "best_sub_batch": result.best_sub_batch,
                "num_blocks": len(block_names),
                "num_states": len(result.state_order),
                "state_batches": ",".join(str(x) for x in result.milp_solution.state_batches),
                "active_states": sum(1 for x in result.milp_solution.state_batches if x > 0),
                "latency": f"{result.total_latency:.6f}",
                "energy": f"{result.total_energy:.6f}",
                "edp": f"{result.total_edp:.6f}",
                "detail_file": str(detail),
            }
        )

    csv_file = run_dir / "summary.csv"
    txt_file = run_dir / "summary.txt"

    fields = [
        "network",
        "mode",
        "source_ref",
        "batch_size",
        "best_sub_batch",
        "num_blocks",
        "num_states",
        "state_batches",
        "active_states",
        "latency",
        "energy",
        "edp",
        "detail_file",
    ]
    with csv_file.open("w", encoding="utf-8", newline="\n") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    with txt_file.open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"official_nns_layer_level run: {ts}\n")
        f.write(f"count={len(rows)}\n")
        f.write(f"details_dir={details_dir}\n\n")
        for r in rows:
            f.write(f"[{r['network']}]\n")
            f.write(f"  mode: {r['mode']}\n")
            f.write(f"  source_ref: {r['source_ref']}\n")
            f.write(f"  batch_size: {r['batch_size']}\n")
            f.write(f"  best_sub_batch: {r['best_sub_batch']}\n")
            f.write(f"  num_blocks: {r['num_blocks']}\n")
            f.write(f"  num_states: {r['num_states']}\n")
            f.write(f"  state_batches: {r['state_batches']}\n")
            f.write(f"  active_states: {r['active_states']}\n")
            f.write(f"  latency: {r['latency']}\n")
            f.write(f"  energy: {r['energy']}\n")
            f.write(f"  edp: {r['edp']}\n")
            f.write(f"  detail_file: {r['detail_file']}\n\n")

    print(f"run_dir={run_dir}")
    print(f"txt={txt_file}")
    print(f"csv={csv_file}")
    print(f"details={details_dir}")


if __name__ == "__main__":
    main()

