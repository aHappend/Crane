"""Run a small synthetic chain through the real ScT and MeT solvers."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).absolute().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from example.schedule_html import write_schedule_html
from model.layer import Layer
from project_paths import repo_rel
from scheduler.block import Block
from scheduler.hardware_profile import paper_7_2_search_params
from search.scheduler_search import SearchConfig, search_schedule


def build_demo() -> tuple[list[Block], SearchConfig]:
    """Three synthetic layers; no datasets, weights or accelerator required."""
    layers = [
        Layer.with_map_dims("stem", 2e6, 0.25, (16, 16, 16, 16)),
        Layer.with_map_dims("hidden", 4e6, 0.50, (32, 16, 16, 16)),
        Layer.with_map_dims("head", 1e6, 0.125, (16, 16, 8, 8)),
    ]
    for parent, child in zip(layers, layers[1:]):
        parent.connect_to(child)
    config = SearchConfig(
        batch_size=8,
        candidate_sub_batches=[1, 2, 4],
        num_pes=16,
        **paper_7_2_config(),
        enable_chain_block_merge=False,
        derive_recursive_traces=False,
        enable_structure_refinement=False,
        max_hierarchy_depth=1,
        allow_solver_fallback=False,
        canonical_fastpath=False,
    )
    return [Block(layer.name, layers=[layer]) for layer in layers], config


def paper_7_2_config() -> dict[str, float]:
    params = paper_7_2_search_params(num_pes=16)
    # The hardware helper also exports this compatibility-only key.
    params.pop("traffic_energy_per_unit")
    return params


def _revision() -> dict[str, object]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=ROOT, text=True, stderr=subprocess.DEVNULL,
        ).strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="new directory for summary.json and schedule.html")
    args = parser.parse_args()
    timestamp = datetime.now(timezone.utc)
    out_dir = args.output_dir or (
        ROOT / "outputs" / "experiments" / f"quickstart_{timestamp:%Y%m%d_%H%M%S_%f}"
    )
    # Preserve previous results; a new run always has its own directory.
    out_dir.mkdir(parents=True, exist_ok=False)
    blocks, config = build_demo()
    result = search_schedule(blocks, config)
    summary = {
        "schema_version": 1,
        "experiment": "synthetic_three_layer_chain",
        "evidence_scope": "workflow_smoke_test",
        "created_at_utc": timestamp.isoformat(),
        "repository": _revision(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.system(),
            "numpy": version("numpy"),
            "ortools": version("ortools"),
        },
        "hardware_profile": "paper_7_2",
        "config": asdict(config),
        "workload": [
            {"name": layer.name, "flops": layer.flops, "output_size_mb": layer.output_size,
             "map_dims": layer.map_dims, "parents": [p.name for p in layer.parents]}
            for block in blocks for layer in block.iter_layers()
        ],
        "result": {
            "best_sub_batch": result.best_sub_batch,
            "sct_solver": result.sct_solver_name,
            "met_solver": result.met_solver_name,
            "latency_seconds": result.total_latency,
            "energy_joules": result.total_energy,
            "edp_joule_seconds": result.total_edp,
            "scheduled_blocks": result.scheduled_blocks,
            "block_dependencies": result.block_dependencies,
            "states": result.state_order,
            "state_batches": result.milp_solution.state_batches,
            "sct": result.sct.table.tolist(),
            "met_s": result.met.sram.tolist(),
            "met_d": result.met.dram.tolist(),
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_schedule_html(
        out_dir / "schedule.html",
        title="Crane quickstart — synthetic three-layer chain",
        meta={"evidence_scope": "workflow smoke test", "best_sub_batch": result.best_sub_batch,
              "sct_solver": result.sct_solver_name, "met_solver": result.met_solver_name},
        scheduled_blocks=result.scheduled_blocks,
        state_order=result.state_order,
        state_categories=result.state_categories,
        state_batches=result.milp_solution.state_batches,
        state_active_blocks=result.state_active_blocks,
        sct=result.sct.table.tolist(),
        met_s=result.met.sram.tolist(),
        met_d=result.met.dram.tolist(),
    )
    print(f"ScT solver: {result.sct_solver_name}; MeT solver: {result.met_solver_name}")
    print(f"Best sub-batch: {result.best_sub_batch}; states: {len(result.state_order)}")
    print(f"Estimated latency: {result.total_latency:.6e} s")
    print(f"Estimated energy: {result.total_energy:.6e} J")
    print(f"Estimated EDP: {result.total_edp:.6e} J*s")
    print(f"Outputs: {repo_rel(out_dir, ROOT)}")


if __name__ == "__main__":
    main()
