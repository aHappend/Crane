"""Run the actual pinned SET executable with a recorded deterministic SA seed.

The only source change replaces the clock-derived seed with a CLI-provided
constant in a temporary copy of main.cpp. Algorithms and cost model are unchanged.
Results use SET's own cost evaluator, so cross-model ratios require calibration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import tempfile
import time

REVISION = "a7bd73912f58d9fea10fadab693eebd4e6e3054d"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set-root", required=True, type=Path)
    parser.add_argument("--network", default="resnet")
    parser.add_argument("--batch", default=64, type=int)
    parser.add_argument("--mesh", default=4, type=int, help="square mesh side length")
    parser.add_argument("--rounds", default=5, type=int, help="SA rounds per layer; four seeded trials per mode")
    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument("--timeout", default=300, type=float)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if min(args.batch, args.mesh, args.rounds, args.timeout) <= 0 or args.seed < 0:
        parser.error("batch, mesh, rounds and timeout must be positive; seed nonnegative")
    source = args.set_root.resolve()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if revision != REVISION:
        parser.error(f"expected SET revision {REVISION}")
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=source):
        parser.error("SET tracked sources must be unmodified")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    out = args.output_dir.resolve()
    subprocess.run(["make", "-j2"], cwd=source, check=True, stdout=subprocess.DEVNULL)
    original = (source / "src/main.cpp").read_text()
    before = "unsigned seed = std::time(nullptr);"
    after = f"unsigned seed = {args.seed}u;"
    if original.count(before) != 1:
        raise RuntimeError("upstream seed initialization changed")
    patched = original.replace(before, after)
    objects = sorted(p for p in (source / "build/objects").rglob("*.o") if p.name != "main.o")
    configuration = {
        "schema_version": 1, "engine": "upstream_SET", "revision": revision,
        "network": args.network, "batch": args.batch, "mesh": [args.mesh, args.mesh],
        "core": "polar", "noc_bandwidth_gb_s": 24, "cost": "EDP",
        "rounds_per_layer": args.rounds, "seed": args.seed, "parallel_trials": 4,
        "timeout_seconds": args.timeout, "python": platform.python_version(),
        "main_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "patched_main_sha256": hashlib.sha256(patched.encode()).hexdigest(),
        "patch": {"before": before, "after": after},
        "cost_model": "SET Polar hardware evaluator; pJ and cycles at 1GHz",
    }
    with tempfile.TemporaryDirectory(prefix="crane-set-seeded-") as temp:
        patched_main = Path(temp) / "main_seeded.cpp"
        binary = Path(temp) / "stschedule_seeded"
        patched_main.write_text(patched)
        subprocess.run(["g++", "-std=c++17", "-O3", f"-I{source / 'include'}", str(patched_main),
                        *(str(p) for p in objects), "-lpthread", "-o", str(binary)], check=True)
        argv = ["--args", "baseline", args.network, str(args.batch), "polar", str(args.mesh),
                str(args.mesh), str(args.mesh), "24", "1", str(args.rounds), "0"]
        configuration["arguments"] = argv
        start = time.monotonic()
        with (out / "run.log").open("w") as log:
            try:
                completed = subprocess.run([str(binary), *argv], cwd=out, stdout=log,
                                           stderr=subprocess.STDOUT, timeout=args.timeout)
                configuration["status"] = "completed" if completed.returncode == 0 else "failed"
                configuration["returncode"] = completed.returncode
            except subprocess.TimeoutExpired:
                configuration["status"] = "timeout"
        configuration["wall_time_seconds"] = time.monotonic() - start
    metrics = {}
    for path in sorted(out.glob("baseline_*_summary.txt")):
        values = dict(re.findall(r"^([^\n:]+):\s*([-+\deE.]+)\s*$", path.read_text(), re.MULTILINE))
        if "Energy" not in values or "Latency" not in values:
            continue
        energy = float(values["Energy"]) * 1e-12
        latency = float(values["Latency"]) * 1e-9
        metrics[path.name.removeprefix("baseline_").removesuffix("_summary.txt")] = {
            "energy_joules": energy, "latency_seconds": latency, "edp_joule_seconds": energy * latency,
            "native_summary": path.name,
        }
    configuration["results"] = metrics
    if configuration["status"] == "completed" and "SET" not in metrics:
        configuration["status"] = "missing_result"
    (out / "manifest.json").write_text(json.dumps(configuration, indent=2) + "\n")
    print(json.dumps({"status": configuration["status"], "seconds": configuration["wall_time_seconds"],
                      "results": metrics, "output_dir": str(args.output_dir)}, indent=2))
    if configuration["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
