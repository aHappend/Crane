# Running and interpreting experiments

## Installation

Use Python 3.11–3.13 in a fresh virtual environment. The Linux and Windows CI
matrix is defined in [`.github/workflows/ci.yml`](../.github/workflows/ci.yml).
The workflow runs on pull requests and pushes to `main`, and supports manual
dispatch. The same checks can also be run locally with the commands below.

```bash
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux/macOS or
`.\.venv\Scripts\Activate.ps1` in Windows PowerShell, then run:

```bash
python -m pip install -r requirements.txt -c constraints.txt
python -m pip check
python tools/doctor.py
```

`requirements.txt` lists dependencies; `constraints.txt` records the tested
versions, including transitive and development dependencies. Constraints limit
versions but do not cause development tools to be installed by themselves.
SCIP comes through OR-Tools; no separate Gurobi installation is used. `pypdf`
supports working with the bundled paper and is not part of the solver loop.

For PowerShell environments that restrict activation, call
`.\.venv\Scripts\python.exe` directly instead of changing system policy.

## 1. First run: a small schedule

```bash
python example/quickstart.py
```

This uses a synthetic three-layer chain, batch size 8, sub-batch candidates
`[1, 2, 4]`, 16 modeled tiles, a §7.2-style hardware profile, and a flat search.
Both table solvers must be `ortools-scip`. It produces a fresh directory under
`outputs/experiments/` with `summary.json` and `schedule.html`.

For a chosen location (the directory must not already exist):

```bash
python example/quickstart.py --output-dir outputs/experiments/my-first-run
```

The JSON includes the complete configuration, synthetic workload, Python /
NumPy / OR-Tools versions, Git commit and dirty flag, solver names, tables and
metrics with units. HTML can be opened directly in a browser. No download of
weights or data is involved.

## 2. Network-family proxy suites

```bash
python example/run_official_nns_suite.py
python example/run_official_nns_layer_level.py
```

Each entrypoint runs 12 hand-authored network-family proxies. The first merges
linear blocks; the second keeps each proxy node as a block. The workload
definitions live in `official_specs()`, not in a C++ parser. These runs are useful
for exploring scheduling behavior; they are not paper-scale accuracy checks.

Each writes `summary.txt`, `summary.csv` and a `details/` directory into its own
timestamped `outputs/experiments/official_nns_*` directory. Per-network details
include solver names and table values. Older revisions of the merged suite
wrote to `outputs/runs/`; those files remain archived.

## 3. Transformer granularity and hierarchy

Inspect the available options first:

```bash
python example/compare_transformer_granularity.py --help
```

An inference-oriented comparison using the §7.2 hardware helper and paper-style
K1/K2 ratios can be launched with:

```bash
python example/compare_transformer_granularity.py --num-pes 16 --paper-hw-7-2 --top-k1-ratio 0.5 --top-k2-ratio 0.2 --verbose-progress
```

Add `--hierarchical --hier-depth 2 --hier-iters 2 --hier-theta 0.02` to enable
structure refinement. `--all-sub-batch-factors` expands the candidate set.
This script uses a manually expanded 471-node chain; matching hardware settings
does not make that workload equivalent to the paper's Transformer-Large model.

Other retained entrypoints:

| Script | Role |
| --- | --- |
| `run_transformer_min_layer_block_experiment.py` | Compare merge sizes on the 471-node chain; configuration is in `run_candidate()` |
| `compare_transformer_stage_layer_with_merge.py` | Stage/layer comparison with merging and optional hierarchical settings |
| `compare_all_networks_stage_vs_layer.py` | Proxy-family granularity comparison with per-run `--timeout-sec` |
| `compare_all_networks_same_source_merge.py` | Proxy-family granularity/merge comparison with `--timeout-sec` and optional hardware/hierarchy settings |
| `resnet50_test.py`, `advanced_networks_test.py` | Older synthetic scheduling demonstrations; these are not unit tests |

The large experiments are exploratory entrypoints. The test suite validates their imports
and applicable help commands, not a full run of every large configuration.
Most MILPs have no solver time limit; the comparison scripts' timeout options
are process-level limits specific to those scripts.

## 4. Training scheduling and recomputation

```bash
python example/run_transformer_training_repro.py --help
python example/run_transformer_training_repro.py --num-pes 2 --batch-size 128 --candidate-sub-batches 4,8,16,32 --verbose-progress
```

The default uses §7.3-style hardware settings, a 32768 MB DRAM capacity and the
synthetic Transformer workload. Results are scheduling estimates for the
FW / BW1 / BW2 phases; no model is trained. This 471-node experiment can be
expensive. The test suite uses a separate two-block training case for a fast
check of the phase plumbing.

Outputs include `summary.txt`, phase rows in `summary.csv`, and
`fw_detail.txt` / `bw1_detail.txt` / `bw2_detail.txt` with corresponding HTML
schedules. Total training EDP is `(sum phase latency) × (sum phase energy)`,
not the sum of phase EDP values.

## Reading and preserving results

ScT entries are cumulative completed sub-batch counts. MeT entries are cutoff
indices; they are not MB occupancy. See [architecture and units](ARCHITECTURE.md).
Always record whether the run uses normalized hardware or a paper hardware
helper. Legacy text output rounds many values to six decimal places; prefer
full-precision numeric records for quantitative comparisons.

The quick start records a manifest automatically. Older experiment scripts have
different output schemas and do not all record the full environment. Preserve
the command, commit, dirty diff if any, complete config, dependency versions,
source workload and raw outputs alongside any result you intend to compare.

Generated experiments are ignored by Git. Share selected outputs as CI artifacts
or a documented benchmark artifact, with their configuration and provenance.
Keep historical runs separate from results regenerated after code changes.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| Missing `numpy` / `ortools`, or import error | Check the active Python executable and install with the same `python -m pip` command. Do not reuse a checked-out virtual environment. |
| `OR-Tools SCIP solver is not available` | Run `tools/doctor.py`; use the pinned OR-Tools wheel on a supported Python/platform. |
| `No feasible ScT candidate found` | Enable verbose progress. Check batch divisibility, candidate sizes, active-state constraints and dependency settings. |
| `No feasible MeT candidate found` | Inspect SRAM/DRAM capacity, output-volume units and candidate-specific errors. Do not silently enable fallback for a paper comparison. |
| Long solve with little output | Run the quick start first; use progress options, fewer candidates or a script with a process timeout. |
| Different schedules between environments | Compare solver and dependency versions, configs and objective values. Tied solutions can yield different tables. |
| `FileExistsError` from quick start | Choose a new output directory; previous results are intentionally preserved. |

For a bug report, include `python tools/doctor.py`, your command and commit, the
smallest reproducing workload, and the relevant traceback or solver log.
