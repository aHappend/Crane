# Running the reproduction experiments

## Install and check the environment

Use Python 3.11–3.13 and run commands from the repository root:

```bash
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux/macOS or
`.\.venv\Scripts\Activate.ps1` in Windows PowerShell. Then:

```bash
python -m pip install -r requirements.txt -c constraints.txt
python -m pip check
python tools/doctor.py
python example/quickstart.py
```

The core runtime and recorded cost profiles do not require a C++ compiler,
GPU, trained weights, dataset download or commercial solver service.
On PowerShell you can call `.\.venv\Scripts\python.exe` directly if activation
is restricted.

## Explore the saved demo

Open `docs/demo/index.html` directly in a browser. It embeds the saved results
and needs no internet access. It supports model/mode filters, state playback,
activation-storage inspection, training memory outcomes and seeded SET summaries.
The controls select existing records; they do not submit live solver jobs.

For HTTP preview:

```bash
python -m http.server 8000 --bind 127.0.0.1 --directory docs/demo
```

If this runs on an SSH server, use `ssh -L 8000:127.0.0.1:8000 <your-host>` from
your computer and browse `http://127.0.0.1:8000` there.

## Rerun the recorded inference and training suites

Choose new output directories; existing directories are refused:

```bash
python -m experiments.run --config experiments/configs/native_core_inference.json --output-dir outputs/experiments/my-inference
python -m experiments.run --config experiments/configs/training_memory.json --output-dir outputs/experiments/my-training
```

The inference matrix has 10 cases using real ResNet-50, VGG-19, GoogLeNet and SET
Transformer-cell graphs. Nested and serial modes share the same recorded Polar
core mappings. The search uses the explicitly listed sub-batch candidates and a
fixed initial hierarchy; it is not an exhaustive whole-network optimum.

The training matrix has 9 cases with batch 256, two tiles and DRAM capacities of
64, 256 and 32768 MB. Its serial uniform-cohort policy produces seven feasible
outcomes and two policy-specific infeasibilities. Expected infeasible cases are
explicitly marked in the config. Unexpected errors, timeouts or mismatched
outcomes make the suite exit nonzero.

The reference machine completed individual nested inference cases in roughly
6–38 seconds and serial cases in roughly 0.1–1.2 seconds. These are observations
from the recorded environment, not runtime guarantees.

### Configuration

A suite JSON contains `defaults` and an explicit `cases` list. Important fields:

| Field | Meaning |
| --- | --- |
| `model` | A name from `workloads.set_models.available_models()` |
| `mode` | `nested`, `serial`, `flat` or the checked `training` reference |
| `batch`, `sub_batches`, `tiles` | Total samples, enumerated factors and tile count |
| `fanout`, `depth` | Initial containment fanout and maximum expansion depth |
| `core_profile` | Recorded SET profile directory; nested/serial inference only |
| `solver_seconds` | Per-SCIP-call limit |
| `timeout_seconds` | Hard deadline for the entire isolated case process |
| `sram_mb`, `dram_mb` | Explicit memory overrides |
| `expected_status` | `completed` by default; documented negative cases use `infeasible` |

A profile must cover the requested network, batch factors and tile counts.
Missing or mismatched profiles fail explicitly. Full profile-based runs require
all composite blocks to expand to individual layers.

Each case writes `case.json`, `result.json`, `run.log`, and, for inference,
`hierarchy.json`, `candidate_failures.json` and `schedule.html`. The suite manifest
records hashes and environments. Inference table occupancy is scoped to top-level
boundary buffers; child budgets are in hierarchy records.

## Regenerate real networks and native core profiles

These optional steps require Linux, Git, Make and a C++17 compiler. The Python
runtime can use the checked-in exports without them.

```bash
git clone https://github.com/SET-Scheduling-Project/SET-ISCA2023.git /tmp/set-reference
git -C /tmp/set-reference checkout a7bd73912f58d9fea10fadab693eebd4e6e3054d
python tools/import_set_networks.py --set-root /tmp/set-reference --out outputs/experiments/set_networks.json
python tools/profile_set_core.py --set-root /tmp/set-reference --network resnet --max-batch 64 --max-tiles 16 --output-dir outputs/experiments/my-resnet-profile
```

The tools require the pinned, unmodified upstream source. The network exporter
uses compiled objects instead of regex-parsing or inventing graph metadata.
The core exporter calls the original Polar mapper for each layer/batch/tile
combination. Its manifest documents included and excluded cost terms.

## Run the actual SET baseline

```bash
python tools/run_set_baseline.py --set-root /tmp/set-reference --network resnet --batch 64 --mesh 4 --rounds 20 --seed 7 --timeout 180 --output-dir outputs/experiments/my-set-run
python -m experiments.native_set_suite --set-root /tmp/set-reference --rounds 20 --seeds 7,19,43 --output-dir outputs/experiments/my-set-suite
```

The suite runs four networks over three fixed seeds. Each native search uses four
internal SA trials and the requested rounds per layer. A temporary source copy
replaces only the clock-derived random seed; the patch, generated-source hashes,
arguments, native summaries and logs are preserved. The checkout is unmodified.
Compilation is excluded from the recorded search wall time.

Native SET's outer placement/traffic evaluator differs from Python's analytical
one. Report raw values and budgets; do not treat their ratio as a reproduced
paper speedup without a matched-evaluator calibration.

## Rebuild figures and demo

Install the recorded plotting environment:

```bash
python -m pip install -r requirements-analysis.txt -c constraints-analysis.txt
```

From fresh run directories:

```bash
python -m experiments.report --inference outputs/experiments/my-inference --training outputs/experiments/my-training --set-baselines outputs/experiments/my-set-suite --output-dir outputs/experiments/my-report --demo outputs/experiments/my-report/index.html
```

Or regenerate entirely from the checked-in raw evidence, without rerunning SET:

```bash
python -m experiments.report --inference experiments/results/reproduction_20260921/raw/inference --training experiments/results/reproduction_20260921/raw/training --set-baselines experiments/results/reproduction_20260921/raw/native_set --output-dir outputs/experiments/rebuilt-report --demo outputs/experiments/rebuilt-report/index.html
```

The generator verifies recorded result/workload hashes, accepts compressed raw
JSON, and produces a CSV, report JSON, Markdown report, SVG/PNG figures and a
self-contained HTML demo.

## Tests and interpretation

```bash
python -m pip install -r requirements-dev.txt -c constraints.txt
python -m ruff check .
python -m pytest -q
```

Tests include independent integer/state enumeration, workload conservation,
interval identity, real graph metadata, nested resource/batch composition,
Figure-6 cohort accounting and experiment artifact generation. CI runs Linux
and Windows checks; long matrices run separately with process deadlines.

The old `example/official_nns*` names refer to historical proxy workloads. Other
older example scripts, including the experimental MILP Transformer training
entrypoint, remain available for comparison but are not the recorded evidence
path described here. See [REPRODUCTION.md](REPRODUCTION.md) before extending claims.

Common failures: use the same Python for pip and execution; run `tools/doctor.py`
for SCIP availability; use valid batch factors; check memory/units and profile
coverage; choose a fresh output directory. A time-limited feasible result is not
an optimality certificate. Preserve the full manifest when reporting a problem.
