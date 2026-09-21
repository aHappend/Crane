# Architecture and data conventions

## Recommended execution path

```text
compiled SET network metadata + recorded native core mappings
  → load_set_model(...): real Layer DAG
  → balanced_blocks(...): initial containment hierarchy
  → NestedSearch.run(...)
      → choose parent sub-batch
      → compute child costs for exactly that sub-batch and assigned tiles
      → compare serial micro-batch and canonical pipeline plans
      → optimize MeT using exact tensor-index intervals
      → validate progress, capacity and tile constraints
  → experiment runner: result JSON, logs, configuration and provenance
  → report generator: verified archive, CSV, scientific figures and offline demo
```

The lightweight `example/quickstart.py` still runs the general SCIP path directly.
The recommended real-network suites are in `experiments/configs/`.

## Module ownership

| Module | Responsibility |
| --- | --- |
| `model/layer.py` | Identity-based graph nodes, per-sample operations/activations, input/weight MB and mapping dimensions |
| `model/dag_parser.py` | Build and validate graph specs, including generator input and cycles/duplicate names |
| `workloads/set_models.py` | Load compiled graph metadata and emit its manifest |
| `workloads/hierarchy.py` | Deterministic bounded-fanout containment; original nodes/edges are preserved |
| `scheduler/block.py` | Ownership, crossing dependencies, boundary tensor volumes and aggregation |
| `scheduler/paper_milp.py` | General ScT program, canonical exact reduction and compute coefficients |
| `scheduler/optimization.py` | Scaled exact integer-product objective, limits and termination reports |
| `scheduler/memory_table.py` | Memory cutoffs, capacity constraints and monotone traffic optimization |
| `scheduler/traffic.py` | Shared interval identities and traffic accounting |
| `search/nested_search.py` | Parent-sub-batch macro composition and cached child plans |
| `search/scheduler_search.py` | Flat candidate search, compatibility API and experimental legacy training/helpers |
| `search/training_cohorts.py` | Checked serial uniform-cohort training reference |
| `cost_model/set_profile.py` | Hash-checked native SET core-profile lookup |
| `experiments/run.py` | Process-isolated experiment cases, deadlines, checks and raw records |
| `experiments/report.py` | Evidence verification, archive, figures and demo generation |

`search_schedule()` routes default hierarchical work to the new nested engine.
Old private refinement helpers remain for compatibility/research history; the
new default does not use their free-boundary retry behavior.

## Units and tensor identity

| Field | Convention |
| --- | --- |
| `Layer.flops` | Operations per sample; exported convolution MACs count as two operations |
| `Layer.output_size`, `Layer.input_size` | Activation / external-input MB per sample |
| `Layer.weight_size` | Static weight MB; never multiplied by sample count |
| Mapping dimensions | Exported data use `(output_channels, batch, height, width)` |
| Bandwidth | MB/s |
| `compute_power_per_tile` | Operations/s despite the historical name “power” |
| Compute energy | Joules/operation in the analytical model |
| Traffic energy | Joules/MB |
| Reported latency / energy / EDP | Seconds / joules / joule-seconds |
| Native SET core profile | Cycles converted at 1 GHz; pJ converted to joules |

A producer with multiple consumers supplies each required tensor interval to
each destination. A consumer with multiple parents needs every parent's input;
demand is not divided by the parent count. Composite outgoing traffic is based
on distinct crossing tensors, not the sum of all internal layer outputs.

## Scheduling and memory tables

`ScT[i,j]` is a cumulative completed-sub-batch count. MeT entries are cumulative
cutoffs; live activation MB is `(ScT−MeT) × output_MB`, summed across blocks.
`SchedulingTable.initial_counts` handles nonzero initial progress explicitly.

Canonical plans have `2N−1` states. A serial micro-batch reference instead has
one state per block per micro-batch and explicitly flushes dead intermediates
between independent micro-batches. The latter remains executable when the
canonical representation has insufficient sub-batches for its dependency gaps.

Each nested child invocation processes exactly one parent sub-batch. The boundary
buffer reserve and child budgets are explicit; states cannot assign more active
blocks than modeled tiles. Profiled runs require full expansion to individual
layers. Depth-truncated analytical runs are reported as such.

The demo's memory chart and the top-level `peak_*_mb` fields show **top-level
boundary buffers**. Child budgets and solver records are separate in the hierarchy
trace. This activation model is not a complete physical SRAM accounting of
registers, weights, native mapper buffers and optimizer state.

## Objective and solver reports

The exact ScT MILP uses binary expansion of integer workloads to linearize the
EDP product. Eligible canonical inference problems can instead enumerate an
exact translated-simplex vertex set. See [MATHEMATICAL_AUDIT.md](MATHEMATICAL_AUDIT.md).

Under the configured uniform traffic costs, both traffic latency and energy
increase with DRAM MB, so the memory solver minimizes that quantity. Its report's
incumbent/bound are in MB; ScT EDP reports use J·s. Limits and statuses are recorded.
A fixed-subproblem optimum does not certify the globally best DNN hierarchy.

## Training paths

The recommended recorded training suite uses `training_cohorts.py`. It describes
FW over all cohorts, BW1 over retained tail cohorts, and recompute/BW2 over the
remaining prefix. All gradient samples are covered exactly once.

The older `_search_training_with_recomputation` implementation remains available
for research comparison. It has not received the same complete physical-gradient
and resource audit and is not the evidence source for the current training report.

## Output provenance

Every new matrix has a suite manifest with source hashes, commit, dependency
versions and configuration, plus per-case logs/results. Negative outcomes are
explicit. Compressed checked-in records preserve the original uncompressed result
hashes. Plot/report regeneration verifies those hashes and the workload definition
before producing visualizations.
