# Architecture and data conventions

## Execution path

```text
experiment entrypoint
  → construct Layer objects and dependencies
  → group layers into Block objects
  → SearchConfig + search_schedule(...)
      → prepare/merge blocks and derive block dependencies
      → enumerate candidate sub-batches
      → optimize_sct_table(...)        compute-side schedule
      → optimize_memory_table(...)     memory-side decisions
      → evaluate combined latency, energy and EDP
      → optionally recurse/refine or search training phases
  → SearchResult → text / CSV / JSON / HTML
```

[`example/quickstart.py`](../example/quickstart.py) is the smallest complete
example. It uses three synthetic layers, a §7.2-style hardware profile, a flat
search and real solvers with fallback disabled.

## Module ownership

| Module | Responsibility |
| --- | --- |
| [`model/layer.py`](../model/layer.py) | Layer workload, output size, four mapping dimensions, parent/child links |
| [`model/dag_parser.py`](../model/dag_parser.py) | Build graphs from dictionary specs and topologically sort them |
| [`scheduler/block.py`](../scheduler/block.py) | Nested block ownership, workload aggregation, dependency projection and linear-chain merging |
| [`scheduler/scheduling_table.py`](../scheduler/scheduling_table.py) | ScT container and legacy table constructors |
| [`scheduler/paper_milp.py`](../scheduler/paper_milp.py) | Main ScT integer program, tile utilization and compute cost coefficients |
| [`scheduler/memory_table.py`](../scheduler/memory_table.py) | MeT integer program, memory constraints and traffic cost |
| [`scheduler/milp_solver.py`](../scheduler/milp_solver.py) | Older state-mix solver and shared `MilpSolution` result type; the main ScT path uses `paper_milp.py` |
| [`scheduler/hardware_profile.py`](../scheduler/hardware_profile.py) | Paper-inspired inference and training profiles and unit conversion |
| [`search/scheduler_search.py`](../search/scheduler_search.py) | `SearchConfig`, `SearchResult`, pruning, hierarchy feedback and FW/BW1/BW2 orchestration |
| [`example/schedule_html.py`](../example/schedule_html.py) | Self-contained HTML reports; no web server required |

Composite blocks own their children through `sub_blocks`. A layer should occur
once in a block tree: putting it both in `layers` and in a child duplicates its
FLOPs and output volume. `merge_linear_blocks()` preserves child ownership.

The public DAG parser expects reusable layer specs in practice; pass a list of
specifications. Experiment builders typically construct `Layer` objects directly.

## Table semantics

For `N` ordered blocks, the default schedule has `2N−1` states. Each block has
a processing window of `N` states starting at its block index.

- `ScT[i, j]` is the **cumulative number of completed sub-batches** for block `j`
  at state `i`. The difference between adjacent rows gives work performed in a
  state. It is not latency, bytes or per-state batch size.
- `MeT_S[i, j]` and `MeT_D[i, j]` are cumulative cutoff indices used by the memory
  model. They are not memory occupancy in MB. Live volume in the implemented
  constraints is `(ScT − MeT) × block_output_volume`, summed over blocks.
- With no special training bounds, each block ends with
  `batch_size // best_sub_batch` completed sub-batches.
- A state's workload can advance several active blocks. Do not expect
  `sum(state_batches)` to equal the number of sub-batches for one block.

Training returns per-phase dictionaries in `SearchResult.phase_results`, keyed
by `fw`, `bw1` and `bw2`. The aggregate result adds phase latency and energy and
then computes their product; adding phase EDPs gives a different quantity.

## Units

The following are the conventions used with the paper hardware helpers:

| Field / metric | Unit |
| --- | --- |
| `Layer.flops` | Operations used by the analytical model |
| `Layer.output_size`, `sram_capacity`, `dram_capacity` | MB (decimal in hardware conversion) |
| `noc_bandwidth`, `dram_bandwidth` | MB/s |
| `compute_power_per_tile` | Operations/s, despite the historical name “power” |
| `compute_energy_per_op` | Joules/operation |
| `noc_energy_per_unit`, `dram_energy_per_unit` | Joules/MB |
| Latency, energy, EDP | Seconds, joules, joule-seconds |
| `num_pes` | Number of modeled hardware tiles in the search |

Some older examples use normalized compute throughput and manually assigned
costs. Their numbers are meaningful within that configuration, not calibrated
hardware timings. The sub-batch scaling limitations are detailed in
[REPRODUCTION.md](REPRODUCTION.md#known-modeling-differences).

`paper_7_2_search_params()` and `paper_7_3_search_params()` also return
`traffic_energy_per_unit`, a legacy compatibility key that is not accepted by
`SearchConfig`. Copy supported fields or remove that key before unpacking.

## Search and failure behavior

Flat search first ranks ScT candidates, then solves MeT for retained candidates,
then chooses a survivor by total EDP. The default combined latency is the maximum
of compute and memory latency; total energy is their sum.

`allow_solver_fallback=False` is the default on `SearchConfig`. Failed candidates
are skipped; if none remain, search raises a runtime error. Enable
`verbose_progress=True` to see the underlying candidate failures. Most solver
calls have no time limit. A successful `ortools-scip` label does not expose an
optimality gap or prove exact EDP optimality.

`derive_recursive_traces` can trigger recursive joint optimization even when
`enable_hierarchical_pipeline=False`; it is not a reporting-only option. For a
flat smoke test, set it to `False` and use `max_hierarchy_depth=1`, as in the quick start.

New algorithm changes should preserve the invariants checked in `tests/`, and
update the [reproduction map](REPRODUCTION.md) when modeling assumptions change.
