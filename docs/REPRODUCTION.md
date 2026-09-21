# Reproduction scope and paper-to-code map

The reference is [Gong et al., MICRO 2025](https://doi.org/10.1145/3725843.3756023).
The repository implements a Python research model of its scheduling workflow.
The paper states in §7.1 that its implementation is C++; this repository should
be cited as a separate implementation.

**Current evidence supports workflow reproduction and small-instance checks.**
There is no validated reproduction here of the paper's complete evaluation,
cycle-accurate cost calibration, or reported performance improvements.

## Method coverage

“Implemented” below means that a code path exists, not that its equivalence to
the paper has been proved. Tests exercise a small subset of feasible instances.

| Paper concept | Implementation | Scope and qualification |
| --- | --- | --- |
| §4 hierarchical blocks and dependencies | [`model/`](../model/), [`scheduler/block.py`](../scheduler/block.py) | Layer DAGs, nested blocks and linear-chain merging. Merging approximates graph partitioning. |
| ScT, cumulative progress and processing windows; Eqs. 1–6 | [`scheduler/paper_milp.py`](../scheduler/paper_milp.py) | Integer variables, default `2N−1` states, completeness, monotonicity and dependency constraints. |
| MeT and memory capacity; Eqs. 7–12 | [`scheduler/memory_table.py`](../scheduler/memory_table.py) | SRAM/DRAM cutoff tables with capacity and availability constraints. |
| §5.3 training and recomputation; Eqs. 13–15 | [`search/scheduler_search.py`](../search/scheduler_search.py), `_search_training_with_recomputation` | FW, BW1 and BW2 search with retention profiles and scaled backward/recompute costs. No tensor training or accuracy evaluation. |
| Compute and traffic cost evaluation | [`scheduler/paper_milp.py`](../scheduler/paper_milp.py), [`search/`](../search/), [`cost_model/`](../cost_model/) | Analytical mapping utilization, bandwidth and energy estimates. No cycle simulator or ARM memory compiler integration. |
| Compute / traffic EDP objectives; Eqs. 22–23 | `_add_mccormick_product_objective` in both table solvers | McCormick relaxation of a product, followed by evaluation of the actual product on selected candidates. This is not an exact globally optimal EDP solver. |
| Candidate pruning and final EDP selection | `_flat_search_prepared` | ScT candidates → top-K1 → MeT → top-K2 → total EDP. K2 is calculated over the surviving stage-2 candidates. |
| §6 hierarchical structure optimization | `_hierarchical_search`, `_recursive_joint_optimize_prepared` | Bounded recursive optimization, cost feedback and trial expansions; heuristic choices differ from a full paper-equivalence implementation. |
| §7.2 / §7.3 hardware settings | [`scheduler/hardware_profile.py`](../scheduler/hardware_profile.py) | Parameter translations with explicit units; choosing a profile alone does not reproduce the experiments. |
| §7 result figures and baseline comparisons | No complete validated harness | SET / Tangram / TileFlow / MBS comparisons, calibration and numerical tolerances remain to be established. |

## Where the workloads come from

### SET reference definitions

`src/nns/*.cpp`, `include/network.h` and `include/nns/nns.h` are third-party
reference material. [`src/nns/SOURCE.txt`](../src/nns/SOURCE.txt) records the
SET-ISCA2023 origin and upstream commit. These files are not a complete SET
build, trained weights, or an importable Crane benchmark dataset.

### Twelve network-family proxies

`official_specs()` in
[`example/run_official_nns_suite.py`](../example/run_official_nns_suite.py)
contains hand-written `(name, GFLOPs, output_MB)` tuples for AlexNet, Darknet19,
DenseNet, GNMT, GoogLeNet, Inception-ResNet, LLM, PNASNet, ResNet, Transformer,
VGG and ZFNet. `build_layer_blocks()` connects these proxy nodes into chains.

The `source_ref` field is a reference pointer. The Python examples do not parse
the corresponding C++ file to recover its exact shapes or graph. “Layer-level”
in the proxy suite means one proxy node per block, not every original DNN layer.

### 471-node Transformer

`build_transformer_min_layers()` manually constructs a 471-node chain inspired
by the SET Transformer definition. `_layer_profile()` assigns costs using layer
names. It uses eight attention groups and hand-written dimensions, whereas the
paper's Transformer-Large evaluation specifies a different workload, including
16 heads and hidden dimension 1024 (§7.2). Do not interpret a matching layer
count or `source_ref` as an exact workload match.

The training example uses this same workload with backward and recomputation
scale factors. It does not measure neural-network training time or accuracy.

## Known modeling differences

1. **Relaxed objective.** A McCormick envelope bounds a bilinear product; without
   additional exactness arguments it need not equal that product. An optimal
   SCIP status for this linearized problem does not certify globally optimal
   EDP for the original formulation. The code also accepts feasible statuses
   and does not currently expose a solver gap in `SearchResult`.
2. **Cost scaling.** Flat search reuses fixed block FLOPs and output sizes across
   sub-batch candidates while changing the number of sub-batches. It does not
   rebuild shape-aware costs for each sub-batch size. Workload calibration and
   normalization across batch choices need validation before quantitative claims.
3. **Hardware approximation.** Costs use aggregate capacity, simplified hop
   assumptions and utilization estimates rather than complete placement,
   contention, register/buffer energy or a validated cycle model.
4. **Units and conventions.** The §7.2 helper uses decimal MB/GB and two ops per
   MAC. Its formula gives 16.384 GB/s DRAM bandwidth for 16 tiles and 147.456 GB/s
   for 144 tiles; the paper lists 16 / 144 GB/s. The training helper defaults to
   32768 MB, while the paper states 32 GB. These conventions are recorded, not
   silently treated as identical configurations.
5. **Search controls.** `strict_paper_mode=True` selects the repository's
   staged/pruned search behavior. It does not remove the above approximations.
   Defaults and example-specific overrides differ; inspect `SearchConfig` and
   record the complete configuration.
6. **Hierarchy boundaries.** Recursive search may retry with free child
   boundaries after a constrained child solve fails. Inspect `hierarchy_notes`
   for `boundary_retry=free` and trace fields for `free_fallback`. This is
   separate from `allow_solver_fallback`, which controls solver replacement.
7. **Graph handling.** The data model supports DAG edges, but many bundled
   examples deliberately build chains. Search infers a linear chain when a
   multi-block input has no dependencies, so an edgeless input is not treated
   as a set of independent branches.

## What the tests establish

The test suite checks real SCIP execution, example imports and CLI startup,
workload preservation across nested block merges, fork/join edge preservation,
ScT completeness and monotonicity, memory bounds/capacity, small training phase
aggregation, parent-bound recursive traces, and quick-start artifact generation.

These are software correctness checks for the covered cases. They do not
establish accuracy of the cost model, complete constraint equivalence, the
quality of large-instance schedules, or reproduction of the paper's headline
21.01× EDP / 2.82× scheduling-speed claims.

Historical files in `outputs/runs/` predate the block double-counting fix. Keep
them as historical artifacts and regenerate results for a new comparison;
do not mix them with outputs from the corrected implementation.

## Remaining work

| Milestone | Evidence needed to consider it complete |
| --- | --- |
| Exact workload reconstruction | Shape- and dependency-preserving imports; per-layer FLOPs/bytes and workload checksums matching each target paper experiment |
| Cost-model validation | Documented batch scaling, precision and units; comparisons against a common SET schedule or cycle model with stated tolerances |
| Solver formulation audit | Equation-by-equation constraints, tiny exhaustive oracles, objective-gap reporting and explicit status/time-limit handling |
| Baseline evaluation | Pinned SET / Tangram / TileFlow / MBS implementations, matched hardware and workloads, raw measurements and repeatable commands |
| Paper figure reproduction | Per-figure scripts, baseline provenance, configuration manifests and numerical comparison tables |
| Redistribution readiness | Maintainer-selected license for original code and verified rights/terms for reference material |

Track improvements against these milestones rather than marking the whole
paper “reproduced” after a successful example run.
