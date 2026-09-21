# Mathematical and implementation audit

This audit separates properties of the implemented mathematical models from
claims about hardware fidelity or reproduction of paper figures. Tests use
independent tiny-instance enumeration and conservation checks; large examples
alone are insufficient evidence of correctness.

## 1. Workload conservation across sub-batches

`Layer.flops`, activation MB and external input MB are **per sample**. For batch
`B`, candidate sub-batch `b` and `Q = B / b`, the per-invocation compute workload
and activation volume are multiplied by `b`. Each layer completes `Q` invocations,
so its total work is `Q × b × FLOPs = B × FLOPs`.

Static weight MB is not multiplied by the sub-batch size. The base traffic model
streams weights per invocation; the recorded SET mapper supplies its own local
buffer-access costs. Neither convention establishes optimal weight residency
across all hierarchy levels.

The former implementation changed `Q` while keeping per-invocation FLOPs fixed.
That made a larger sub-batch appear to reduce total arithmetic. The regression
test fixes a single layer and checks equal total work across all batch factors.

## 2. Exact integer-product EDP formulation

For a fixed ScT cost model, latency and energy have the form

```text
L = Σ l_i w_i
E = e_0 + Σ e_i w_i,     0 ≤ w_i ≤ U_i, integer
```

Expand each integer workload as `w_i = Σ 2^k b_ik`, where `b_ik` is binary. After
normalizing latency to `x = L / L_max` in `[0,1]`, introduce `z_ik = x b_ik` with:

```text
0 ≤ z_ik ≤ x
z_ik ≤ b_ik
z_ik ≥ x − (1 − b_ik)
```

These constraints are exact for a binary multiplier. Minimizing the normalized
linear expression `e_0 x + Σ e_i 2^k z_ik` therefore minimizes the fixed-model
EDP exactly, subject to solver termination. Scaling avoids physical EDP values
being lost near absolute feasibility tolerances.

`scheduler/optimization.py` also retains an explicit `relaxation` mode. A single
continuous McCormick envelope can have a zero lower bound across many different
schedules. The Crane paper itself discusses piecewise/McCormick approximation
in §5.5; the exact integer formulation is a documented implementation choice,
not a claim that the paper never used a relaxation.

## 3. Exact canonical inference reduction

For `N` blocks and `2N−1` canonical states, completeness gives
`Σ(k=j..j+N−1) w_k = Q` for every block `j`. Subtracting adjacent equations gives:

```text
w_(i+N) = w_i       for i = 0 .. N−2
Σ(i=0..N−1) w_i = Q
```

For a forward dependency `p → c`, with `p < c`, paper Eq.5 imposes a progress gap
`g`. At state `N+c−2`, the parent is complete and the child lacks exactly
`w_(c−1)` invocations, hence `w_(c−1) ≥ g`. This condition is also sufficient:
at earlier relevant states, the parent-child difference is a sum of nonnegative
workloads containing that term.

The eligible feasible set is consequently a translated simplex: each of the
first `N` workloads has a known integer lower bound, and their sum is `Q`.
For positive linear `L` and `E`, `log(L E) = log(L) + log(E)` is concave. A minimum
over this simplex is attained at a vertex. Its `N` vertices are integral: place
all remaining workload in one coordinate after satisfying the lower bounds.

`canonical_fastpath` enumerates these vertices. It is used only without extra
state-balancing, custom training boundaries or restrictive tile-capacity bounds.
Other cases use the full MILP. Exhaustive enumeration tests compare both paths
against independently constructed feasible ScT tables.

This proves optimality only for that fixed ScT subproblem. It does not prove
global optimality across hierarchy choices, cost models, memory decisions or
the complete DNN scheduling space.

## 4. Tensor identities and multiple parents

A consumer's demand is an index interval `(previous_progress, current_progress]`.
Each dependency must provide its own full interval. Requirements are not divided
by the number of parents.

- Direct transfers intersect demand with the parent's newly produced interval.
- SRAM reads intersect remaining demand with `(MeT_S, previous_parent_progress]`.
- DRAM covers the remaining valid historical interval.
- Missing data in both memories causes failure; it is never silently invented
  as a DRAM read.

The MILP and cost report share `scheduler/traffic.py`. Tests specifically cover
equal-length but disjoint producer/consumer intervals and a two-parent join.
For composite blocks, only distinct tensors crossing a block boundary are
charged to each destination; internal outputs are not all treated as outgoing
traffic.

## 5. Memory objective and capacity

ScT/MeT entries are cumulative integer indices, not bytes. Live memory volume is
`Σ_j (ScT_ij − MeT_ij) × output_MB_j` for each state and memory type.

With the implemented uniform per-MB costs and `H_D ≥ H_S = 1`, both traffic
latency and traffic energy are nondecreasing affine functions of total DRAM
traffic. Minimizing DRAM MB therefore minimizes their product. This avoids an
unnecessarily loose product relaxation and the old unit-dependent SRAM tie-breaker.
An exact piecewise integer clamp encodes the portion of each requested interval
that is unavailable in SRAM.

Memory solver reports use DRAM MB for their incumbent/bound; ScT EDP reports use
physical energy-delay units. A feasible time-limited result is explicitly labeled
`feasible`, with a bound and relative gap when available.

## 6. Nested composition and resource budgets

`search/nested_search.py` treats a child schedule as one invocation processing
exactly the parent's selected sub-batch. Child cost is evaluated for its assigned
tile count and memory budget. Costs from one sub-batch size are never reused as
if calibrated for another size.

A composite reserves half its activation capacity for its own boundary buffers;
the other half is split among children by tile share. The same conservative split
is applied recursively. Canonical states with more active blocks than available
tiles are disabled. A serial micro-batch schedule is evaluated as an alternative,
including for a single input sample. The default path no longer resamples unrelated
state axes and retries failed child constraints with free boundaries.

The fixed partition, half-budget split and bounded hierarchy search are explicit
implementation choices. They can leave performance on the table. Native core
profiles require full leaf expansion; a depth-truncated composite cannot silently
fall back to analytical core costs in a profiled run.

## 7. Training cohort reference

For `Q` sub-batches, retain the last `K` uniform cohorts:

```text
FW:                  1 .. Q
BW1:                 Q−K+1 .. Q
recompute then BW2:  1 .. Q−K
```

Every sample receives one forward and one backward execution; only discarded
cohorts are recomputed. With `Q=3`, `K=1`, the ledger matches the sample ranges
displayed in paper Figure 6. Capacity reserves one cohort of gradient workspace,
and recomputation needs one activation cohort plus that workspace.

This is a checked serial reference with uniform retention, an analytical backward
work multiplier and an activation-oriented capacity model. It excludes optimizer
state and the full per-layer checkpoint search. A recorded `infeasible` outcome
is restricted to this policy. The older MILP training path remains experimental
and must not be substituted as fully validated training evidence.
