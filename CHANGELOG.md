# Changelog

## Unreleased

### Real-workload reproduction and recorded experiments

- Export 16 compiled SET network graphs with tensor shapes, per-sample work,
  residual edges and dynamic-weight dependencies; preserve source hashes.
- Correct sub-batch workload scaling, complete multi-parent demand, tensor-index
  traffic matching, DAG handling and crossing-tensor accounting.
- Add normalized exact integer-product ScT optimization, an audited canonical
  vertex reduction, solver time limits, status/bound/gap reports and exact
  monotone DRAM-traffic minimization.
- Replace the default hierarchical path with explicit parent-sub-batch macro
  composition, tile limits, conservative memory budgets and serial alternatives.
- Add hash-checked native SET Polar core profiles, a seeded upstream SET runner,
  and a checked uniform-cohort training reference with explicit infeasibility.
- Run and archive 10 inference comparisons, 9 training capacity configurations
  and 12 native SET searches; retain the VGG regression and two policy-infeasible
  training configurations. These are not claims of reproduced headline speedups.
- Add one-command experiment/report tools, scientific figures and a self-contained
  interactive demo, validated at desktop/mobile sizes in a real browser.
- Expand independent mathematical oracles, graph/profile checks and experiment
  integration tests. See the mathematical audit and recorded report for scope.

### Documentation and reproducibility

- Identify the MICRO 2025 paper, DOI and authors; add English/Chinese READMEs,
  citation metadata and third-party provenance.
- Document architecture, units, experiment commands, paper-to-code coverage,
  proxy workload assumptions and remaining validation work.
- Add an environment check that solves a small SCIP integer program, a synthetic
  quick start with JSON/HTML outputs, and a pinned dependency snapshot.
- Add contributor guidance, issue/PR templates and Linux/Windows CI.
- Activate the CI workflow after the maintainer authorized workflow access;
  run checks on pull requests and pushes to `main`, and upload quick-start reports.

### Correctness and execution fixes

- Restore path-helper imports/bootstrap in nine example entrypoints so direct
  execution and help commands no longer raise `NameError` at startup.
- Fix merged blocks counting child layers both directly and recursively.
  **This changes workload totals and can change schedules and estimated costs.**
  Regenerate comparisons; historical `outputs/runs/` files are not a current
  numerical baseline.
- Derive parent completion bounds before building recursive intra-block traces;
  this removes the undefined `child_lb` / `child_ub` failure.
- Write new merged proxy-suite outputs to a dedicated
  `outputs/experiments/official_nns_suite_<timestamp>/` directory, consistent
  with the other maintained experiment entrypoints.
- Add regression tests for block accounting, dependency preservation, script
  startup, ScT/MeT invariants, training phase aggregation and recursive traces.

### Repository hygiene

- Stop tracking the checked-in virtual environment, bytecode caches and local
  editor configuration. Git history and archived research outputs are preserved.
- Ignore new test/lint caches, build artifacts and local environment files.
