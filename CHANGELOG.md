# Changelog

## Unreleased

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
