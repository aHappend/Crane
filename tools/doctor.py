"""Check the runtime dependencies and execute a tiny SCIP integer program."""

from importlib.metadata import PackageNotFoundError, version
import sys


def main() -> int:
    print(f"Python {sys.version.split()[0]}")
    if sys.version_info < (3, 11):
        print("Use Python 3.11 or newer.", file=sys.stderr)
        return 1

    missing = []
    for name in ("numpy", "ortools", "pypdf"):
        try:
            print(f"{name} {version(name)}")
        except PackageNotFoundError:
            missing.append(name)
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}", file=sys.stderr)
        print("Run: python -m pip install -r requirements.txt -c constraints.txt", file=sys.stderr)
        return 1

    try:
        import numpy as np
        from ortools.linear_solver import pywraplp

        if int(np.array([1, 2]).sum()) != 3:
            raise RuntimeError("NumPy self-check failed")
        solver = pywraplp.Solver.CreateSolver("SCIP")
        if solver is None:
            raise RuntimeError("This OR-Tools installation does not provide SCIP")
        solver.SetTimeLimit(5000)
        x = solver.IntVar(0, 4, "x")
        solver.Add(x >= 1.5)
        solver.Minimize(x)
        if solver.Solve() != pywraplp.Solver.OPTIMAL or abs(x.solution_value() - 2) > 1e-6:
            raise RuntimeError("SCIP integer-program self-check failed")
        print(f"SCIP available: {solver.SolverVersion()}")
    except Exception as exc:
        print(f"Runtime check failed: {exc}", file=sys.stderr)
        return 1
    print("Environment ready. Run: python example/quickstart.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
