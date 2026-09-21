"""Numerically scaled objectives and honest termination reports for SCIP.

For integer workload counts, binary expansion makes the EDP product an exact
MILP, rather than a single loose continuous McCormick envelope. Exactness is
about the formulated fixed-cost model; a time-limited incumbent is not a proof
of optimality and the analytical cost model still needs external calibration.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Sequence


@dataclass(frozen=True)
class SolverReport:
    status: str
    objective_model: str
    incumbent: float
    best_bound: float
    relative_gap: float | None
    wall_time_seconds: float
    variables: int
    constraints: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def configure_solver(solver, time_limit_s: float) -> None:
    if not math.isfinite(time_limit_s) or time_limit_s <= 0:
        raise ValueError("solver time limit must be finite and positive")
    solver.SetTimeLimit(max(1, int(time_limit_s * 1000)))
    solver.SetNumThreads(1)
    solver.SetSolverSpecificParametersAsString(
        "display/verblevel = 0\nrandomization/randomseedshift = 0\n"
        "numerics/feastol = 1e-8\n"
    )


def product_objective(
    solver,
    latency_expr,
    energy_terms: Sequence[tuple[Any, float, int]],
    latency_upper_bound: float,
    *,
    energy_constant: float = 0.0,
    method: str = "exact",
) -> float:
    """Minimize L*E where E is affine in bounded nonnegative integer counts.

    Returns the scale converting the solver's objective/bound to physical EDP.
    Binary products z=x*b are exact through four linear inequalities when b is
    binary. Counts are still limited by the original integer variables' bounds.
    """
    if method not in {"exact", "relaxation"}:
        raise ValueError("EDP method must be 'exact' or 'relaxation'")
    if energy_constant < 0 or any(c < 0 or ub < 0 for _, c, ub in energy_terms):
        raise ValueError("energy coefficients and count bounds must be nonnegative")
    e_upper = energy_constant + sum(c * ub for _, c, ub in energy_terms)
    if latency_upper_bound <= 0 or e_upper <= 0:
        solver.Minimize(0)
        return 1.0
    x = solver.NumVar(0, 1, "normalized_latency")
    solver.Add(x == latency_expr / latency_upper_bound)
    if method == "relaxation":
        y = solver.NumVar(0, 1, "normalized_energy")
        solver.Add(y == (energy_constant + solver.Sum(v * c for v, c, _ in energy_terms)) / e_upper)
        z = solver.NumVar(0, 1, "normalized_edp")
        solver.Add(z >= x + y - 1)
        solver.Add(z <= x)
        solver.Add(z <= y)
        solver.Minimize(z)
    else:
        terms = [x * (energy_constant / e_upper)]
        for i, (count, coefficient, upper) in enumerate(energy_terms):
            if upper == 0 or coefficient == 0:
                continue
            bits = []
            for k in range(int(upper).bit_length()):
                bit = solver.BoolVar(f"edp_bit_{i}_{k}")
                product = solver.NumVar(0, 1, f"edp_product_{i}_{k}")
                solver.Add(product <= x)
                solver.Add(product <= bit)
                solver.Add(product >= x - (1 - bit))
                bits.append(bit * (1 << k))
                terms.append(product * ((1 << k) * coefficient / e_upper))
            solver.Add(count == solver.Sum(bits))
        solver.Minimize(solver.Sum(terms))
    return float(latency_upper_bound * e_upper)


def solver_report(solver, status: int, objective_model: str, scale: float = 1.0) -> SolverReport:
    names = {0: "optimal", 1: "feasible", 2: "infeasible", 3: "unbounded",
             4: "abnormal", 5: "model_invalid", 6: "not_solved"}
    if status not in (0, 1):
        raise RuntimeError(f"SCIP terminated with status={names.get(status, str(status))}")
    incumbent = float(solver.Objective().Value()) * scale
    bound = float(solver.Objective().BestBound()) * scale
    gap = max(0.0, incumbent - bound) / max(abs(incumbent), 1e-30) if math.isfinite(bound) else None
    return SolverReport(names[status], objective_model, incumbent, bound, gap,
                        solver.WallTime() / 1000, solver.NumVariables(), solver.NumConstraints())
