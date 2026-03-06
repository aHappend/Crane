from dataclasses import dataclass
from typing import List, Sequence

try:
    from ortools.linear_solver import pywraplp
except Exception:  # pragma: no cover
    pywraplp = None


@dataclass
class MilpSolution:
    state_batches: List[int]
    latency: float
    energy: float
    objective: float
    solver_name: str


class MilpSolver:
    def __init__(self, max_batch_per_state: int = 10_000) -> None:
        self.max_batch_per_state = max_batch_per_state

    def optimize_state_mix(
        self,
        total_sub_batches: int,
        state_latency: Sequence[float],
        state_energy: Sequence[float],
        weight_latency: float = 1.0,
        weight_energy: float = 1.0,
        max_batches_per_state: Sequence[int] | None = None,
        min_batches_per_state: Sequence[int] | None = None,
        min_active_states: int = 1,
        min_batch_if_active: int = 1,
    ) -> MilpSolution:
        if len(state_latency) != len(state_energy):
            raise ValueError("state_latency and state_energy must have the same length")

        n_states = len(state_latency)
        if max_batches_per_state is not None and len(max_batches_per_state) != n_states:
            raise ValueError("max_batches_per_state length must equal number of states")
        if min_batches_per_state is not None and len(min_batches_per_state) != n_states:
            raise ValueError("min_batches_per_state length must equal number of states")

        if pywraplp is not None:
            return self._solve_with_ortools(
                total_sub_batches,
                state_latency,
                state_energy,
                weight_latency,
                weight_energy,
                max_batches_per_state,
                min_batches_per_state,
                min_active_states,
                min_batch_if_active,
            )
        return self._solve_analytic(
            total_sub_batches,
            state_latency,
            state_energy,
            weight_latency,
            weight_energy,
            max_batches_per_state,
            min_batches_per_state,
            min_active_states,
            min_batch_if_active,
        )

    def _solve_with_ortools(
        self,
        total_sub_batches: int,
        state_latency: Sequence[float],
        state_energy: Sequence[float],
        weight_latency: float,
        weight_energy: float,
        max_batches_per_state: Sequence[int] | None,
        min_batches_per_state: Sequence[int] | None,
        min_active_states: int,
        min_batch_if_active: int,
    ) -> MilpSolution:
        solver = pywraplp.Solver.CreateSolver("SCIP")
        if solver is None:
            return self._solve_analytic(
                total_sub_batches,
                state_latency,
                state_energy,
                weight_latency,
                weight_energy,
                max_batches_per_state,
                min_batches_per_state,
                min_active_states,
                min_batch_if_active,
            )

        n_states = len(state_latency)
        upper_bounds = [self.max_batch_per_state] * n_states
        if max_batches_per_state is not None:
            upper_bounds = [int(v) for v in max_batches_per_state]

        lower_bounds = [0] * n_states
        if min_batches_per_state is not None:
            lower_bounds = [int(v) for v in min_batches_per_state]

        x = [solver.IntVar(0, max(0, upper_bounds[i]), f"s_{i}") for i in range(n_states)]
        for i in range(n_states):
            solver.Add(x[i] >= lower_bounds[i])

        y = [solver.IntVar(0, 1, f"y_{i}") for i in range(n_states)]
        for i in range(n_states):
            m_i = max(0, upper_bounds[i])
            solver.Add(x[i] <= m_i * y[i])
            if min_batch_if_active > 0:
                solver.Add(x[i] >= min_batch_if_active * y[i])

        solver.Add(sum(x) == int(total_sub_batches))
        solver.Add(sum(y) >= int(max(1, min_active_states)))

        latency_expr = solver.Sum(x[i] * float(state_latency[i]) for i in range(n_states))
        energy_expr = solver.Sum(x[i] * float(state_energy[i]) for i in range(n_states))
        solver.Minimize(weight_latency * latency_expr + weight_energy * energy_expr)

        status = solver.Solve()
        if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            raise RuntimeError("MILP solver failed to find a feasible solution")

        batches = [int(round(v.solution_value())) for v in x]
        latency = sum(b * l for b, l in zip(batches, state_latency))
        energy = sum(b * e for b, e in zip(batches, state_energy))
        objective = weight_latency * latency + weight_energy * energy
        return MilpSolution(batches, latency, energy, objective, "ortools-scip")

    def _solve_analytic(
        self,
        total_sub_batches: int,
        state_latency: Sequence[float],
        state_energy: Sequence[float],
        weight_latency: float,
        weight_energy: float,
        max_batches_per_state: Sequence[int] | None,
        min_batches_per_state: Sequence[int] | None,
        min_active_states: int,
        min_batch_if_active: int,
    ) -> MilpSolution:
        n_states = len(state_latency)
        unit_cost = [weight_latency * l + weight_energy * e for l, e in zip(state_latency, state_energy)]
        order = sorted(range(n_states), key=lambda i: unit_cost[i])

        upper = [self.max_batch_per_state] * n_states
        if max_batches_per_state is not None:
            upper = [int(v) for v in max_batches_per_state]
        lower = [0] * n_states
        if min_batches_per_state is not None:
            lower = [int(v) for v in min_batches_per_state]

        batches = lower[:]
        remaining = int(total_sub_batches) - sum(batches)
        if remaining < 0:
            raise RuntimeError("Infeasible constraints: mandatory lower bounds exceed total_sub_batches")

        active_target = max(1, min(int(min_active_states), n_states, int(total_sub_batches)))
        for i in order[:active_target]:
            need = max(0, min_batch_if_active - batches[i])
            add = min(need, max(0, upper[i] - batches[i]))
            batches[i] += add
            remaining -= add

        if remaining < 0:
            raise RuntimeError("Infeasible constraints while enforcing active state lower bounds")

        for i in order:
            if remaining <= 0:
                break
            room = max(0, upper[i] - batches[i])
            add = min(room, remaining)
            batches[i] += add
            remaining -= add

        if remaining != 0:
            raise RuntimeError("Infeasible constraints: upper bounds too tight for total_sub_batches")

        active_cnt = sum(1 for b in batches if b >= min_batch_if_active)
        if active_cnt < active_target:
            raise RuntimeError("Infeasible constraints: cannot satisfy min_active_states")

        latency = sum(b * l for b, l in zip(batches, state_latency))
        energy = sum(b * e for b, e in zip(batches, state_energy))
        objective = weight_latency * latency + weight_energy * energy
        return MilpSolution(batches, latency, energy, objective, "analytic-fallback")
