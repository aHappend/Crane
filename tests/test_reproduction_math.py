from dataclasses import replace
from itertools import product

import numpy as np
from ortools.linear_solver import pywraplp
import pytest

from model.layer import Layer
from model.dag_parser import parse_layers
from scheduler.block import Block
from scheduler.memory_table import MemoryTable, optimize_memory_table
from scheduler.optimization import configure_solver, product_objective, solver_report
from scheduler.paper_milp import optimize_sct_table
from scheduler.scheduling_table import SchedulingTable
from scheduler.traffic import estimate_traffic
from search.scheduler_search import SearchConfig, search_schedule
from workloads.set_models import available_models, load_set_model


@pytest.mark.parametrize("scale", [1.0, 1e-12])
def test_exact_product_matches_exhaustive_integer_oracle(scale):
    solver = pywraplp.Solver.CreateSolver("SCIP")
    configure_solver(solver, 5)
    a, b = solver.IntVar(0, 4, "a"), solver.IntVar(0, 4, "b")
    solver.Add(a + b == 4)
    normalization = product_objective(solver, (3 * a + b) * scale,
                                      [(a, scale, 4), (b, 3 * scale, 4)], 16 * scale)
    status = solver.Solve()
    report = solver_report(solver, status, "exact_edp", normalization)
    expected = min((3*x + 4-x) * (x + 3*(4-x)) * scale**2 for x in range(5))
    assert report.status == "optimal"
    assert report.incumbent == pytest.approx(expected, abs=1e-30)
    assert report.best_bound == pytest.approx(expected, abs=1e-30)


@pytest.mark.parametrize("fastpath", [False, True])
def test_sct_matches_independent_exhaustive_state_enumeration(fastpath):
    n, q = 3, 4
    latency = [1, 3, 2, 4, 1]
    energy = [4, 2, 5, 2, 3]
    candidates = []
    for workloads in product(range(q + 1), repeat=2*n-1):
        table = np.array([[sum(workloads[k] for k in range(j, min(i+1, n+j)))
                           for j in range(n)] for i in range(2*n-1)])
        if not np.all(table[-1] == q):
            continue
        if any(table[i, p] < table[i, c] + 1 for p, c in ((0, 1), (1, 2))
               for i in range(c, n+c-1)):
            continue
        candidates.append(sum(a*b for a,b in zip(workloads,latency)) *
                          sum(a*b for a,b in zip(workloads,energy)))
    lat = [[float(latency[i]) for _ in range(n)] for i in range(2*n-1)]
    ene = [[energy[i] / sum(j <= i < n+j for j in range(n)) for _ in range(n)] for i in range(2*n-1)]
    result = optimize_sct_table([1]*n, [1]*n, q, [(0,1),(1,2)], 4, 1, 1,
                                dependency_gap=1, allow_fallback=False,
                                canonical_fastpath=fastpath,
                                state_block_latency_override=lat, state_block_energy_override=ene)
    assert result.report.status == "optimal"
    assert result.objective == pytest.approx(min(candidates))


def test_sub_batching_conserves_compute_work():
    layer = Layer("single", 8e6, 0.5, map_dims=(16, 16, 16, 16))
    cfg = SearchConfig(8, [1], 1000, 1000, num_pes=1, compute_power_per_tile=1e9,
                       derive_recursive_traces=False, enable_chain_block_merge=False)
    results = [search_schedule([Block("single", layers=[layer])], replace(cfg, candidate_sub_batches=[sb]))
               for sb in (1,2,4,8)]
    assert [r.compute_latency for r in results] == pytest.approx([0.064]*4)
    assert [r.compute_energy for r in results] == pytest.approx([64e6 * 1e-12]*4)


def test_each_parent_provides_full_matching_sample_interval():
    # Child sample 1 was produced in state 0. State-1 parents produce sample 2,
    # which cannot satisfy the child's demand for sample 1 by direct forwarding.
    sct = SchedulingTable(np.array([[1,1,0],[2,2,1],[2,2,2]], dtype=float))
    met = MemoryTable.zeros(3, 3)
    direct, sram, dram = estimate_traffic(sct, met, [2,3,4], [(0,2),(1,2)], [0,0,0])
    assert direct == 0
    assert sram == 10  # two samples from BOTH parent volumes (2+3).
    assert dram == 0
    met.sram[0,:2] = 1
    met.dram[0,:2] = 1
    with pytest.raises(ValueError, match="discarded"):
        estimate_traffic(sct, met, [2,3,4], [(0,2),(1,2)], [0,0,0])


@pytest.mark.parametrize("sram_capacity,expected_dram", [(0.0, 3.0), (1.0, 1.0)])
def test_memory_capacity_changes_traffic_without_changing_required_samples(sram_capacity, expected_dram):
    sct = SchedulingTable(np.array([[1,0],[2,1],[2,2]], dtype=float))
    result = optimize_memory_table(sct, [1,1], [(0,1)], sram_capacity, 100,
                                   allow_fallback=False, block_input_volumes=[0.5,0])
    _, _, dram = estimate_traffic(sct, result.table, [1,1], [(0,1)], [0.5,0])
    assert dram == pytest.approx(expected_dram)
    assert result.report.status == "optimal"


def test_parser_accepts_generator_and_rejects_bad_graphs():
    specs = [{"name":"a","flops":1,"output_size":1},
             {"name":"b","flops":2,"output_size":1,"parents":["a"]}]
    layers = parse_layers(x for x in specs)
    assert layers['b'].parents == [layers['a']]
    with pytest.raises(ValueError, match="duplicate"):
        parse_layers([specs[0], specs[0]])


def test_exported_networks_preserve_real_branches_and_operation_counts():
    assert len(available_models()) == 16
    resnet = load_set_model("resnet50")
    assert len(resnet.layers) == 72
    assert sum(len(x.parents) for x in resnet.layers) == 87
    conv1 = resnet.layers[0]
    assert conv1.flops == 2 * 3 * 64 * 7 * 7 * 112 * 112
    assert conv1.input_size == pytest.approx(3*224*224/1e6)
    assert sum(len(x.parents) > 1 for x in resnet.layers) == 16
    transformer = load_set_model("transformer")
    assert len(transformer.layers) == 471
    assert sum(len(x.parents) for x in transformer.layers) == 661


def test_disconnected_blocks_do_not_acquire_invented_dependencies():
    cfg = SearchConfig(4, [1], 100, 100, derive_recursive_traces=False, enable_chain_block_merge=False)
    blocks = [Block(str(i), layers=[Layer(str(i), 1, 1)]) for i in range(2)]
    assert search_schedule(blocks, cfg).block_dependencies == []
