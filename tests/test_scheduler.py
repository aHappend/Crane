from dataclasses import replace

import numpy as np
import pytest

from example.quickstart import build_demo
from scheduler.block import merge_linear_blocks
from scheduler.hardware_profile import paper_7_2_search_params, paper_7_3_search_params
from search.scheduler_search import _derive_recursive_intra_block_traces, search_schedule


def test_inference_schedule_is_complete_and_within_memory_capacity():
    blocks, config = build_demo()
    result = search_schedule(blocks, config)
    assert result.sct_solver_name == result.met_solver_name == "ortools-scip"
    assert result.best_sub_batch in config.candidate_sub_batches
    total = config.batch_size // result.best_sub_batch
    sct = result.sct.table
    assert sct.shape == (5, 3)
    assert np.all(np.diff(sct, axis=0) >= 0)
    assert np.all(sct[-1] == total)
    for parent, child in result.block_dependencies:
        assert np.all(sct[:, parent] >= sct[:, child])
    volumes = np.array([block.total_output_size() for block in blocks])
    for met, capacity in ((result.met.sram, config.sram_capacity), (result.met.dram, config.dram_capacity)):
        assert np.all(met >= 0)
        assert np.all(met <= sct)
        assert np.all(np.diff(met, axis=0) >= 0)
        assert np.all((sct - met) @ volumes <= capacity + 1e-6)
    assert np.isfinite(result.total_edp) and result.total_edp > 0
    assert result.total_edp == pytest.approx(result.total_latency * result.total_energy)


def test_training_exposes_all_three_phases():
    blocks, config = build_demo()
    config = replace(config, batch_size=4, candidate_sub_batches=[2], enable_training_recomputation=True)
    result = search_schedule(blocks[:2], config)
    assert set(result.phase_results) == {"fw", "bw1", "bw2"}
    assert result.total_latency == pytest.approx(sum(float(p["total_latency"]) for p in result.phase_results.values()))
    assert result.total_energy == pytest.approx(sum(float(p["total_energy"]) for p in result.phase_results.values()))
    assert np.isfinite(result.total_edp) and result.total_edp > 0


def test_recursive_trace_uses_parent_completion_bounds():
    blocks, config = build_demo()
    merged, _, _ = merge_linear_blocks(blocks, [(0, 1), (1, 2)], max_layers_per_block=2)
    parent = search_schedule(merged, config)
    traces, notes = _derive_recursive_intra_block_traces(merged, parent, config, depth_remaining=1, lineage=[])
    assert traces, notes
    assert any("recursive_trace_ok" in note for note in notes)
    assert not any("recursive_trace_failed" in note for note in notes)
    child = traces[0]
    assert child["sct"][-1][-1] == parent.sct.table[-1, 0]


def test_hardware_profile_units():
    inference = paper_7_2_search_params(16)
    assert inference["sram_capacity"] == 16
    assert inference["compute_power_per_tile"] == pytest.approx(2.048e12)
    assert inference["dram_bandwidth"] == pytest.approx(16384)
    assert inference["noc_energy_per_unit"] == pytest.approx(0.7 * 8e-6)
    training = paper_7_3_search_params(2)
    assert training["sram_capacity"] == 20
    assert training["dram_bandwidth"] == 300000
    assert training["dram_energy_per_unit"] == pytest.approx(3.9 * 8e-6)
