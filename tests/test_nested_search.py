from dataclasses import replace

import numpy as np
import pytest

from example.quickstart import build_demo
from scheduler.block import derive_block_dependencies
from search.nested_search import NestedSearch
from workloads.hierarchy import balanced_blocks


def test_nested_invocations_process_exactly_the_parent_sub_batch():
    blocks,cfg=build_demo()
    layers=[next(b.iter_layers()) for b in blocks]
    cfg=replace(cfg,batch_size=4,candidate_sub_batches=[1,2],num_pes=2,canonical_fastpath=True)
    grouped=balanced_blocks(layers,2)
    engine=NestedSearch(depth=5)
    result=engine.run(grouped,cfg)
    assert not engine.unexpanded
    assert np.all(result.sct.table[-1] * result.best_sub_batch == cfg.batch_size)
    assert all(len(active)<=cfg.num_pes for active in result.state_active_blocks)
    invocations=[t for t in result.hierarchy_traces if 'child_batch_size' in t]
    assert invocations
    for trace in invocations:
        assert trace['child_batch_size']==trace['parent_sub_batch']
        assert trace['child_batch_size'] % trace['child_sub_batch']==0
        assert trace['tiles']<=cfg.num_pes
    assert all('free_fallback' not in x for x in result.hierarchy_notes)


def test_single_sample_uses_a_feasible_serial_micro_schedule():
    blocks,cfg=build_demo()
    cfg=replace(cfg,batch_size=1,candidate_sub_batches=[1],num_pes=1,canonical_fastpath=True)
    result=NestedSearch(depth=1).run(blocks,cfg)
    assert result.sct_solver_name=='serial-microbatch'
    assert result.block_dependencies==derive_block_dependencies(blocks)
    assert np.all(result.sct.table[-1]==1)
    assert all(len(active)==1 for active in result.state_active_blocks)
    assert result.total_edp==pytest.approx(result.total_latency*result.total_energy)
