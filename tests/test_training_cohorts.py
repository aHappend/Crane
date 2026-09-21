from dataclasses import replace

import pytest

from model.layer import Layer
from search.scheduler_search import SearchConfig
from search.training_cohorts import run_training_cohorts
from workloads.set_models import NetworkWorkload


def example():
    first,second=Layer('first',10,1,input_size=1),Layer('second',20,1,input_size=0)
    first.connect_to(second)
    workload=NetworkWorkload('paper_figure6',[first,second],{},1,1)
    cfg=SearchConfig(3,[1],1,100,num_pes=2)
    return workload,cfg


def test_paper_figure6_cohorts_are_disjoint_and_cover_every_gradient():
    workload,cfg=example()
    result=run_training_cohorts(workload,cfg,retained=1)
    assert result['cohorts']=={'fw_indices':[1,3],'bw1_indices':[3,3],'recompute_bw2_indices':[1,2]}
    assert result['checkpoint']['forward_final_met_d']==[2,2]
    assert result['phases']['fw']['operations']==90
    assert result['phases']['bw1']['operations']==60
    assert result['phases']['recompute']['operations']==60
    assert result['phases']['bw2']['operations']==120
    assert all(result['checks'].values())


def test_tighter_memory_requires_more_recomputation():
    workload,cfg=example()
    full=run_training_cohorts(workload,cfg)
    limited=run_training_cohorts(workload,replace(cfg,dram_capacity=4))
    assert full['metrics']['recomputed_sub_batches']==0
    assert limited['metrics']['recomputed_sub_batches']==2
    assert limited['metrics']['training_operations']>full['metrics']['training_operations']
    with pytest.raises(RuntimeError,match='no cohort schedule'):
        run_training_cohorts(workload,replace(cfg,dram_capacity=1))
