import json

from experiments.run import execute_case


def test_real_workload_experiment_writes_checkable_tables(tmp_path):
    case={'model':'zfnet','mode':'serial','batch':2,'sub_batches':[1],
          'tiles':4,'solver_seconds':2}
    result=execute_case(case,tmp_path)
    assert result['status']=='completed'
    assert all(result['checks'].values())
    assert result['workload']['nodes']==11
    assert len(result['tables']['blocks'])==11
    assert result['metrics']['operations_per_batch']==2*result['workload']['operations_per_sample']
    assert (tmp_path/'schedule.html').exists()
    assert json.loads((tmp_path/'candidate_failures.json').read_text())==[]
