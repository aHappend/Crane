import json
from pathlib import Path
import subprocess
import sys

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


def test_infeasible_policy_is_an_explicit_recorded_outcome(tmp_path):
    config=tmp_path/'config.json'
    config.write_text(json.dumps({'cases':[{'model':'resnet50','mode':'training','batch':256,
        'tiles':2,'dram_mb':64,'sub_batches':[1],'expected_status':'infeasible'}]}))
    proc=subprocess.run([sys.executable,'-m','experiments.run','--config',str(config),
                         '--output-dir',str(tmp_path/'run')],cwd=Path(__file__).parents[1],
                        capture_output=True,text=True,timeout=20)
    assert proc.returncode==0,proc.stderr
    result=json.loads((tmp_path/'run/case_000/result.json').read_text())
    assert result['status']=='infeasible'
    assert result['capacity_analysis']['minimum_required_dram_mb']>64
    assert 'metrics' not in result
