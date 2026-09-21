"""Run an explicit experiment matrix with per-case deadlines and raw artifacts."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import numpy as np

from cost_model.set_profile import SetCoreProfile
from example.schedule_html import write_schedule_html
from scheduler.hardware_profile import paper_7_2_search_params, paper_7_3_search_params
from search.nested_search import NestedSearch
from search.scheduler_search import SearchConfig, search_schedule
from workloads.hierarchy import balanced_blocks
from workloads.set_models import load_set_model

ROOT = Path(__file__).absolute().parents[1]


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def provenance():
    hashes = {}
    for folder in ('model','scheduler','search','cost_model','workloads','experiments'):
        for path in sorted((ROOT/folder).rglob('*.py')):
            hashes[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        dirty = bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit,dirty = None,None
    return {'commit':commit,'dirty':dirty,'source_sha256':hashes,
            'source_digest':hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),
            'python':platform.python_version(),'platform':platform.platform(),
            'numpy':version('numpy'),'ortools':version('ortools')}


def execute_case(case: dict, out: Path) -> dict:
    training = case.get('mode') == 'training'
    workload=load_set_model(case['model'],activation_bytes=int(case.get('activation_bytes',2 if training else 1)),
                            weight_bytes=int(case.get('weight_bytes',2 if training else 1)))
    tiles=int(case.get('tiles',2 if training else 16))
    hardware=(paper_7_3_search_params(tiles) if training else paper_7_2_search_params(tiles))
    hardware.pop('traffic_energy_per_unit')
    if 'sram_mb' in case: hardware['sram_capacity']=float(case['sram_mb'])
    if 'dram_mb' in case: hardware['dram_capacity']=float(case['dram_mb'])
    batch=int(case.get('batch',64))
    sub_batches=case.get('sub_batches',[x for x in (1,2,4,8,16,32,64) if batch % x==0])
    cfg=SearchConfig(batch,sub_batches,num_pes=tiles,**hardware,
        enable_chain_block_merge=False,derive_recursive_traces=False,enable_structure_refinement=False,
        solver_time_limit_s=float(case.get('solver_seconds',5)),max_hierarchy_depth=1,
        dependency_gap=1,canonical_fastpath=case.get('edp_method','exact')=='exact',
        edp_method=case.get('edp_method','exact'))
    fanout=int(case.get('fanout',4))
    blocks=balanced_blocks(workload.layers,fanout)
    profile=SetCoreProfile(ROOT/case['core_profile'],workload) if case.get('core_profile') else None
    started=time.monotonic()
    if training:
        from search.training_cohorts import run_training_cohorts
        result=run_training_cohorts(workload,cfg,retained=case.get('retained_sub_batches'))
        result.update({'case':case,'workload':workload.manifest(),'config':asdict(cfg),
                       'elapsed_seconds':time.monotonic()-started})
        return result
    if case.get('mode','nested')=='flat':
        result=search_schedule(blocks,cfg)
        search_details={'algorithm':'flat_canonical','unexpanded_blocks':len(blocks)}
    else:
        if case.get('mode')=='serial':
            blocks=workload.blocks()
        engine=NestedSearch(depth=int(case.get('depth',6)),core_profile=profile,
                            plans=('serial',) if case.get('mode')=='serial' else ('serial','pipeline'))
        result=engine.run(blocks,cfg)
        reports=[r for child in engine.cache.values() for r in child.solver_reports]+result.solver_reports
        search_details={'algorithm':'nested_sub_batch_macros','cache_entries':len(engine.cache),
            'solver_calls':engine.solver_calls,'infeasible_attempts':len(engine.failures),
            'unexpanded_blocks':sorted(engine.unexpanded),
            'solver_status_counts':dict(Counter(r['status'] for r in reports)),
            'max_recorded_gap':max((r['relative_gap'] or 0 for r in reports),default=0.0)}
        write_json(out/'candidate_failures.json',engine.failures)
    elapsed=time.monotonic()-started
    volumes=np.array([b.boundary_output_size()*result.best_sub_batch for b in blocks])
    sct,ms,md=result.sct.table,result.met.sram,result.met.dram
    checks={
        'finite_positive_costs':all(np.isfinite(x) and x>0 for x in (result.total_latency,result.total_energy,result.total_edp)),
        'complete_batch':bool(np.allclose(sct[-1]*result.best_sub_batch,batch)),
        'monotone_sct':bool(np.all(np.diff(sct,axis=0)>=-1e-7)),
        'memory_cutoffs_bounded':bool(np.all(ms>=-1e-7) and np.all(md>=-1e-7) and np.all(ms<=sct+1e-7) and np.all(md<=sct+1e-7)),
        'dependency_progress':all(bool(np.all(sct[:,p]>=sct[:,c]-1e-7)) for p,c in result.block_dependencies),
        'sram_capacity':bool(np.all((sct-ms)@volumes<=cfg.sram_capacity+1e-6)),
        'dram_capacity':bool(np.all((sct-md)@volumes<=cfg.dram_capacity+1e-6)),
        'physical_tile_budget':all(len(active)<=cfg.num_pes for active,w in zip(result.state_active_blocks,result.milp_solution.state_batches) if w>0),
    }
    payload={
        'status':'completed' if all(checks.values()) else 'validation_failed',
        'case':case,'workload':workload.manifest(),'config':asdict(cfg),'elapsed_seconds':elapsed,
        'cost_model':'SET_Polar_core_plus_analytical_traffic' if profile else 'analytical_compute_and_traffic',
        'core_profile':profile.manifest if profile else None,'search':search_details,'checks':checks,
        'metrics':{'latency_seconds':result.total_latency,'energy_joules':result.total_energy,
            'edp_joule_seconds':result.total_edp,'best_sub_batch':result.best_sub_batch,
            'peak_sram_mb':float(np.max((sct-ms)@volumes)),
            'peak_dram_mb':float(np.max((sct-md)@volumes)),
            'operations_per_batch':sum(layer.flops for layer in workload.layers)*batch},
        'solver_reports':result.solver_reports,'hierarchy_notes':result.hierarchy_notes,
        'tables':{'blocks':result.scheduled_blocks,'dependencies':result.block_dependencies,
            'states':result.state_order,'state_batches':result.milp_solution.state_batches,
            'sct':sct.tolist(),'met_s':ms.tolist(),'met_d':md.tolist()},
    }
    write_json(out/'hierarchy.json',result.hierarchy_traces)
    write_schedule_html(out/'schedule.html',title=f"{case['model']} / {case.get('mode','nested')}",
        meta={'batch':batch,'tiles':tiles,'sub_batch':result.best_sub_batch,'cost_model':payload['cost_model'],
              'validation':payload['status']},scheduled_blocks=result.scheduled_blocks,
        state_order=result.state_order,state_categories=result.state_categories,
        state_batches=result.milp_solution.state_batches,state_active_blocks=result.state_active_blocks,
        sct=sct.tolist(),met_s=ms.tolist(),met_d=md.tolist(),hierarchy_notes=result.hierarchy_notes)
    return payload


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--case-file',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--case-dir',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.case_file:
        case=json.loads(args.case_file.read_text())
        try:
            result=execute_case(case,args.case_dir)
        except Exception as exc:
            traceback.print_exc()
            result={'status':'error','case':case,'error':f'{type(exc).__name__}: {exc}'}
        write_json(args.case_dir/'result.json',result)
        raise SystemExit(0 if result['status']=='completed' else 1)
    if not args.config or not args.output_dir:
        parser.error('--config and --output-dir are required')
    config=json.loads(args.config.read_text())
    args.output_dir.mkdir(parents=True,exist_ok=False)
    out=args.output_dir.resolve()
    manifest={'schema_version':1,'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'suite':config,'config_sha256':hashlib.sha256(args.config.read_bytes()).hexdigest(),
        'environment':provenance(),'cases':[]}
    write_json(out/'manifest.json',manifest)
    for i,overrides in enumerate(config['cases']):
        case=dict(config.get('defaults',{}));case.update(overrides)
        case_dir=out/f'case_{i:03d}'
        case_dir.mkdir()
        case_file=case_dir/'case.json';write_json(case_file,case)
        started=time.monotonic()
        with (case_dir/'run.log').open('w') as log:
            try:
                completed=subprocess.run([sys.executable,'-m','experiments.run','--case-file',str(case_file),
                    '--case-dir',str(case_dir)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                    timeout=float(case.get('timeout_seconds',180)))
                status='completed' if completed.returncode==0 else 'failed'
            except subprocess.TimeoutExpired:
                status='timeout'
        path=case_dir/'result.json'
        if path.exists():
            result=json.loads(path.read_text());status=result['status']
        else:
            result={'status':status,'case':case};write_json(path,result)
        manifest['cases'].append({'directory':case_dir.name,'status':status,
            'elapsed_seconds':time.monotonic()-started,'result_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        write_json(out/'manifest.json',manifest)
        print(f"{i+1}/{len(config['cases'])} {case['model']} {case.get('mode','nested')} B{case.get('batch')} T{case.get('tiles')}: {status}",flush=True)
    print(f'Results: {args.output_dir}')
    if any(x['status']!='completed' for x in manifest['cases']):
        raise SystemExit(1)


if __name__=='__main__':
    main()
