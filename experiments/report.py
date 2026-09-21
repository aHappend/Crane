"""Verify recorded run hashes, archive raw evidence, and build figures and demo."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from math import ceil
from pathlib import Path
import shutil
import statistics

import numpy as np

from workloads.hierarchy import balanced_blocks
from workloads.set_models import load_set_model


def read_json(path):
    path=Path(path)
    data=path.read_bytes() if path.exists() else gzip.decompress(path.with_suffix(path.suffix+'.gz').read_bytes())
    return json.loads(data),data


def collect_suite(source, destination):
    manifest,_=read_json(source/'manifest.json')
    destination.mkdir(parents=True,exist_ok=True)
    if (source/'manifest.json').resolve()!=(destination/'manifest.json').resolve():
        shutil.copy2(source/'manifest.json',destination/'manifest.json')
    results=[]
    for row in manifest['cases']:
        case_dir=source/row['directory']
        result,data=read_json(case_dir/'result.json')
        if hashlib.sha256(data).hexdigest()!=row['result_sha256']:
            raise ValueError(f'result checksum mismatch: {case_dir}')
        target=destination/row['directory'];target.mkdir(exist_ok=True)
        for name in ('case.json','result.json','hierarchy.json','candidate_failures.json','run.log'):
            path=case_dir/name
            compressed=path.with_suffix(path.suffix+'.gz')
            if path.exists() or compressed.exists():
                raw=path.read_bytes() if path.exists() else gzip.decompress(compressed.read_bytes())
                (target/(name+'.gz')).write_bytes(gzip.compress(raw,mtime=0))
        result['evidence_path']=str(target/'result.json.gz')
        results.append(result)
    return manifest,results


def thin_result(result):
    data={k:v for k,v in result.items() if k not in ('tables','candidates','workload','config')}
    data['workload']={k:v for k,v in result.get('workload',{}).items() if k not in ('source',)}
    data['hardware']={k:result.get('config',{}).get(k) for k in ('sram_capacity','dram_capacity','num_pes','batch_size')}
    if 'tables' in result:
        original=result['tables'];case=result['case']
        workload=load_set_model(case['model'],activation_bytes=result['workload']['activation_bytes'],weight_bytes=result['workload']['weight_bytes'])
        if workload.manifest()['definition_sha256']!=result['workload']['definition_sha256']:
            raise ValueError('recorded workload metadata hash differs from the current definition')
        blocks=workload.blocks() if case.get('mode')=='serial' else balanced_blocks(workload.layers,int(case.get('fanout',4)))
        if [b.name for b in blocks]!=original['blocks']:
            raise ValueError('recorded blocks no longer match the workload definition')
        volumes=np.array([b.boundary_output_size()*result['metrics']['best_sub_batch'] for b in blocks])
        table=np.array(original['sct']);ms=np.array(original['met_s']);md=np.array(original['met_d'])
        sram=(table-ms)@volumes;dram=(table-md)@volumes
        indices=sorted(set([*range(0,len(table),max(1,ceil(len(table)/128))),len(table)-1]))
        data['tables']={'blocks':original['blocks'],'total_states':len(table),'indices':indices,
            'states':[original['states'][i] for i in indices],
            'sct':[original['sct'][i] for i in indices],
            'sram_mb':[float(sram[i]) for i in indices],'dram_mb':[float(dram[i]) for i in indices]}
    return data


def native_references(source,destination):
    manifest=json.loads((source/'suite.json').read_text())
    destination.mkdir(parents=True,exist_ok=True)
    if len(manifest['runs'])!=4*len(manifest['seeds']) or any(x['returncode'] for x in manifest['runs']):
        raise ValueError('native SET suite is incomplete or has failed runs')
    if (source/'suite.json').resolve()!=(destination/'suite.json').resolve():
        shutil.copy2(source/'suite.json',destination/'suite.json')
    rows=[]
    for item in manifest['runs']:
        run=source/item['directory'];target=destination/item['directory'];target.mkdir(exist_ok=True)
        for path in run.iterdir():
            if path.is_file():
                name=path.name.removesuffix('.gz')
                raw=gzip.decompress(path.read_bytes()) if path.suffix=='.gz' else path.read_bytes()
                (target/(name+'.gz')).write_bytes(gzip.compress(raw,mtime=0))
        result,_=read_json(run/'manifest.json')
        rows.append(result)
    return rows


def make_figures(inference,training,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'crane-reproduction'})
    colors=['#147d73','#e5b24a','#476eae']
    grouped={}
    for row in inference:
        if row['status']=='completed' and row['case']['batch']==64 and row['case']['tiles']==16:
            grouped.setdefault(row['case']['model'],{})[row['case']['mode']]=row
    names=[name for name,rows in grouped.items() if {'nested','serial'}<=rows.keys()]
    ratios=[grouped[name]['nested']['metrics']['edp_joule_seconds']/grouped[name]['serial']['metrics']['edp_joule_seconds'] for name in names]
    fig,ax=plt.subplots(figsize=(8,4.3),layout='constrained')
    bars=ax.bar([x.replace('_','\n') for x in names],ratios,color=[colors[0] if x<=1 else colors[1] for x in ratios],width=.58)
    ax.axhline(1,color='#7d8794',linestyle='--',linewidth=1)
    ax.set_ylim(0,max(1.18,max(ratios)*1.15));ax.set_ylabel('EDP / serial reference EDP')
    ax.set_title('Recorded comparison · batch 64 · 16 tiles',loc='left',fontweight='bold')
    ax.bar_label(bars,labels=[f'{x:.3f}×' for x in ratios],padding=4)
    fig.text(.015,.005,'Same SET core profiles and analytical traffic model. Lower is better; includes the VGG regression.',fontsize=8,color='#52606d')
    fig.savefig(out/'edp_comparison.svg',metadata={'Date':None});fig.savefig(out/'edp_comparison.png',dpi=170)
    plt.close(fig)
    models=sorted({r['case']['model'] for r in training})
    fig,axes=plt.subplots(1,len(models),figsize=(10,3.8),layout='constrained',sharey=True)
    for ax,model in zip(np.atleast_1d(axes),models):
        rows=sorted([r for r in training if r['case']['model']==model],key=lambda r:r['case']['dram_mb'])
        for i,row in enumerate(rows):
            if row['status']!='completed':
                ax.text(i,.08,'infeasible',rotation=90,ha='center',color='#a34332',fontsize=9)
                ax.scatter([i],[.025],marker='x',color='#a34332')
            else:
                m=row['metrics'];q=row['case']['batch']/m['best_sub_batch']
                fraction=m['recomputed_sub_batches']/q
                ax.bar(i,fraction,color=colors[2],width=.55)
                ax.text(i,fraction+.025,f'{fraction:.0%}',ha='center',fontsize=9)
        ax.set_xticks(range(len(rows)),[str(r['case']['dram_mb']) for r in rows]);ax.set_xlabel('DRAM capacity (MB)')
        ax.set_title(model);ax.set_ylim(0,1.13)
    np.atleast_1d(axes)[0].set_ylabel('Fraction of sub-batches recomputed')
    fig.suptitle('Uniform-cohort training reference · batch 256 · 2 tiles',x=.01,ha='left',fontweight='bold')
    fig.savefig(out/'training_memory.svg',metadata={'Date':None});fig.savefig(out/'training_memory.png',dpi=170)
    plt.close(fig)
    for path in out.glob('*.svg'):
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inference',type=Path,required=True)
    parser.add_argument('--training',type=Path,required=True)
    parser.add_argument('--set-baselines',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,default=Path('experiments/results/reproduction_20260921'))
    parser.add_argument('--demo',type=Path,default=Path('docs/demo/index.html'))
    args=parser.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    im,inference=collect_suite(args.inference,out/'raw/inference')
    tm,training=collect_suite(args.training,out/'raw/training')
    native=native_references(args.set_baselines,out/'raw/native_set')
    figures=out/'figures';figures.mkdir(exist_ok=True)
    make_figures(inference,training,figures)
    fields=['model','mode','batch','tiles','dram_mb','status','latency_seconds','energy_joules','edp_joule_seconds','best_sub_batch','elapsed_seconds']
    with (out/'summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator="\n");writer.writeheader()
        for row in inference+training:
            data={k:row['case'].get(k,'') for k in fields}
            data.update({k:row.get('metrics',{}).get(k,'') for k in fields if k in row.get('metrics',{})})
            data.update(status=row['status'],elapsed_seconds=row.get('elapsed_seconds',''))
            writer.writerow(data)
    dataset={'inference':[thin_result(x) for x in inference],
             'training':[thin_result(x) for x in training],'native_set':native,
             'provenance':{'inference':im['environment'],'training':tm['environment']},
             'paper_doi':'10.1145/3725843.3756023'}
    (out/'report.json').write_text(json.dumps(dataset,separators=(',',':'),allow_nan=False)+'\n')
    template=Path(__file__).with_name('templates')/'demo.html'
    safe=json.dumps(dataset,separators=(',',':'),allow_nan=False).replace('<','\\u003c')
    args.demo.parent.mkdir(parents=True,exist_ok=True)
    args.demo.write_text(template.read_text().replace('__RECORDED_DATA__',safe),encoding='utf-8')
    lines=['# Recorded reproduction experiments','',
        'Metrics are cost-model estimates. Search time is measured wall time. These runs do not establish reproduction of the paper’s headline speedups.','',
        '## Inference with native SET core profiles','',
        '| Model | Mode | Batch | Latency (ms) | Energy (mJ) | EDP (J·s) | Search (s) |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for r in inference:
        if r['status']!='completed':continue
        c,m=r['case'],r['metrics']
        lines.append(f"| {c['model']} | {c['mode']} | {c['batch']} | {m['latency_seconds']*1000:.3f} | {m['energy_joules']*1000:.3f} | {m['edp_joule_seconds']:.7g} | {r['elapsed_seconds']:.2f} |")
    lines.extend(['','![EDP comparison](figures/edp_comparison.svg)','',
        'Both modes use identical recorded SET Polar core mappings. External traffic uses the Python analytical model. Nested execution uses a conservative half-boundary/half-child memory split. Table occupancy is scoped to top-level boundary buffers; child budgets are recorded separately. VGG-19’s slight EDP regression is retained.','',
        '## Training memory reference','',
        'The uniform-cohort serial reference checks FW/BW1/recomputation/BW2 coverage and an activation-plus-gradient-workspace budget. It does not exhaust the paper’s per-layer checkpoint choices.','',
        '![Training memory](figures/training_memory.svg)','',
        '| Model | DRAM (MB) | Outcome | Sub-batch | Retained | Recomputed |',
        '| --- | ---: | --- | ---: | ---: | ---: |'])
    for r in training:
        c,m=r['case'],r.get('metrics',{})
        lines.append(f"| {c['model']} | {c['dram_mb']} | {r['status']} | {m.get('best_sub_batch','—')} | {m.get('retained_sub_batches','—')} | {m.get('recomputed_sub_batches','—')} |")
    lines.extend(['','## Native SET reference runs','',
        'Three seeds, 20 SA rounds per layer, four internal trials, batch 64, 4×4 Polar tiles. These use SET’s full placement/traffic evaluator. Cross-model EDP ratios are not claimed as reproduced speedups.','',
        '| Network | Runs | Median latency (ms) | Median energy (mJ) | Median EDP (J·s) | Median search (s) |',
        '| --- | ---: | ---: | ---: | ---: | ---: |'])
    for name in sorted({r['network'] for r in native}):
        rows=[r for r in native if r['network']==name and r['status']=='completed' and 'SET' in r['results']]
        if not rows:continue
        med=lambda key:statistics.median(r['results']['SET'][key] for r in rows)
        lines.append(f"| {name} | {len(rows)} | {med('latency_seconds')*1000:.3f} | {med('energy_joules')*1000:.3f} | {med('edp_joule_seconds'):.7g} | {statistics.median(r['wall_time_seconds'] for r in rows):.2f} |")
    lines.extend(['','## Provenance and raw evidence','',
        '- `summary.csv`: full-precision tabular metrics; `report.json`: demo/figure data.',
        '- `raw/inference` and `raw/training`: original manifests, gzip-compressed results, configurations and solver logs. Result hashes are verified before report generation.',
        '- `raw/native_set`: upstream seed patches, configurations, native summaries, trees and logs. The pinned source revision is recorded in every manifest.',
        f"- Inference source commit: `{im['environment']['commit']}`; source digest: `{im['environment']['source_digest']}`.",
        '- The experiment source file hashes are authoritative when documentation/report commits are newer than a run.',
        '- Static weight/optimizer-state capacity and full placement-aware traffic calibration remain outside the training reference/analytical traffic model.',''])
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(f'Report: {out / "REPORT.md"}\nDemo: {args.demo}')


if __name__=='__main__':
    main()
