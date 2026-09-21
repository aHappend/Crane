"""Run multiple seeded, budgeted instances of the unmodified SET search."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--set-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--rounds',type=int,default=20)
    parser.add_argument('--seeds',default='7,19,43')
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).absolute().parents[1]
    manifest={'schema_version':1,'created_at_utc':datetime.now(timezone.utc).isoformat(),
              'rounds_per_layer':args.rounds,'seeds':[int(x) for x in args.seeds.split(',')],'runs':[]}
    for model in ('resnet','vgg','goog','trans_cell'):
        for seed in manifest['seeds']:
            dest=args.output_dir/f'{model}_seed{seed}'
            log=args.output_dir/f'{model}_seed{seed}.log'
            with log.open('w') as output:
                completed=subprocess.run([sys.executable,str(root/'tools/run_set_baseline.py'),
                    '--set-root',str(args.set_root),'--network',model,'--batch','64','--mesh','4',
                    '--rounds',str(args.rounds),'--seed',str(seed),'--timeout','180','--output-dir',str(dest)],
                    stdout=output,stderr=subprocess.STDOUT)
            record={'network':model,'seed':seed,'returncode':completed.returncode,'directory':dest.name}
            manifest['runs'].append(record)
            (args.output_dir/'suite.json').write_text(json.dumps(manifest,indent=2)+'\n')
            print(f'{model} seed={seed}: returncode={completed.returncode}',flush=True)
    if any(row['returncode'] for row in manifest['runs']):
        raise SystemExit(1)


if __name__=='__main__':
    main()
