"""Export actual SET Polar intra-layer mappings for use by Crane's scheduler.

The core mapper accounts for MACs, local buffers and local interconnect. External
NoC/DRAM traffic, placement and shared-buffer write energy are separate; this is
an intra-layer plugin, not the full SET inter-layer evaluator.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time

REVISION = "a7bd73912f58d9fea10fadab693eebd4e6e3054d"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set-root", type=Path, required=True)
    parser.add_argument("--network", default="resnet")
    parser.add_argument("--max-batch", type=int, default=64)
    parser.add_argument("--max-tiles", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.max_batch < 1 or args.max_tiles < 1 or args.max_batch & (args.max_batch-1):
        parser.error("max-batch must be a power of two and tile count positive")
    root = args.set_root.resolve()
    revision = subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()
    if revision != REVISION:
        parser.error("SET source revision does not match the pinned revision")
    if subprocess.check_output(["git","status","--porcelain","--untracked-files=no"],cwd=root):
        parser.error("SET tracked sources must be unmodified")
    args.output_dir.mkdir(parents=True,exist_ok=False)
    subprocess.run(["make","-j2"],cwd=root,check=True,stdout=subprocess.DEVNULL)
    original = (root/'src/main.cpp').read_text()
    signature = 'int main(int argc, char** argv)'
    if original.count(signature) != 1:
        raise RuntimeError("upstream entrypoint changed")
    bridge = Path(__file__).with_name('export_set_costs.cpp').read_text()
    generated = original.replace(signature,'int upstream_main(int argc, char** argv)') + '\n' + bridge
    objects = sorted(p for p in (root/'build/objects').rglob('*.o') if p.name!='main.o')
    started = time.monotonic()
    raw = args.output_dir/'mappings.csv'
    with tempfile.TemporaryDirectory(prefix='crane-set-core-') as temp:
        source,binary = Path(temp)/'profile.cpp',Path(temp)/'profile'
        source.write_text(generated)
        subprocess.run(['g++','-std=c++17','-O3',f'-I{root / "include"}',str(source),
                        *(str(x) for x in objects),'-lpthread','-o',str(binary)],check=True)
        with raw.open('w') as output:
            output.write('layer_id,batch,tiles,status,cycles,energy_pj,mac_pj,buffer_pj,bus_pj,ubuf_pj,K,B,H,W\n')
            output.flush()
            subprocess.run([str(binary),args.network,str(args.max_batch),str(args.max_tiles)],
                           stdout=output,check=True,timeout=600)
    rows=list(csv.DictReader(raw.open()))
    manifest={
        'schema_version':1,'kind':'SET_Polar_intra_layer','upstream_revision':revision,
        'network':args.network,'max_batch':args.max_batch,'max_tiles':args.max_tiles,
        'rows':len(rows),'valid_rows':sum(x['status']=='ok' for x in rows),
        'frequency_hz':1e9,'core_mac_count':1024,'core_sram_bytes':1048576,
        'cost_units':{'cycles':'1 ns at 1 GHz','energy_pj':'picojoules summed across allocated tiles'},
        'included':['MAC','local buffers','local buses','intra-tile buffer accesses'],
        'excluded':['external NoC/DRAM','placement-dependent hops','external shared-buffer writes'],
        'csv_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),
        'bridge_sha256':hashlib.sha256(bridge.encode()).hexdigest(),
        'generated_source_sha256':hashlib.sha256(generated.encode()).hexdigest(),
        'elapsed_seconds':time.monotonic()-started,
    }
    (args.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':
    main()
