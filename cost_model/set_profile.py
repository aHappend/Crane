"""Read a recorded profile from the actual, pinned SET intra-layer mapper."""
import csv
import hashlib
import json
from pathlib import Path


class SetCoreProfile:
    def __init__(self, directory, workload):
        directory=Path(directory)
        self.manifest=json.loads((directory/'manifest.json').read_text())
        data=directory/'mappings.csv'
        if hashlib.sha256(data.read_bytes()).hexdigest()!=self.manifest['csv_sha256']:
            raise ValueError('SET profile checksum mismatch')
        if self.manifest['upstream_revision']!=workload.source['revision']:
            raise ValueError('workload and cost profile source revisions differ')
        aliases={'resnet50':'resnet','resnet101':'resnet101','googlenet':'goog',
                 'vgg19':'vgg','transformer':'trans','transformer_cell':'trans_cell',
                 'bert_large_cell':'bert','gpt2_xl_prefill_cell':'gpt_prefill',
                 'gpt2_xl_decode_cell':'gpt_decode'}
        if aliases.get(workload.name,workload.name)!=self.manifest['network']:
            raise ValueError('workload does not match the SET profile network')
        self.layer_ids={layer.name:i for i,layer in enumerate(workload.layers)}
        self.rows={}
        for row in csv.DictReader(data.open()):
            key=tuple(int(row[k]) for k in ('layer_id','batch','tiles'))
            self.rows[key]=row

    def evaluate(self, layer, sub_batch, tiles):
        key=(self.layer_ids[layer.name],int(sub_batch),int(tiles))
        if key not in self.rows:
            raise ValueError(f'profile has no mapping for {key}; regenerate with sufficient batch/tiles')
        row=self.rows[key]
        if row['status']!='ok':
            raise RuntimeError(f'SET core mapper found no valid mapping for {key}')
        return float(row['cycles'])/self.manifest['frequency_hz'],float(row['energy_pj'])*1e-12
