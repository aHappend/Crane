import csv
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from cost_model.set_profile import SetCoreProfile
from experiments.report import collect_suite
from workloads.set_models import load_set_model

ROOT=Path(__file__).parents[1]


def test_native_core_profile_has_complete_keys_and_component_accounting():
    directory=ROOT/'experiments/reference_profiles/resnet50_t16'
    profile=SetCoreProfile(directory,load_set_model('resnet50'))
    rows=list(csv.DictReader((directory/'mappings.csv').open()))
    assert len(rows)==72*7*16
    assert len(profile.rows)==len(rows)
    for row in rows:
        assert row['status']=='ok'
        assert int(row['K'])*int(row['B'])*int(row['H'])*int(row['W'])==int(row['tiles'])
        components=sum(float(row[k]) for k in ('mac_pj','buffer_pj','bus_pj','ubuf_pj'))
        assert float(row['energy_pj'])==pytest.approx(components,rel=1e-8)
    latency,energy=profile.evaluate(load_set_model('resnet50').layers[0],1,1)
    assert latency==pytest.approx(307328e-9)
    assert energy>0


def test_report_rejects_tampered_evidence(tmp_path):
    source=tmp_path/'source';case=source/'case_000';case.mkdir(parents=True)
    raw=b'{"status":"completed","case":{}}'
    (case/'result.json.gz').write_bytes(gzip.compress(raw))
    manifest={'cases':[{'directory':'case_000','result_sha256':hashlib.sha256(raw).hexdigest()}]}
    (source/'manifest.json').write_text(json.dumps(manifest))
    _,results=collect_suite(source,tmp_path/'good')
    assert results[0]['status']=='completed'
    (case/'result.json.gz').write_bytes(gzip.compress(raw.replace(b'completed',b'modified')))
    with pytest.raises(ValueError,match='checksum mismatch'):
        collect_suite(source,tmp_path/'bad')
