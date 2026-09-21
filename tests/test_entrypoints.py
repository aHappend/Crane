import importlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).absolute().parents[1]
EXAMPLES = sorted(p.stem for p in (ROOT / "example").glob("*.py") if p.name != "__init__.py")


@pytest.mark.parametrize("name", EXAMPLES)
def test_example_can_be_imported_without_running_experiments(name):
    importlib.import_module(f"example.{name}")


@pytest.mark.parametrize("script", [
    "compare_all_networks_same_source_merge.py",
    "compare_all_networks_stage_vs_layer.py",
    "compare_transformer_granularity.py",
    "compare_transformer_stage_layer_with_merge.py",
    "run_transformer_training_repro.py",
    "quickstart.py",
])
def test_direct_script_help_works_outside_repository(script, tmp_path):
    proc = subprocess.run([sys.executable, str(ROOT / "example" / script), "--help"],
                          cwd=tmp_path, text=True, capture_output=True, timeout=20)
    assert proc.returncode == 0, proc.stderr
    assert "usage:" in proc.stdout


def test_quickstart_writes_reproducible_artifacts(tmp_path):
    out = tmp_path / "run"
    proc = subprocess.run([sys.executable, str(ROOT / "example" / "quickstart.py"), "--output-dir", str(out)],
                          cwd=tmp_path, text=True, capture_output=True, timeout=30)
    assert proc.returncode == 0, proc.stderr
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["evidence_scope"] == "workflow_smoke_test"
    assert summary["config"]["allow_solver_fallback"] is False
    assert summary["environment"]["ortools"]
    assert len(summary["workload"]) == 3
    assert summary["result"]["sct_solver"] == summary["result"]["met_solver"] == "ortools-scip"
    assert "synthetic three-layer chain" in (out / "schedule.html").read_text(encoding="utf-8")


def test_doctor_executes_scip():
    proc = subprocess.run([sys.executable, str(ROOT / "tools" / "doctor.py")],
                          text=True, capture_output=True, timeout=20)
    assert proc.returncode == 0, proc.stderr
    assert "SCIP available:" in proc.stdout
