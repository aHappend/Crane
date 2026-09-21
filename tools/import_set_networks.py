"""Compile the pinned SET definitions and export graph/shape metadata to JSON.

Requires a separate upstream checkout and a C++17 compiler. The runtime uses
the resulting JSON and does not need a compiler or a network connection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

UPSTREAM = "https://github.com/SET-Scheduling-Project/SET-ISCA2023"
REVISION = "a7bd73912f58d9fea10fadab693eebd4e6e3054d"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("workloads/set_networks.json"))
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    source = args.set_root.resolve()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if revision != REVISION:
        parser.error(f"expected upstream revision {REVISION}, got {revision}")
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=source):
        parser.error("upstream tracked sources have local changes")
    subprocess.run(["make", f"-j{max(1, args.jobs)}"], cwd=source, check=True)
    exporter = Path(__file__).with_suffix(".cpp")
    # The C++ exporter has a descriptive name independent of this CLI's name.
    exporter = exporter.with_name("export_set_networks.cpp")
    objects = sorted(p for p in (source / "build/objects").rglob("*.o") if p.name != "main.o")
    with tempfile.TemporaryDirectory(prefix="crane-export-") as temp:
        binary = Path(temp) / "export"
        subprocess.run(["g++", "-std=c++17", "-O2", f"-I{source / 'include'}",
                        str(exporter), *(str(p) for p in objects), "-lpthread", "-o", str(binary)], check=True)
        networks = json.loads(subprocess.check_output([str(binary)], text=True))
    source_hashes = {
        p.relative_to(source).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ("src/nns", "include/nns") for p in sorted((source / folder).glob("*")) if p.is_file()
    }
    payload = {
        "schema_version": 1,
        "source": {"repository": UPSTREAM, "revision": revision, "sha256": source_hashes},
        "units": {"shapes": "elements [channels, height, width]", "operations": "per sample; multiply-accumulate = 2 ops"},
        "method": "compiled upstream Network objects; branches and activation/weight dependencies preserved",
        "networks": networks,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for name, layers in networks.items():
        print(f"{name}: {len(layers)} nodes, {sum(len(x['parents']) for x in layers)} edges")
    print(f"Written: {args.out}")


if __name__ == "__main__":
    main()
