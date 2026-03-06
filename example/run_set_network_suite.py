import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.layer import Layer
from scheduler.block import Block
from search.scheduler_search import SearchConfig, search_schedule


def load_profile(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def build_blocks(profile: dict) -> list[Block]:
    blocks: list[Block] = []
    prev_last = None
    for b in profile['blocks']:
        layers = []
        for item in b['layers']:
            layer = Layer(item['name'], float(item['flops']), float(item['output_size']))
            if prev_last is not None:
                prev_last.connect_to(layer)
            prev_last = layer
            layers.append(layer)
        blocks.append(Block(name=b['name'], layers=layers))
    return blocks


def candidate_sub_batches(batch_size: int) -> list[int]:
    base = [2, 4, 8, 16, 32]
    vals = [x for x in base if x <= batch_size and batch_size % x == 0]
    return vals if vals else [1]


def run_one(profile_path: Path) -> dict:
    profile = load_profile(profile_path)
    blocks = build_blocks(profile)
    batch = int(profile.get('batch_size', 64))

    cfg = SearchConfig(
        batch_size=batch,
        candidate_sub_batches=candidate_sub_batches(batch),
        sram_capacity=1e9,
        dram_capacity=1e9,
        num_pes=4,
        min_active_states=5,
        min_batch_if_active=1,
        max_state_share=0.45,
    )

    result = search_schedule(blocks, cfg)
    active_states = sum(1 for x in result.milp_solution.state_batches if x > 0)

    return {
        'network': profile['name'],
        'profile_file': str(profile_path).replace('\\', '/'),
        'batch_size': batch,
        'best_sub_batch': result.best_sub_batch,
        'state_order': ' -> '.join(result.state_order),
        'state_categories': ','.join(result.state_categories),
        'state_batches': ','.join(str(v) for v in result.milp_solution.state_batches),
        'active_states': active_states,
        'latency': f"{result.total_latency:.6f}",
        'energy': f"{result.total_energy:.6f}",
        'edp': f"{result.total_edp:.6f}",
        'solver': result.milp_solution.solver_name,
    }


def main() -> None:
    profile_dir = ROOT / 'set_network_src' / 'profiles'
    files = sorted(profile_dir.glob('*.json'))
    if not files:
        raise RuntimeError('No profile files found in set_network_src/profiles')

    rows = [run_one(path) for path in files]

    out_dir = ROOT / 'outputs' / 'runs'
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    txt_path = out_dir / f'set_network_suite_{ts}.txt'
    csv_path = out_dir / f'set_network_suite_{ts}.csv'

    with txt_path.open('w', encoding='utf-8', newline='\n') as f:
        f.write(f'SET-style network suite run: {ts}\n')
        f.write(f'profiles={len(rows)}\n\n')
        for row in rows:
            f.write(f"[{row['network']}]\n")
            f.write(f"  best_sub_batch: {row['best_sub_batch']}\n")
            f.write(f"  state_order: {row['state_order']}\n")
            f.write(f"  state_batches: {row['state_batches']}\n")
            f.write(f"  active_states: {row['active_states']}\n")
            f.write(f"  latency: {row['latency']}\n")
            f.write(f"  energy: {row['energy']}\n")
            f.write(f"  edp: {row['edp']}\n")
            f.write(f"  solver: {row['solver']}\n\n")

    fieldnames = [
        'network', 'profile_file', 'batch_size', 'best_sub_batch', 'state_order',
        'state_categories', 'state_batches', 'active_states', 'latency', 'energy',
        'edp', 'solver'
    ]
    with csv_path.open('w', encoding='utf-8', newline='\n') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f'txt={txt_path}')
    print(f'csv={csv_path}')


if __name__ == '__main__':
    main()
