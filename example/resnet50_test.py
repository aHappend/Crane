import sys
from pathlib import Path

import numpy as np

ROOT = project_root_from(__file__, 1)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.layer import Layer
from scheduler.block import Block
from search.scheduler_search import SearchConfig, search_schedule


np.set_printoptions(precision=2, suppress=True)


def build_resnet50_style_blocks() -> list[Block]:
    conv1 = Layer("conv1", flops=3.0e9, output_size=20.0)
    conv2 = Layer("conv2", flops=4.0e9, output_size=25.0)
    conv3 = Layer("conv3", flops=4.5e9, output_size=24.0)
    conv4 = Layer("conv4", flops=5.2e9, output_size=22.0)
    conv5 = Layer("conv5", flops=5.8e9, output_size=21.0)

    conv1.connect_to(conv2)
    conv2.connect_to(conv3)
    conv3.connect_to(conv4)
    conv4.connect_to(conv5)

    block_a = Block("StageA", layers=[conv1, conv2])
    block_b = Block("StageB", layers=[conv3])
    block_c = Block("StageC", layers=[conv4, conv5])
    return [block_a, block_b, block_c]


def print_state_view(result) -> None:
    print("State details:")
    for i, (name, cat, pe, batch, active_blocks) in enumerate(
        zip(
            result.state_order,
            result.state_categories,
            result.active_pes,
            result.milp_solution.state_batches,
            result.state_active_blocks,
        )
    ):
        block_names = [result.scheduled_blocks[idx] for idx in active_blocks if 0 <= idx < len(result.scheduled_blocks)]
        print(
            f"  idx={i:<2d} name={name:<3s} cat={cat:<6s} active_pe={pe:<2d} "
            f"active_blocks={block_names} assigned_sub_batches={batch}"
        )


def main() -> None:
    blocks = build_resnet50_style_blocks()
    config = SearchConfig(
        batch_size=64,
        candidate_sub_batches=[2, 4, 8, 16],
        num_pes=4,
        sram_capacity=5000.0,
        dram_capacity=10000.0,
        enable_chain_block_merge=True,
        max_layers_per_block=3,
        min_layers_per_block=2,
        min_active_states=2,
        min_batch_if_active=1,
        max_state_share=0.6,
    )

    result = search_schedule(blocks, config)

    print(f"Best sub-batch = {result.best_sub_batch}")
    print(f"ScT solver     = {result.sct_solver_name}")
    print(f"MeT solver     = {result.met_solver_name}")
    print(f"State order    = {' -> '.join(result.state_order)}")
    print(f"State batches  = {result.milp_solution.state_batches}")
    print(f"ComputeLatency = {result.compute_latency:.4f}")
    print(f"ComputeEnergy  = {result.compute_energy:.4f}")
    print(f"MemoryLatency  = {result.memory_latency:.4f}")
    print(f"MemoryEnergy   = {result.memory_energy:.4f}")
    print(f"Latency        = {result.total_latency:.4f}")
    print(f"Energy         = {result.total_energy:.4f}")
    print(f"EDP            = {result.total_edp:.4f}")

    print_state_view(result)

    print("\nScheduled blocks (ScT columns):")
    for j, name in enumerate(result.scheduled_blocks):
        print(f"  col={j} block={name}")

    print("\nScT (rows=state, cols=block):")
    print(result.sct.table)

    print("\nMeT_S (rows=state, cols=block):")
    print(result.met.sram)

    print("\nMeT_D (rows=state, cols=block):")
    print(result.met.dram)


if __name__ == "__main__":
    main()





