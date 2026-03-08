import sys
from pathlib import Path

import numpy as np

ROOT = project_root_from(__file__, 1)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.layer import Layer
from scheduler.block import Block
from search.scheduler_search import SearchConfig, SearchResult, search_schedule


np.set_printoptions(precision=2, suppress=True)


def _print_summary(tag: str, result: SearchResult) -> None:
    print(f"\n=== {tag} ===")
    print(f"Best sub-batch = {result.best_sub_batch}")
    print(f"MILP backend   = {result.milp_solution.solver_name}")
    print(f"State order    = {' -> '.join(result.state_order)}")
    print(f"State batches  = {result.milp_solution.state_batches}")
    print(f"Latency        = {result.total_latency:.4f}")
    print(f"Energy         = {result.total_energy:.4f}")
    print(f"EDP            = {result.total_edp:.4f}")
    print("State details:")
    for i, (name, cat, pe, batch, unit_l, unit_e) in enumerate(
        zip(
            result.state_order,
            result.state_categories,
            result.active_pes,
            result.milp_solution.state_batches,
            result.state_unit_latency,
            result.state_unit_energy,
        )
    ):
        print(
            f"  idx={i:<2d} name={name:<3s} cat={cat:<6s} active_pe={pe:<2d} "
            f"unitL={unit_l:>12.2f} unitE={unit_e:>8.2f} assigned={batch}"
        )
    print("ScT (rows=state, cols=block):")
    print(result.sct.table)


def build_resnet101_like_blocks() -> list[Block]:
    stem = Block(
        "stem",
        layers=[
            Layer("conv1", flops=2.8e9, output_size=26.0),
            Layer("pool1", flops=0.5e9, output_size=24.0),
        ],
    )

    stage2 = Block(
        "stage2",
        layers=[
            Layer("res2_1", flops=3.6e9, output_size=28.0),
            Layer("res2_2", flops=3.4e9, output_size=28.0),
            Layer("res2_3", flops=3.2e9, output_size=27.0),
        ],
    )

    stage3 = Block(
        "stage3",
        layers=[
            Layer("res3_1", flops=5.0e9, output_size=25.0),
            Layer("res3_2", flops=4.8e9, output_size=25.0),
            Layer("res3_3", flops=4.7e9, output_size=24.0),
            Layer("res3_4", flops=4.6e9, output_size=24.0),
        ],
    )

    stage4 = Block(
        "stage4",
        layers=[
            Layer("res4_1", flops=6.1e9, output_size=22.0),
            Layer("res4_2", flops=5.9e9, output_size=22.0),
            Layer("res4_3", flops=5.7e9, output_size=21.0),
        ],
    )

    head = Block(
        "head",
        layers=[
            Layer("avgpool", flops=0.4e9, output_size=10.0),
            Layer("fc", flops=0.6e9, output_size=2.0),
        ],
    )

    return [stem, stage2, stage3, stage4, head]


def build_transformer_like_blocks() -> list[Block]:
    embed = Block(
        "embed",
        layers=[
            Layer("token_embedding", flops=1.8e9, output_size=35.0),
            Layer("pos_embedding", flops=0.4e9, output_size=35.0),
        ],
    )

    encoder1 = Block(
        "enc1",
        layers=[
            Layer("attn1", flops=4.2e9, output_size=40.0),
            Layer("mlp1", flops=6.8e9, output_size=42.0),
        ],
    )

    encoder2 = Block(
        "enc2",
        layers=[
            Layer("attn2", flops=4.0e9, output_size=40.0),
            Layer("mlp2", flops=6.5e9, output_size=41.0),
        ],
    )

    encoder3 = Block(
        "enc3",
        layers=[
            Layer("attn3", flops=3.8e9, output_size=39.0),
            Layer("mlp3", flops=6.2e9, output_size=40.0),
        ],
    )

    decoder_head = Block(
        "decoder_head",
        layers=[
            Layer("cross_attn", flops=4.5e9, output_size=36.0),
            Layer("lm_head", flops=2.2e9, output_size=12.0),
        ],
    )

    return [embed, encoder1, encoder2, encoder3, decoder_head]


def main() -> None:
    common_cfg = dict(
        batch_size=128,
        candidate_sub_batches=[2, 4, 8, 16, 32],
        num_pes=4,
        sram_capacity=15000.0,
        dram_capacity=30000.0,
        min_active_states=5,
        min_batch_if_active=1,
        max_state_share=0.45,
    )

    resnet_result = search_schedule(build_resnet101_like_blocks(), SearchConfig(**common_cfg))
    _print_summary("ResNet101-like", resnet_result)

    transformer_result = search_schedule(build_transformer_like_blocks(), SearchConfig(**common_cfg))
    _print_summary("Transformer-like", transformer_result)


if __name__ == "__main__":
    main()



