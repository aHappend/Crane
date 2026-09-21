import pytest

from model.layer import Layer
from scheduler.block import Block, derive_block_dependencies, merge_linear_blocks


def test_merge_preserves_workload_and_nested_ownership():
    layers = [Layer(str(i), 10 * (i + 1), i + 1) for i in range(4)]
    for parent, child in zip(layers, layers[1:]):
        parent.connect_to(child)
    original = [Block(layer.name, layers=[layer]) for layer in layers]
    merged, deps, mapping = merge_linear_blocks(
        original, derive_block_dependencies(original), max_layers_per_block=2
    )
    assert len(merged) == 2
    assert deps == [(0, 1)]
    assert mapping == {0: 0, 1: 0, 2: 1, 3: 1}
    assert [id(layer) for block in merged for layer in block.iter_layers()] == [id(x) for x in layers]
    assert sum(block.total_flops() for block in merged) == pytest.approx(100)
    assert sum(block.total_output_size() for block in merged) == pytest.approx(10)
    assert sum(block.layer_count() for block in merged) == 4
    # Merging composites must preserve each original layer exactly once too.
    nested, nested_deps, _ = merge_linear_blocks(merged, deps, max_layers_per_block=4)
    assert len(nested) == 1
    assert nested_deps == []
    assert nested[0].layer_count() == 4
    assert nested[0].total_flops() == pytest.approx(100)
    assert [id(layer) for layer in nested[0].iter_layers()] == [id(x) for x in layers]


def test_merge_preserves_fork_join_dependencies():
    layers = [Layer(name, 1, 1) for name in ("input", "left", "right", "join")]
    for p, c in ((0, 1), (0, 2), (1, 3), (2, 3)):
        layers[p].connect_to(layers[c])
    blocks = [Block(layer.name, layers=[layer]) for layer in layers]
    deps = derive_block_dependencies(blocks)
    merged, merged_deps, _ = merge_linear_blocks(blocks, deps, max_layers_per_block=4)
    assert len(merged) == 4
    assert merged_deps == [(0, 1), (0, 2), (1, 3), (2, 3)]
