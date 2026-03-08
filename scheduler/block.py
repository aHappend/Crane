from dataclasses import dataclass, field
from typing import Iterator, List, Sequence

from model.layer import Layer


@dataclass
class Block:
    """Hierarchical block used by Crane scheduling."""

    name: str
    sub_blocks: List["Block"] = field(default_factory=list)
    layers: List[Layer] = field(default_factory=list)

    def add_layer(self, layer: Layer) -> None:
        self.layers.append(layer)

    def add_sub_block(self, block: "Block") -> None:
        self.sub_blocks.append(block)

    def iter_layers(self) -> Iterator[Layer]:
        for layer in self.layers:
            yield layer
        for sub in self.sub_blocks:
            yield from sub.iter_layers()

    def total_flops(self) -> float:
        return sum(layer.flops for layer in self.iter_layers())

    def total_output_size(self) -> float:
        return sum(layer.output_size for layer in self.iter_layers())

    def layer_count(self) -> int:
        return sum(1 for _ in self.iter_layers())

    def aggregate_map_dims(self) -> tuple[float, float, float, float]:
        layers = list(self.iter_layers())
        if not layers:
            return (1.0, 1.0, 1.0, 1.0)

        total_flops = sum(max(1.0, float(layer.flops)) for layer in layers)
        acc = [0.0, 0.0, 0.0, 0.0]
        for layer in layers:
            weight = max(1.0, float(layer.flops)) / max(1.0, total_flops)
            dims = layer.effective_map_dims()
            for i in range(4):
                acc[i] += weight * max(1.0, float(dims[i]))

        return tuple(max(1.0, v) for v in acc)


def derive_block_dependencies(blocks: Sequence[Block]) -> list[tuple[int, int]]:
    """Build block-level dependency edges from layer-level DAG links."""

    layer_to_block: dict[int, int] = {}
    for bi, block in enumerate(blocks):
        for layer in block.iter_layers():
            layer_to_block[id(layer)] = bi

    deps: set[tuple[int, int]] = set()
    for bi, block in enumerate(blocks):
        for layer in block.iter_layers():
            for child in layer.children:
                bj = layer_to_block.get(id(child))
                if bj is not None and bj != bi:
                    deps.add((bi, bj))

    return sorted(deps)


def merge_linear_blocks(
    blocks: Sequence[Block],
    dependencies: Sequence[tuple[int, int]],
    max_layers_per_block: int = 3,
    min_layers_per_block: int = 2,
) -> tuple[list[Block], list[tuple[int, int]], dict[int, int]]:
    """Merge consecutive linear-chain blocks to approximate graph partitioning."""

    n = len(blocks)
    if n == 0:
        return [], [], {}
    if max_layers_per_block <= 0:
        raise ValueError("max_layers_per_block must be > 0")

    dep_set = set(dependencies)
    parent_cnt = [0] * n
    child_cnt = [0] * n
    for p, c in dep_set:
        if 0 <= p < n and 0 <= c < n and p != c:
            child_cnt[p] += 1
            parent_cnt[c] += 1

    groups: list[list[int]] = []
    i = 0
    while i < n:
        group = [i]
        layer_budget = blocks[i].layer_count()
        j = i

        while j + 1 < n:
            nxt = j + 1
            can_chain = (
                (j, nxt) in dep_set
                and child_cnt[j] == 1
                and parent_cnt[nxt] == 1
                and layer_budget + blocks[nxt].layer_count() <= max_layers_per_block
            )
            if not can_chain:
                break
            group.append(nxt)
            layer_budget += blocks[nxt].layer_count()
            j = nxt

        if len(group) < min_layers_per_block:
            group = [i]
            j = i

        groups.append(group)
        i = j + 1

    merged_blocks: list[Block] = []
    old_to_new: dict[int, int] = {}
    for new_idx, grp in enumerate(groups):
        if len(grp) == 1:
            src = blocks[grp[0]]
            new_block = Block(name=src.name, layers=list(src.layers), sub_blocks=list(src.sub_blocks))
        else:
            merged_layers: list[Layer] = []
            merged_subs: list[Block] = []
            label_parts: list[str] = []
            for old_idx in grp:
                src = blocks[old_idx]
                merged_layers.extend(list(src.layers))
                merged_subs.append(src)
                label_parts.append(src.name)
            new_block = Block(name="|".join(label_parts), layers=merged_layers, sub_blocks=merged_subs)

        merged_blocks.append(new_block)
        for old_idx in grp:
            old_to_new[old_idx] = new_idx

    merged_deps: set[tuple[int, int]] = set()
    for p, c in dep_set:
        np = old_to_new.get(p)
        nc = old_to_new.get(c)
        if np is None or nc is None or np == nc:
            continue
        merged_deps.add((np, nc))

    return merged_blocks, sorted(merged_deps), old_to_new
