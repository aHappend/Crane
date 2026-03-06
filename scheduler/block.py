from dataclasses import dataclass, field
from typing import Iterator, List

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
