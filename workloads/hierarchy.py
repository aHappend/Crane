"""Deterministic bounded-fanout hierarchy over an existing topological DAG.

Only containment changes: every layer and every original dependency is retained.
The generic partition is an explicit search starting point, not an imported
paper-author partition. Gradual partition/refinement remains in the scheduler.
"""
from __future__ import annotations

from math import ceil
from typing import Sequence

from model.layer import Layer
from scheduler.block import Block


def balanced_blocks(layers: Sequence[Layer], fanout: int = 8) -> list[Block]:
    if fanout < 2:
        raise ValueError("fanout must be at least two")
    if len(layers) <= fanout:
        return [Block(layer.name, layers=[layer]) for layer in layers]
    width = ceil(len(layers) / fanout)
    result = []
    for start in range(0, len(layers), width):
        group = layers[start:start + width]
        result.append(Block(f"B{start:04d}-{start+len(group)-1:04d}",
                            sub_blocks=balanced_blocks(group, fanout)))
    return result
