from dataclasses import dataclass

import numpy as np


@dataclass
class SchedulingTable:
    table: np.ndarray

    @classmethod
    def zeros(cls, num_states: int, num_blocks: int) -> "SchedulingTable":
        return cls(table=np.zeros((num_states, num_blocks), dtype=float))

    @property
    def num_states(self) -> int:
        return self.table.shape[0]

    @property
    def num_blocks(self) -> int:
        return self.table.shape[1]

    def set(self, state: int, block: int, value: float) -> None:
        self.table[state, block] = value

    def get(self, state: int, block: int) -> float:
        return float(self.table[state, block])


def build_even_sct(num_states: int, block_workloads: list[float], total_sub_batches: int) -> SchedulingTable:
    """Create a cumulative ScT with state-wise even progress."""

    sct = SchedulingTable.zeros(num_states=num_states, num_blocks=len(block_workloads))
    cumulative = np.zeros(len(block_workloads), dtype=float)

    for i in range(num_states):
        share = (i + 1) / num_states
        target = np.floor(share * total_sub_batches)
        for b in range(len(block_workloads)):
            cumulative[b] = max(cumulative[b], target)
            sct.set(i, b, cumulative[b])

    return sct


def build_weighted_sct(num_states: int, block_workloads: list[float], total_sub_batches: int) -> SchedulingTable:
    """Create a cumulative ScT where block progress differs by workload.

    Heavier blocks progress slower in early states, and all blocks reach the same
    cumulative total at the final state.
    """

    num_blocks = len(block_workloads)
    sct = SchedulingTable.zeros(num_states=num_states, num_blocks=num_blocks)
    if num_states <= 1:
        sct.table[:, :] = float(total_sub_batches)
        return sct

    max_w = max(block_workloads) if block_workloads else 1.0
    norm = [w / max_w for w in block_workloads]
    # Lighter block => larger speed, heavier block => smaller speed.
    speed = [1.25 - 0.5 * n for n in norm]

    cumulative = np.zeros(num_blocks, dtype=float)
    for i in range(num_states):
        if i == 0:
            target_vec = np.zeros(num_blocks, dtype=float)
        elif i == num_states - 1:
            target_vec = np.full(num_blocks, float(total_sub_batches))
        else:
            base = i / (num_states - 1)
            target_vec = np.array(
                [
                    min(
                        float(total_sub_batches),
                        np.floor(base * total_sub_batches * speed[b]),
                    )
                    for b in range(num_blocks)
                ],
                dtype=float,
            )

        for b in range(num_blocks):
            cumulative[b] = max(cumulative[b], target_vec[b])
            sct.set(i, b, cumulative[b])

    return sct
