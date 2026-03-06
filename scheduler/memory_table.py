from dataclasses import dataclass

import numpy as np

from scheduler.scheduling_table import SchedulingTable


@dataclass
class MemoryTable:
    """Track SRAM and DRAM low-water marks per state and block."""

    sram: np.ndarray
    dram: np.ndarray

    @classmethod
    def zeros(cls, num_states: int, num_blocks: int) -> "MemoryTable":
        return cls(
            sram=np.zeros((num_states, num_blocks), dtype=float),
            dram=np.zeros((num_states, num_blocks), dtype=float),
        )

    def set_sram(self, state: int, block: int, value: float) -> None:
        self.sram[state, block] = value

    def set_dram(self, state: int, block: int, value: float) -> None:
        self.dram[state, block] = value


def build_memory_table(sct: SchedulingTable, sram_keep_ratio: float = 0.6) -> MemoryTable:
    met = MemoryTable.zeros(sct.num_states, sct.num_blocks)
    for i in range(sct.num_states):
        for j in range(sct.num_blocks):
            cumulative = sct.get(i, j)
            met.set_sram(i, j, cumulative * (1.0 - sram_keep_ratio))
            met.set_dram(i, j, 0.0)
    return met
