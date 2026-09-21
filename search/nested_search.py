"""Compose executable nested schedules at explicit sub-batch boundaries.

A child invocation processes exactly one parent sub-batch on its assigned tiles.
No resampling of unrelated parent/child state indices or free-boundary retry is
used. Half of each composite's activation budget is reserved for its boundary
buffers; the other half is shared by concurrent children by tile allocation.
The partition and budget split are explicit conservative search choices.
"""
from __future__ import annotations

from dataclasses import replace
from math import prod
import time

import numpy as np

from scheduler.block import Block, derive_block_dependencies, derive_block_edge_volumes
from scheduler.memory_table import MemoryOptimizationResult, MemoryTable, optimize_memory_table
from scheduler.paper_milp import ScTOptimizationResult, _integer_tile_allocation, _state_cost_coeffs
from scheduler.scheduling_table import SchedulingTable
from search.scheduler_search import (
    SearchConfig, SearchResult, _Candidate, _build_result_from_candidate,
    _combine_total_latency, _estimate_memory_cost, _flat_search_prepared,
)


def _children(block: Block) -> list[Block]:
    return list(block.sub_blocks) if block.sub_blocks else [Block(x.name, layers=[x]) for x in block.layers]


class NestedSearch:
    def __init__(self, *, depth: int = 6, buffer_fraction: float = 0.5, core_profile=None,
                 plans: tuple[str, ...] = ('serial', 'pipeline')):
        if depth < 1 or not 0 < buffer_fraction < 1:
            raise ValueError("depth must be positive and buffer fraction between zero and one")
        self.depth = depth
        self.core_profile = core_profile
        if not plans or not set(plans) <= {'serial', 'pipeline'}:
            raise ValueError('plans must contain serial and/or pipeline')
        self.plans = plans
        self.buffer_fraction = buffer_fraction
        self.cache: dict[tuple, SearchResult] = {}
        self.solver_calls = 0
        self.failures: list[str] = []
        self.unexpanded: set[str] = set()

    def _cost(self, block: Block, sub_batch: int, tiles: int, cfg: SearchConfig, depth: int):
        children = _children(block)
        if len(children) <= 1 or depth <= 1:
            if block.layer_count() > 1:
                self.unexpanded.add(block.name)
                if self.core_profile is not None:
                    raise RuntimeError("SET profile requires full expansion to individual layers; increase depth")
            if self.core_profile is not None and block.layer_count() == 1:
                layer = next(block.iter_layers())
                latency, energy = self.core_profile.evaluate(layer, sub_batch, tiles)
                return latency, energy, None
            lat, ene = _state_cost_coeffs([block.total_flops() * sub_batch],
                [block.aggregate_map_dims(sub_batch)], None, None, None, None, 1, tiles,
                cfg.compute_power_per_tile, cfg.compute_energy_per_op)
            return lat[0], ene[0], None
        child_cfg = replace(cfg, batch_size=sub_batch, num_pes=tiles,
            candidate_sub_batches=[x for x in range(1, sub_batch+1) if sub_batch % x == 0],
            use_all_sub_batch_factors=False, enable_chain_block_merge=False,
            derive_recursive_traces=False, enable_hierarchical_pipeline=False,
            sram_capacity=cfg.sram_capacity * (1-self.buffer_fraction) * tiles / cfg.num_pes,
            dram_capacity=cfg.dram_capacity * (1-self.buffer_fraction) * tiles / cfg.num_pes)
        key = (id(block), sub_batch, tiles, child_cfg.sram_capacity, child_cfg.dram_capacity, depth)
        if key not in self.cache:
            self.cache[key] = self._search(children, child_cfg, depth-1)
        result = self.cache[key]
        return result.total_latency, result.total_energy, result

    def _serial(self, blocks, cfg, sub_batch, lat, energy, inputs, weights, memory_cfg):
        n = len(blocks)
        repetitions = cfg.batch_size // sub_batch
        template = SchedulingTable(np.tril(np.ones((n,n))))
        volumes = [b.boundary_output_size() * sub_batch for b in blocks]
        dependencies = derive_block_dependencies(blocks)
        edges = {edge: v*sub_batch for edge,v in derive_block_edge_volumes(blocks).items()}
        memory = optimize_memory_table(template, volumes, dependencies,
            memory_cfg.sram_capacity, memory_cfg.dram_capacity,
            noc_bandwidth=cfg.noc_bandwidth, dram_bandwidth=cfg.dram_bandwidth,
            noc_energy_per_unit=cfg.noc_energy_per_unit, dram_energy_per_unit=cfg.dram_energy_per_unit,
            dram_noc_hops=cfg.dram_noc_hops, allow_fallback=False,
            block_input_volumes=[x*sub_batch for x in inputs], block_weight_volumes=weights,
            edge_volumes=edges, solver_time_limit_s=cfg.solver_time_limit_s,
            force_final_sram_empty=True)
        self.solver_calls += 1
        mem_l, mem_e = _estimate_memory_cost(template, memory.table, volumes, dependencies,
            cfg.noc_bandwidth, cfg.dram_bandwidth, cfg.noc_energy_per_unit,
            cfg.dram_energy_per_unit, cfg.dram_noc_hops, 1, 1,
            [x*sub_batch for x in inputs], weights, edges)
        # Flush dead intermediates at the end of each independent micro-batch.
        memory.table.sram[-1] = template.table[-1]
        memory.table.dram[-1] = template.table[-1]
        sct = SchedulingTable(np.concatenate([template.table+i for i in range(repetitions)]))
        met = MemoryTable(np.concatenate([memory.table.sram+i for i in range(repetitions)]),
                          np.concatenate([memory.table.dram+i for i in range(repetitions)]))
        total_l, total_e = sum(lat)*repetitions, sum(energy)*repetitions
        schedule = ScTOptimizationResult(sct, [1]*(n*repetitions), lat*repetitions,
            energy*repetitions, total_l*total_e, "serial-microbatch")
        memory = MemoryOptimizationResult(met, memory.objective*repetitions**2, memory.solver_name,
            replace(memory.report, incumbent=memory.report.incumbent*repetitions,
                    best_bound=memory.report.best_bound*repetitions) if memory.report else None)
        candidate = _Candidate(sub_batch, repetitions, schedule, total_l, total_e, total_l*total_e,
                               memory, mem_l*repetitions, mem_e*repetitions,
                               mem_l*mem_e*repetitions**2)
        return _build_result_from_candidate(candidate, [b.name for b in blocks], dependencies,
            [f"micro{k}:block{j}" for k in range(repetitions) for j in range(n)],
            ["serial"]*(n*repetitions), [cfg.num_pes]*(n*repetitions),
            [[j] for _ in range(repetitions) for j in range(n)], cfg.latency_combine_mode, 0,
            ["plan=serial_microbatch"], "root")

    def _search(self, blocks: list[Block], cfg: SearchConfig, depth: int) -> SearchResult:
        deps = derive_block_dependencies(blocks)
        n = len(blocks)
        composite = [len(_children(b)) > 1 and depth > 1 for b in blocks]
        memory_cfg = replace(cfg,
            sram_capacity=cfg.sram_capacity * (self.buffer_fraction if any(composite) else 1),
            dram_capacity=cfg.dram_capacity * (self.buffer_fraction if any(composite) else 1))
        inputs = [0.0 if nested else block.external_input_size() for block,nested in zip(blocks,composite)]
        weights = [0.0 if nested else block.weight_volume() for block,nested in zip(blocks,composite)]
        best = None
        for sb in cfg.candidate_sub_batches:
            sb = int(sb)
            if sb <= 0 or cfg.batch_size % sb:
                continue
            traces = []
            try:
                serial_l, serial_e = [], []
                for block in blocks:
                    latency, energy, child = self._cost(block, sb, cfg.num_pes, cfg, depth)
                    serial_l.append(latency); serial_e.append(energy)
                    if child:
                        traces.append({"parent_block":block.name, "parent_sub_batch":sb,
                            "child_batch_size":sb, "tiles":cfg.num_pes,
                            "child_sub_batch":child.best_sub_batch, "plan":"serial",
                            "latency":latency, "energy":energy,
                            "sram_budget_mb":cfg.sram_capacity*(1-self.buffer_fraction),
                            "solver_reports":child.solver_reports})
                serial = self._serial(blocks, cfg, sb, serial_l, serial_e, inputs, weights, memory_cfg)
                serial.hierarchy_traces.extend(traces)
                if best is None or serial.total_edp < best.total_edp:
                    best = serial
            except RuntimeError as exc:
                self.failures.append(f"depth={depth} sb={sb} serial: {exc}")
            # Canonical pipelining uses only states with actual available tiles.
            if 'pipeline' not in self.plans or n > 2*cfg.num_pes:
                continue  # Some middle blocks would have no active feasible state.
            try:
                latency_rows = [[None]*n for _ in range(2*n-1)]
                energy_rows = [[None]*n for _ in range(2*n-1)]
                traces = []
                for state in range(2*n-1):
                    active = [j for j in range(n) if j <= state < j+n]
                    if len(active) > cfg.num_pes:
                        continue
                    allocation, _ = _integer_tile_allocation(active, [b.total_flops() for b in blocks], cfg.num_pes)
                    for j, tiles in zip(active, allocation):
                        latency, energy, child = self._cost(blocks[j], sb, tiles, cfg, depth)
                        latency_rows[state][j] = latency
                        energy_rows[state][j] = energy
                        if child:
                            traces.append({"parent_block":blocks[j].name, "parent_state":state,
                                "parent_sub_batch":sb, "child_batch_size":sb,
                                "tiles":tiles, "child_sub_batch":child.best_sub_batch,
                                "plan":"pipeline", "latency":latency, "energy":energy,
                                "sram_budget_mb":cfg.sram_capacity*(1-self.buffer_fraction)*tiles/cfg.num_pes,
                                "solver_reports":child.solver_reports})
                fixed = replace(memory_cfg, candidate_sub_batches=[sb], use_all_sub_batch_factors=False,
                    derive_recursive_traces=False, enable_structure_refinement=False,
                    max_hierarchy_depth=1)
                pipeline = _flat_search_prepared(blocks, deps, fixed,
                    state_block_latency_override=latency_rows, state_block_energy_override=energy_rows,
                    block_input_volumes_override=inputs, block_weight_volumes_override=weights,
                    forbid_tile_overcommit=True, trace_path="root")
                self.solver_calls += 2
                pipeline.hierarchy_notes.append("plan=canonical_pipeline")
                pipeline.hierarchy_traces.extend(traces)
                if best is None or pipeline.total_edp < best.total_edp:
                    best = pipeline
            except RuntimeError as exc:
                self.failures.append(f"depth={depth} sb={sb} pipeline: {exc}")
        if best is None:
            raise RuntimeError("no feasible nested plan for the supplied batch and memory budget")
        return best

    def run(self, blocks: list[Block], config: SearchConfig) -> SearchResult:
        started = time.monotonic()
        result = self._search(blocks, config, self.depth)
        result.hierarchy_notes.extend([
            "hierarchy=parent_sub_batch_macros", f"buffer_fraction={self.buffer_fraction}",
            f"cached_child_invocations={len(self.cache)}", f"solver_calls={self.solver_calls}",
            f"search_seconds={time.monotonic()-started:.6f}",
            f"unexpanded_composite_blocks={len(self.unexpanded)}",
            f"infeasible_candidate_attempts={len(self.failures)}",
        ])
        return result
