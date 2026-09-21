"""Load compiled SET network metadata without parsing C++ or inventing layers."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from math import prod
from pathlib import Path

from model.dag_parser import topological_sort
from model.layer import Layer
from scheduler.block import Block

DATA_PATH = Path(__file__).with_name("set_networks.json")


@lru_cache(maxsize=1)
def _metadata() -> dict:
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))


def available_models() -> tuple[str, ...]:
    return tuple(sorted(_metadata()["networks"]))


@dataclass
class NetworkWorkload:
    name: str
    layers: list[Layer]
    source: dict
    activation_bytes: int
    weight_bytes: int

    def blocks(self) -> list[Block]:
        return [Block(layer.name, layers=[layer]) for layer in self.layers]

    def manifest(self) -> dict:
        return {
            "name": self.name, "source": self.source,
            "definition_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
            "nodes": len(self.layers), "edges": sum(len(x.parents) for x in self.layers),
            "operations_per_sample": sum(x.flops for x in self.layers),
            "static_weight_mb": sum(x.weight_size for x in self.layers),
            "activation_bytes": self.activation_bytes, "weight_bytes": self.weight_bytes,
            "workload_basis": "per_sample", "ops_per_mac": 2,
            "graph_source": "compiled SET Network objects",
        }


def load_set_model(name: str, *, activation_bytes: int = 1, weight_bytes: int = 1) -> NetworkWorkload:
    if activation_bytes <= 0 or weight_bytes <= 0:
        raise ValueError("element sizes must be positive")
    data = _metadata()
    if name not in data["networks"]:
        raise ValueError(f"unknown model {name!r}; choose from {', '.join(available_models())}")
    rows = data["networks"][name]
    layers = []
    for i, row in enumerate(rows):
        if int(row["id"]) != i:
            raise ValueError("upstream node IDs must be contiguous")
        k, h, w = map(int, row["output_shape"])
        _, ih, iw = map(int, row["input_shape"])
        layer = Layer(
            name=str(row["name"]), flops=float(row["operations_per_sample"]),
            output_size=prod((k, h, w)) * activation_bytes / 1e6,
            op_type=str(row["op_type"]), map_dims=(float(k), 1.0, float(h), float(w)),
            input_size=int(row["external_input_channels"]) * ih * iw * activation_bytes / 1e6,
            weight_size=0.0 if row["weight_parents"] else float(row["weight_elements"]) * weight_bytes / 1e6,
            batch_dimension=1,
        )
        layers.append(layer)
        for parent in row["parents"]:
            parent = int(parent)
            if not 0 <= parent < i:
                raise ValueError("upstream graph is not a topologically ordered DAG")
            layers[parent].connect_to(layer)
    ordered = topological_sort(layers)
    if len({x.name for x in ordered}) != len(ordered):
        raise ValueError("upstream graph contains duplicate layer names")
    return NetworkWorkload(name, layers, data["source"], activation_bytes, weight_bytes)
