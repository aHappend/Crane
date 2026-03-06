from collections import deque
from typing import Dict, Iterable, List

from model.layer import Layer


def parse_layers(specs: Iterable[dict]) -> Dict[str, Layer]:
    """Build a DAG from list specs.

    Each spec:
    {
      "name": str,
      "flops": float,
      "output_size": float,
      "parents": [name, ...]
    }
    """

    layer_dict: Dict[str, Layer] = {}
    for spec in specs:
        layer_dict[spec["name"]] = Layer(
            name=spec["name"],
            flops=float(spec["flops"]),
            output_size=float(spec["output_size"]),
        )

    for spec in specs:
        child = layer_dict[spec["name"]]
        for parent_name in spec.get("parents", []):
            layer_dict[parent_name].connect_to(child)

    return layer_dict


def topological_sort(layers: Iterable[Layer]) -> List[Layer]:
    layer_list = list(layers)
    indegree = {layer.name: len(layer.parents) for layer in layer_list}
    by_name = {layer.name: layer for layer in layer_list}

    queue = deque([layer.name for layer in layer_list if indegree[layer.name] == 0])
    result: List[Layer] = []

    while queue:
        name = queue.popleft()
        node = by_name[name]
        result.append(node)
        for child in node.children:
            indegree[child.name] -= 1
            if indegree[child.name] == 0:
                queue.append(child.name)

    if len(result) != len(layer_list):
        raise ValueError("DAG contains cycles")

    return result
