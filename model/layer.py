from dataclasses import dataclass, field
from typing import List


@dataclass
class Layer:
    """Layer node in the DNN DAG."""

    name: str
    flops: float
    output_size: float
    parents: List["Layer"] = field(default_factory=list)
    children: List["Layer"] = field(default_factory=list)

    def connect_to(self, child: "Layer") -> None:
        if child not in self.children:
            self.children.append(child)
        if self not in child.parents:
            child.parents.append(self)
