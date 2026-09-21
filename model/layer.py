from dataclasses import dataclass, field
from typing import List, Sequence


@dataclass(eq=False)
class Layer:
    """DAG node. FLOPs and activation sizes are always per input sample."""

    name: str
    flops: float
    output_size: float
    parents: List["Layer"] = field(default_factory=list)
    children: List["Layer"] = field(default_factory=list)
    op_type: str = "generic"
    map_dims: tuple[float, float, float, float] | None = None
    input_size: float | None = None
    weight_size: float = 0.0
    batch_dimension: int | None = None

    def connect_to(self, child: "Layer") -> None:
        if child not in self.children:
            self.children.append(child)
        if self not in child.parents:
            child.parents.append(self)

    def effective_map_dims(self, sub_batch: int = 1) -> tuple[float, float, float, float]:
        """Return four non-zero mapping dims for SET-style factorization.

        When explicit dims are unavailable, derive a stable fallback from FLOPs
        and output size so the evaluator still behaves deterministically.
        """

        if self.map_dims is not None:
            vals = [max(1.0, float(v)) for v in self.map_dims]
            if self.batch_dimension is not None:
                vals[self.batch_dimension] *= sub_batch
            return vals[0], vals[1], vals[2], vals[3]

        out = max(1.0, float(self.output_size))
        flops = max(1.0, float(self.flops))
        spatial = max(1.0, out ** 0.5)
        channel = max(1.0, flops / max(1.0, out * spatial))
        return (
            max(1.0, channel),
            max(1.0, spatial),
            max(1.0, spatial),
            max(1.0, flops / max(1.0, channel * spatial * spatial)),
        )

    @classmethod
    def with_map_dims(
        cls,
        name: str,
        flops: float,
        output_size: float,
        dims: Sequence[float],
        op_type: str = "generic",
    ) -> "Layer":
        if len(dims) != 4:
            raise ValueError("dims must contain exactly 4 values")
        return cls(
            name=name,
            flops=float(flops),
            output_size=float(output_size),
            op_type=str(op_type),
            map_dims=tuple(max(1.0, float(v)) for v in dims),
        )
