"""TaskState: the state a scene's score() reads, as named arrays that the scene declares.

scene.yaml declares the arrays in its `state` section:

    state:
      dims:                  # named axes: a size, or one label per index
        goal: [G1, G2, M]
        level: 16
      fields:                # one array per field; no dims means a scalar
        kind: {dims: [goal, level], dtype: int8, values: [empty, pin, cup]}
        winner: {dtype: int8, values: [none, red, blue, tie]}

`values` names the integer codes 0, 1, 2, ... of a field. A TaskState is a dict from field name
to an array of the declared shape, built with any array module (NumPy or JAX), so the same state
works in spar-state and inside the JAX simulator. state.parquet stores it as JSON.
"""

import json
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator

TaskState = dict[str, Any]

Dtype = Literal["bool", "int8", "int16", "int32", "float32"]

# Attribute names of each dtype on an array module such as numpy or jax.numpy.
_XP_DTYPES = {
    "bool": "bool_",
    "int8": "int8",
    "int16": "int16",
    "int32": "int32",
    "float32": "float32",
}


class StateFieldSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dims: list[str] = []
    dtype: Dtype
    values: list[str] | None = None  # labels of the integer codes 0, 1, 2, ...


class TaskStateSpec(BaseModel):
    """The `state` section of scene.yaml."""

    model_config = ConfigDict(extra="forbid")

    dims: dict[str, int | list[str]]
    fields: dict[str, StateFieldSpec]

    @model_validator(mode="after")
    def _check_fields(self) -> Self:
        for name, field in self.fields.items():
            if unknown := [dim for dim in field.dims if dim not in self.dims]:
                raise ValueError(f"state field {name!r} uses undeclared dims {unknown}")
            if field.values is not None and not field.dtype.startswith("int"):
                raise ValueError(f"state field {name!r} has values, so it needs an integer dtype")
        return self

    def size(self, dim: str) -> int:
        size = self.dims[dim]
        return size if isinstance(size, int) else len(size)

    def shape(self, field: str) -> tuple[int, ...]:
        return tuple(self.size(dim) for dim in self.fields[field].dims)

    def index(self, dim: str, label: str) -> int:
        """Position of a label along a labeled dim, for example the index of a goal."""
        labels = self.dims[dim]
        if isinstance(labels, int):
            raise ValueError(f"dim {dim!r} has no labels")
        return labels.index(label)

    def code(self, field: str, label: str) -> int:
        """Integer code of a label in a field's values."""
        values = self.fields[field].values
        if values is None:
            raise ValueError(f"state field {field!r} has no values")
        return values.index(label)

    def empty(self, xp: Any) -> TaskState:
        """A state with every array zero, built with the array module xp."""
        return {
            name: xp.zeros(self.shape(name), dtype=getattr(xp, _XP_DTYPES[field.dtype]))
            for name, field in self.fields.items()
        }

    def check(self, state: TaskState) -> None:
        """Raise ValueError unless the state has every field, with the declared shape, kind of
        dtype (bool, int or float) and codes."""
        if missing := sorted(set(self.fields) - set(state)):
            raise ValueError(f"state is missing fields {missing}")
        if unexpected := sorted(set(state) - set(self.fields)):
            raise ValueError(f"state has unexpected fields {unexpected}")
        for name, field in self.fields.items():
            array = state[name]
            if tuple(array.shape) != self.shape(name):
                raise ValueError(
                    f"state.{name}: shape {tuple(array.shape)}, expected {self.shape(name)}"
                )
            kind = str(array.dtype).rstrip("0123456789_")
            if kind != field.dtype.rstrip("0123456789"):
                raise ValueError(f"state.{name}: dtype {array.dtype}, expected {field.dtype}")
            if (
                field.values is not None
                and array.size
                and not (0 <= int(array.min()) and int(array.max()) < len(field.values))
            ):
                raise ValueError(f"state.{name}: codes outside 0..{len(field.values) - 1}")

    def to_json(self, state: TaskState) -> str:
        self.check(state)
        return json.dumps({name: state[name].tolist() for name in self.fields})

    def from_json(self, text: str, xp: Any) -> TaskState:
        data = json.loads(text)
        state = {
            name: xp.asarray(data[name], dtype=getattr(xp, _XP_DTYPES[field.dtype]))
            for name, field in self.fields.items()
            if name in data
        }
        self.check(state)
        return state
