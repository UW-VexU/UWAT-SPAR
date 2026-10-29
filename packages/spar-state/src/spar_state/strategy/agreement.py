"""Cohen's kappa between coders, and between coders and the pipeline.

Kappa needs categories, so each field is reduced to one label per code:
    sequence            the whole sequence
    per_robot_sequence  the robots' sequences, in sorted order
    vector              which entry is largest, or "none" when all are 0
    per_robot_simplex   each robot's largest part, in sorted order
    per_robot_time      "entered" or "never" per robot, in sorted order
Labels for robots are sorted so that robot order never counts as disagreement.
"""

from typing import Any

import numpy as np

from spar_schema.scene import StrategyCodeSpec, StrategyFieldSpec


def cohens_kappa(a: list, b: list) -> float:
    """Cohen's kappa for two raters' labels of the same items. NaN when it is undefined.

    It is undefined when both raters always give the same one label, so chance agreement is 1.
    """
    if len(a) != len(b) or not a:
        raise ValueError("kappa needs two equally long, non-empty lists of labels")
    observed = float(np.mean([x == y for x, y in zip(a, b, strict=True)]))
    labels = set(a) | set(b)
    chance = sum((a.count(label) / len(a)) * (b.count(label) / len(b)) for label in labels)
    if chance == 1.0:
        return float("nan")
    return (observed - chance) / (1.0 - chance)


def field_kappas(
    codes_a: list[dict[str, Any]], codes_b: list[dict[str, Any]], spec: StrategyCodeSpec
) -> dict[str, float]:
    """Kappa per field between two lists of codes for the same alliances, in the same order."""
    return {
        name: cohens_kappa(
            [category(code[name], field) for code in codes_a],
            [category(code[name], field) for code in codes_b],
        )
        for name, field in spec.fields.items()
    }


def category(value: Any, field: StrategyFieldSpec) -> Any:
    """One field's value reduced to a label that kappa can compare."""
    if field.type == "sequence":
        return tuple(value)
    if field.type == "per_robot_sequence":
        return tuple(sorted(tuple(robot) for robot in value))
    if field.type == "vector":
        return "none" if not any(value) else int(np.argmax(value))
    if field.type == "per_robot_simplex":
        parts = (field.model_extra or {}).get("parts") or list(range(len(value[0]) if value else 0))
        return tuple(sorted(parts[int(np.argmax(robot))] for robot in value))
    if field.type == "per_robot_time":
        return tuple(sorted("never" if v is None else "entered" for v in value))
    raise ValueError(f"unknown strategy code field type {field.type!r}")
