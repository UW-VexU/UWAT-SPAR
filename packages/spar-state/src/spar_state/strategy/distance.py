"""Per-field distances (RapidFuzz edit distance, L1, absolute seconds) and the weighted total.

Each field's distance is between 0 (the same) and 1 (as different as that field can be):
    sequence            normalized edit distance (edits over the longer sequence's length)
    per_robot_sequence  the mean of the robots' edit distances
    vector              the mean absolute difference of its numbers, which are shares
    per_robot_simplex   the mean of half each robot's L1 distance between its shares
    per_robot_time      the mean of each robot's time difference over the match length, or the
                        field's missing_penalty when only one of the two has a time
Per-robot fields compare robots in whichever pairing is closest, since robot order means nothing,
and a missing robot counts as an empty sequence, even shares or no time. The code distance is the
weighted sum of the field distances (spec section 5.4). A weight or missing_penalty that is still
TODO counts as 1.
"""

from itertools import permutations
from typing import Any

import numpy as np
from rapidfuzz.distance import Levenshtein

from spar_schema.scene import StrategyCodeSpec, StrategyFieldSpec


def code_distance(
    a: dict[str, Any], b: dict[str, Any], spec: StrategyCodeSpec, match_length_s: float
) -> float:
    """The weighted sum of field distances between two codes."""
    distances = field_distances(a, b, spec, match_length_s)
    return float(sum(_number(spec.fields[name].weight) * d for name, d in distances.items()))


def field_distances(
    a: dict[str, Any], b: dict[str, Any], spec: StrategyCodeSpec, match_length_s: float
) -> dict[str, float]:
    return {
        name: field_distance(a[name], b[name], field, match_length_s)
        for name, field in spec.fields.items()
    }


def field_distance(a: Any, b: Any, field: StrategyFieldSpec, match_length_s: float) -> float:
    extra = field.model_extra or {}
    if field.type == "sequence":
        return float(Levenshtein.normalized_distance(a, b))
    if field.type == "per_robot_sequence":
        return _best_pairing(a, b, [], lambda x, y: Levenshtein.normalized_distance(x, y))
    if field.type == "vector":
        return float(np.mean(np.abs(np.asarray(a, float) - np.asarray(b, float))))
    if field.type == "per_robot_simplex":
        parts = len(extra.get("parts", [])) or len((a or b or [[1.0]])[0])
        even = [1.0 / parts] * parts
        return _best_pairing(
            a, b, even, lambda x, y: float(np.abs(np.asarray(x) - np.asarray(y)).sum()) / 2
        )
    if field.type == "per_robot_time":
        penalty = _number(extra.get("missing_penalty"))

        def time_distance(x, y):
            if x is None and y is None:
                return 0.0
            if x is None or y is None:
                return penalty
            return min(1.0, abs(x - y) / match_length_s)

        return _best_pairing(a, b, None, time_distance)
    raise ValueError(f"unknown strategy code field type {field.type!r}")


def _best_pairing(a: list, b: list, missing, distance) -> float:
    """The mean distance between robots, over the pairing of a's robots and b's that is closest."""
    n = max(len(a), len(b))
    if n == 0:
        return 0.0
    a = list(a) + [missing] * (n - len(a))
    b = list(b) + [missing] * (n - len(b))
    return float(
        min(
            np.mean([distance(x, y) for x, y in zip(a, order, strict=True)])
            for order in permutations(b)
        )
    )


def _number(value: Any) -> float:
    return float(value) if isinstance(value, int | float) else 1.0
