"""Episode to strategy code per alliance, using the scene's strategy_code.yaml.

A strategy code is a dict from field name to value, with one value shape per field type:
    sequence            ["R1", "T_red_1"]          IDs in order
    per_robot_sequence  [["R1"], ["R3", "R4"]]     one sequence per robot
    vector              [0.9, 0.0, 0.0, 0.0]       `length` numbers
    per_robot_simplex   [[0.5, 0.5, 0, 0], ...]    per robot, shares of `parts` adding to 1
    per_robot_time      [94.0, None]               per robot, seconds or null
Robots are in no particular order; distances compare them in the order that matches best. The
scene's "strategy" plugin does the extraction: extract_code(episode, events, scene) returns a code
per alliance.
"""

import hashlib
import json
import math
from typing import Any

from spar_schema import EpisodeReader, Scene, StrategyRecord
from spar_schema.scene import StrategyCodeSpec

FIELD_TYPES = ("sequence", "per_robot_sequence", "vector", "per_robot_simplex", "per_robot_time")


def extract_codes(
    episode: EpisodeReader, events: list[dict[str, Any]], scene: Scene
) -> dict[str, dict[str, Any]]:
    """Each alliance's strategy code, from the scene's strategy plugin, checked."""
    codes = scene.plugin("strategy")(episode, events, scene)
    for alliance, code in codes.items():
        check_code(code, scene.strategy_code, label=alliance)
    return codes


def strategy_record(codes: dict[str, dict[str, Any]], scene: Scene, coder: str = "pipeline"):
    return StrategyRecord(codes=codes, coder=coder, code_version=code_version(scene))


def code_version(scene: Scene) -> str:
    """A tag that changes whenever the strategy code definition changes."""
    definition = json.dumps(scene.strategy_code.model_dump(), sort_keys=True)
    return f"{scene.name}-{hashlib.sha256(definition.encode()).hexdigest()[:12]}"


def check_code(code: dict[str, Any], spec: StrategyCodeSpec, label: str = "code") -> None:
    """Raise ValueError unless the code has every field, each in its type's shape."""
    if set(code) != set(spec.fields):
        raise ValueError(f"{label}: fields {sorted(code)}, expected {sorted(spec.fields)}")
    for name, field in spec.fields.items():
        value, where = code[name], f"{label}.{name}"
        extra = field.model_extra or {}
        if field.type == "sequence":
            _check_ids(value, where)
        elif field.type == "per_robot_sequence":
            _check_list(value, where)
            for robot in value:
                _check_ids(robot, where)
        elif field.type == "vector":
            _check_numbers(value, where, extra.get("length"))
        elif field.type == "per_robot_simplex":
            _check_list(value, where)
            for robot in value:
                _check_numbers(robot, where, len(extra.get("parts", [])) or None)
                if any(v < 0 for v in robot) or not math.isclose(sum(robot), 1.0, abs_tol=1e-6):
                    raise ValueError(f"{where}: each robot's shares must be 0 or more and add to 1")
        elif field.type == "per_robot_time":
            _check_list(value, where)
            if not all(v is None or isinstance(v, int | float) for v in value):
                raise ValueError(f"{where}: expected seconds or null per robot")
        else:
            raise ValueError(
                f"{where}: unknown field type {field.type!r}, expected one of {FIELD_TYPES}"
            )


def _check_list(value, where):
    if not isinstance(value, list):
        raise ValueError(f"{where}: expected a list")


def _check_ids(value, where):
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ValueError(f"{where}: expected a list of IDs")


def _check_numbers(value, where, length):
    if not isinstance(value, list) or not all(isinstance(v, int | float) for v in value):
        raise ValueError(f"{where}: expected a list of numbers")
    if length is not None and len(value) != length:
        raise ValueError(f"{where}: expected {length} numbers, got {len(value)}")
