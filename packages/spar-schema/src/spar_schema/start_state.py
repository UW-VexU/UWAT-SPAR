"""StartState and its Parquet schema.

spar-data samples moments from real episodes into start_states.parquet. Simulators (spar-sim, and
the physics repo through env.reset_from) place robots and objects to match one of them.
"""

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel

from spar_schema.io import read_table, validate_table
from spar_schema.scene import Scene
from spar_schema.version import file_metadata

ROBOT_STATE_TYPE = pa.struct(
    [
        pa.field("agent_id", pa.string(), nullable=False),
        pa.field("x", pa.float32(), nullable=False),  # field position (m)
        pa.field("y", pa.float32(), nullable=False),
        pa.field("heading", pa.float32(), nullable=False),  # radians
        pa.field("vx", pa.float32(), nullable=False),  # m/s
        pa.field("vy", pa.float32(), nullable=False),
        pa.field("omega", pa.float32(), nullable=False),  # rad/s
    ]
)

# One row per sampled moment.
START_STATES_SCHEMA = pa.schema(
    [
        pa.field("episode_id", pa.string(), nullable=False),
        pa.field("t", pa.float64(), nullable=False),  # seconds from match start
        pa.field("time_left_s", pa.float64(), nullable=False),
        pa.field("robots", pa.list_(ROBOT_STATE_TYPE), nullable=False),
        pa.field("objects", pa.string(), nullable=False),  # JSON list, one dict per object
    ]
)


class RobotState(BaseModel):
    agent_id: str
    x: float
    y: float
    heading: float
    vx: float
    vy: float
    omega: float


class StartState(BaseModel):
    """One sampled moment from a real episode: everything a simulator needs to start from it."""

    episode_id: str
    t: float
    time_left_s: float
    robots: list[RobotState]
    objects: list[dict[str, Any]] = []  # one dict per object, keyed like objects.parquet columns


def write_start_states(
    path: str | Path,
    states: Iterable[StartState],
    scene: Scene,
    *,
    git_commit: str | None = None,
) -> None:
    rows = [
        {**state.model_dump(exclude={"objects"}), "objects": json.dumps(state.objects)}
        for state in states
    ]
    table = pa.Table.from_pylist(rows, schema=START_STATES_SCHEMA)
    table = validate_table(table, START_STATES_SCHEMA, name="start_states")
    metadata = file_metadata(scene.name, scene.rules_version, git_commit)
    pq.write_table(table.replace_schema_metadata(metadata), path)


def read_start_states(path: str | Path) -> list[StartState]:
    table = read_table(path, START_STATES_SCHEMA)
    return [
        StartState(**{**row, "objects": json.loads(row["objects"])})
        for row in table.select(START_STATES_SCHEMA.names).to_pylist()
    ]
