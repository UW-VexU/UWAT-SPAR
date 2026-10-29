"""StartState and its Parquet schema.

spar-data samples moments from real episodes into start_states.parquet. Simulators (spar-sim, and
the physics repo through env.reset_from) place robots and objects to match one of them.
EpisodeMoments gives the StartState of one episode at any time: each robot's pose and velocity
and every object's state, from the latest track and object row at or before it.
"""

import json
from bisect import bisect_right
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel

from spar_schema.io import EpisodeReader, read_table, validate_table
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
        pa.field("alliance", pa.string()),  # from the episode roster
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
        pa.field("program", pa.string()),
        pa.field("autonomous_winner", pa.string()),  # once the autonomous period is over
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
    alliance: str | None = None


class StartState(BaseModel):
    """One sampled moment from a real episode: everything a simulator needs to start from it."""

    episode_id: str
    t: float
    time_left_s: float
    robots: list[RobotState]
    objects: list[dict[str, Any]] = []  # one dict per object, keyed like objects.parquet columns
    program: str | None = None
    # The official autonomous winner, when the moment is after the autonomous period.
    autonomous_winner: str | None = None


class AsOf:
    """The latest row at or before a time, for each key, over one table; built once, asked often."""

    def __init__(self, table: pa.Table | None, key: str):
        self.rows = [] if table is None else table.to_pylist()
        self.times: dict[str, list[float]] = {}
        self.index: dict[str, list[int]] = {}
        for i in sorted(range(len(self.rows)), key=lambda i: self.rows[i]["t"]):
            name = self.rows[i][key]
            self.times.setdefault(name, []).append(self.rows[i]["t"])
            self.index.setdefault(name, []).append(i)

    def at(self, t: float) -> list[dict[str, Any]]:
        found = []
        for name, times in self.times.items():
            k = bisect_right(times, t)
            if k:
                found.append(self.rows[self.index[name][k - 1]])
        return found


class EpisodeMoments:
    """StartStates from one episode: EpisodeMoments(reader, scene).at(t)."""

    def __init__(self, reader: EpisodeReader, scene: Scene):
        self.meta = reader.meta
        periods = scene.programs[self.meta.program]
        if isinstance(periods.autonomous_s, str) or isinstance(periods.driver_s, str):
            raise ValueError(f"program {self.meta.program!r} still has TODO period lengths")
        self.autonomous_s = float(periods.autonomous_s)
        self.match_length = self.autonomous_s + float(periods.driver_s)
        outcome = reader.outcome
        self.autonomous_winner = outcome.autonomous_winner if outcome else None
        self.robots = AsOf(reader.tracks, "agent_id")
        self.objects = AsOf(reader.objects, "object_id")

    def at(self, t: float) -> StartState:
        roster = self.meta.roster
        return StartState(
            episode_id=self.meta.episode_id,
            t=t,
            time_left_s=self.match_length - t,
            robots=[
                _robot(row, roster[row["agent_id"]].alliance if row["agent_id"] in roster else None)
                for row in self.robots.at(t)
            ],
            objects=[_object(row) for row in self.objects.at(t)],
            program=self.meta.program,
            autonomous_winner=self.autonomous_winner if t >= self.autonomous_s else None,
        )


def _robot(row: dict[str, Any], alliance: str | None) -> RobotState:
    return RobotState(
        agent_id=row["agent_id"],
        x=row["x"],
        y=row["y"],
        heading=row["heading"],
        vx=row["vx"] or 0.0,
        vy=row["vy"] or 0.0,
        omega=row["omega"] or 0.0,
        alliance=alliance,
    )


def _object(row: dict[str, Any]) -> dict[str, Any]:
    kept = {k: v for k, v in row.items() if k not in ("t", "frame", "conf", "provenance")}
    kept["attrs"] = json.loads(row["attrs"]) if row.get("attrs") else {}
    return kept


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
