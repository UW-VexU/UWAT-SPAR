"""Builds task state over time from field tracks and object tracks.

At any moment, the latest track row for each robot and each object at or before that time makes a
Snapshot, and the scene's state plugin turns the Snapshot into the scene's TaskState.
"""

import json
from typing import Any

import numpy as np
import pyarrow as pa

from spar_schema import STATE_SCHEMA, EpisodeReader, Scene, Snapshot, validate_table


def latest_rows(table: pa.Table | None, key: str, t: float) -> list[dict[str, Any]]:
    """The last row at or before t for each value of the `key` column, as dicts."""
    if table is None or table.num_rows == 0:
        return []
    times = table.column("t").to_numpy()
    keys = np.asarray(table.column(key).to_pylist(), dtype=str)
    rows = np.flatnonzero(times <= t)
    if rows.size == 0:
        return []
    order = rows[np.lexsort((times[rows], keys[rows]))]  # by key, then by time
    last = order[np.r_[keys[order][1:] != keys[order][:-1], True]]
    return table.take(last).to_pylist()


def data_end(episode: EpisodeReader) -> float:
    """The time of the last track or object row in the episode."""
    ends = [
        float(table.column("t").to_numpy().max())
        for table in (episode.tracks, episode.objects)
        if table
    ]
    if not ends:
        raise ValueError(f"{episode.path} has no tracks or objects")
    return max(ends)


def snapshot_at(
    episode: EpisodeReader,
    t: float,
    *,
    checkpoint: str | None = None,
    autonomous_winner: str = "none",
) -> Snapshot:
    """The match at time t: each robot's and each object's latest row at or before t."""
    alliance = {agent: entry.alliance for agent, entry in episode.meta.roster.items()}
    robots = [
        {**row, "alliance": alliance.get(row["agent_id"])}
        for row in latest_rows(episode.tracks, "agent_id", t)
    ]
    objects = [
        {**row, "attrs": json.loads(row["attrs"]) if row.get("attrs") else {}}
        for row in latest_rows(episode.objects, "object_id", t)
    ]
    return Snapshot(
        t=t,
        checkpoint=checkpoint,
        program=episode.meta.program,
        autonomous_winner=autonomous_winner,
        robots=robots,
        objects=objects,
    )


def states_table(
    rows: list[tuple[float, str | None, dict[str, float], str]], scene: Scene
) -> pa.Table:
    """state.parquet rows from (t, checkpoint, points per alliance, state JSON) tuples."""
    table = pa.table(
        {
            "t": [row[0] for row in rows],
            "checkpoint": [row[1] for row in rows],
            "points_red": [row[2]["red"] for row in rows],
            "points_blue": [row[2]["blue"] for row in rows],
            "state": [row[3] for row in rows],
        }
    )
    return validate_table(
        table, STATE_SCHEMA, name="state", allowed={"checkpoint": set(scene.checkpoints)}
    )
