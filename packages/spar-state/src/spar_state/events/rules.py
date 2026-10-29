"""First version: events from object changes next to a robot.

object_events() turns objects.parquet status changes into events:
    pick     an object becomes held
    place    an object becomes placed (goal_id says where)
    remove   a placed object stops being placed
Each names the robot that did it: the row's holder when perception knows it, otherwise the nearest
robot within reach of where it happened. region_events() adds enter_<region> and leave_<region>
when a robot's center crosses into or out of one of the scene's field regions; a region nested in
a group (such as one Quadrant) is named by its own key. A robot already inside a region at its
first row gets no enter event. A scene's "events" plugin can add more; Override adds Toggle flips.
extract_events() runs them all, and events_table() makes events.parquet rows.
"""

import json
from collections.abc import Callable
from typing import Any

import numpy as np
import pyarrow as pa

from spar_schema import EVENTS_SCHEMA, EpisodeReader, EventSource, Scene, validate_table
from spar_schema.frames import polygon_distance
from spar_state.state import latest_rows

REACH_M = 0.6  # how far from a robot's center something can happen and still be its doing

Nearest = Callable[[float, tuple[float, float], float], str | None]


def extract_events(episode: EpisodeReader, scene: Scene) -> list[dict[str, Any]]:
    """Object, region and scene-specific events, in time order."""
    nearest = nearest_robot_finder(episode)
    found = object_events(episode, scene, nearest) + region_events(episode, scene)
    if scene.has_plugin("events"):
        found += scene.plugin("events")(episode, scene, nearest)
    return sorted(
        found, key=lambda event: (event["t"], event["type"], event.get("object_id") or "")
    )


def object_events(
    episode: EpisodeReader, scene: Scene, nearest: Nearest, reach_m: float = REACH_M
) -> list[dict[str, Any]]:
    if episode.objects is None:
        return []
    rows = sorted(episode.objects.to_pylist(), key=lambda row: (row["object_id"], row["t"]))
    found = []
    previous: dict[str, Any] | None = None
    for row in rows:
        if previous is None or previous["object_id"] != row["object_id"]:
            previous = {"status": "free", "holder": None, "goal_id": None}
        kind = _change(previous["status"], row["status"])
        if kind:
            goal_id = (
                row["goal_id"]
                if kind == "place"
                else previous["goal_id"]
                if kind == "remove"
                else None
            )
            agent = row["holder"] or (previous["holder"] if kind != "pick" else None)
            if agent is None:
                where = _position(row, goal_id, scene)
                agent = nearest(row["t"], where, reach_m) if where else None
            found.append(
                _event(
                    row["t"],
                    kind,
                    agent,
                    object_id=row["object_id"],
                    goal_id=goal_id,
                    conf=row["conf"],
                )
            )
        previous = row
    return found


def region_events(
    episode: EpisodeReader, scene: Scene, min_frames: int = 3
) -> list[dict[str, Any]]:
    """Enter and leave events, once a robot's center stays on the new side for min_frames rows."""
    if episode.tracks is None:
        return []
    regions = scene_regions(scene.spec.field.regions)
    tracks = episode.tracks.select(["t", "agent_id", "x", "y"]).to_pylist()
    by_agent: dict[str, list[dict[str, Any]]] = {}
    for row in sorted(tracks, key=lambda row: row["t"]):
        by_agent.setdefault(row["agent_id"], []).append(row)
    found = []
    for agent, rows in by_agent.items():
        for name, polygon in regions.items():
            inside = [polygon_distance(r["x"], r["y"], polygon) <= 0 for r in rows]
            state, run_start, run = inside[0], 0, 0
            for k, value in enumerate(inside):
                if value == state:
                    run = 0
                    continue
                if run == 0:
                    run_start = k
                run += 1
                if run >= min_frames:
                    state, run = value, 0
                    kind = f"{'enter' if value else 'leave'}_{name}"
                    found.append(_event(rows[run_start]["t"], kind, agent, target_id=name))
    return found


def nearest_robot_finder(episode: EpisodeReader) -> Nearest:
    """A function giving the robot nearest a point at a time, within reach, or None."""

    def nearest(t: float, xy: tuple[float, float], reach_m: float) -> str | None:
        robots = latest_rows(episode.tracks, "agent_id", t)
        if not robots:
            return None
        distances = [np.hypot(r["x"] - xy[0], r["y"] - xy[1]) for r in robots]
        best = int(np.argmin(distances))
        return robots[best]["agent_id"] if distances[best] <= reach_m else None

    return nearest


def events_table(events: list[dict[str, Any]]) -> pa.Table:
    """events.parquet rows for a list of events."""
    columns = EVENTS_SCHEMA.names
    table = pa.table({name: [event.get(name) for event in events] for name in columns})
    return validate_table(
        table, EVENTS_SCHEMA, name="events", allowed={"source": {s.value for s in EventSource}}
    )


def scene_regions(regions: dict[str, Any]) -> dict[str, list]:
    """The scene's region polygons by name, including those nested one level in a group."""
    found = {}
    for name, value in regions.items():
        if _is_polygon(value):
            found[name] = value
        elif isinstance(value, dict):
            found.update({key: item for key, item in value.items() if _is_polygon(item)})
    return found


def instance_positions(scene: Scene) -> dict[str, tuple[float, float]]:
    """Where each named field element is (goals, Toggles, ...), from objects.*.instances.*.xy."""
    return scene.instance_positions


def _change(before: str, after: str) -> str | None:
    if after == before:
        return None
    if after == "held":
        return "pick"
    if after == "placed":
        return "place"
    if before == "placed":
        return "remove"
    return None


def _position(row: dict[str, Any], goal_id: str | None, scene: Scene) -> tuple[float, float] | None:
    if row["x"] is not None and row["y"] is not None:
        return float(row["x"]), float(row["y"])
    return instance_positions(scene).get(goal_id) if goal_id else None


def _is_polygon(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) >= 3
        and all(isinstance(p, list | tuple) and len(p) == 2 for p in value)
    )


def _event(
    t, kind, agent, *, object_id=None, goal_id=None, target_id=None, conf=None, source="rule"
):
    return {
        "t": float(t),
        "agent_id": agent,
        "type": kind,
        "object_id": object_id,
        "goal_id": goal_id,
        "target_id": target_id,
        "conf": conf,
        "source": source,
    }


def parse_attrs(row: dict[str, Any]) -> dict[str, Any]:
    """An objects.parquet row's attrs, parsed from JSON."""
    return json.loads(row["attrs"]) if row.get("attrs") else {}
