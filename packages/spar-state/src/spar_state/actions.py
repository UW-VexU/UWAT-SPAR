"""Turns events into plan-level actions (actions.parquet), marked inferred.

infer_actions() asks the scene's actions plugin which action is behind each event, then lays each
robot's actions end to end: an action starts when the robot's previous one ended (the first at
the episode's first track) and ends at its event. A robot that places an object it was not seen
picking up gets a pick first, taking the first half of that time. Keys in args that start with
an underscore are notes, not arguments:
    _event    the event type the action came from
    _object   the object involved, when known
    _implied  true on a pick added before a place
Events without a robot, and events the plugin has no action for, are left out and counted. A
robot's last action is the last thing it was seen doing.
"""

import json
from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa

from spar_schema import (
    ACTIONS_SCHEMA,
    EpisodeReader,
    Scene,
    action_set,
    validate_table,
)
from spar_schema.start_state import AsOf


@dataclass
class InferredActions:
    rows: list[dict[str, Any]] = field(default_factory=list)  # actions.parquet rows, args as dicts
    unattributed: int = 0  # events with no robot
    unmapped: int = 0  # events with a robot but no action


def infer_actions(
    episode: EpisodeReader, events: list[dict[str, Any]], scene: Scene
) -> InferredActions:
    to_action = scene.plugin("actions")
    actions = action_set(scene, episode.meta.program)
    objects = AsOf(episode.objects, "object_id")
    robots = AsOf(episode.tracks, "agent_id")
    first = episode.tracks.column("t").to_numpy().min() if episode.tracks else 0.0
    result = InferredActions()
    ended: dict[str, float] = {}
    holding: dict[str, set[str]] = {}

    for event in sorted(events, key=lambda e: e["t"]):
        agent, t, object_id = event["agent_id"], float(event["t"]), event.get("object_id")
        if agent is None:
            result.unattributed += 1
            continue
        obj = _parsed(objects.latest(object_id, t)) if object_id else None
        pose = robots.latest(agent, t)
        xy = (pose["x"], pose["y"]) if pose else None
        found = _mapped(to_action(event, obj, xy, scene), actions)
        if found is None:
            result.unmapped += 1
            continue
        start = ended.get(agent, float(first))
        held = holding.setdefault(agent, set())
        notes = {"_event": event["type"]} | ({"_object": object_id} if object_id else {})
        if event["type"] == "place" and object_id not in held:
            pick = _mapped(to_action({**event, "type": "pick"}, obj, xy, scene), actions)
            if pick is not None:
                middle = (start + t) / 2
                implied = {"_event": "pick", "_object": object_id, "_implied": True}
                result.rows.append(_row(start, middle, agent, pick, implied))
                start = middle
        if event["type"] == "pick":
            held.add(object_id)
        elif event["type"] == "place":
            held.discard(object_id)
        result.rows.append(_row(start, t, agent, found, notes))
        ended[agent] = t
    return result


def actions_table(rows: list[dict[str, Any]]) -> pa.Table:
    """actions.parquet rows, with args stored as JSON."""
    table = pa.table(
        {
            name: [json.dumps(row[name]) if name == "args" else row[name] for row in rows]
            for name in ACTIONS_SCHEMA.names
        }
    )
    return validate_table(table, ACTIONS_SCHEMA, name="actions")


def _mapped(found, actions) -> tuple[str, dict[str, Any]] | None:
    """The plugin's (action, args), if the action set has it with those labels."""
    if found is None:
        return None
    try:
        actions.encode(*found)
    except KeyError:
        return None
    return found


def _row(start, end, agent, found, notes) -> dict[str, Any]:
    name, args = found
    return {
        "t_start": float(start),
        "t_end": float(end),
        "agent_id": agent,
        "action": name,
        "args": args | notes,
        "inferred": True,
    }


def _parsed(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return row | {"attrs": json.loads(row["attrs"]) if row.get("attrs") else {}}
