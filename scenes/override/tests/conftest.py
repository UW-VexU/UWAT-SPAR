"""A synthetic V5RC Override match where robots drive to Goals, a Loader, Toggles and the Midfield.

Used by the events and strategy tests. Times are seconds from match start; the match is 15 s of
autonomous and 105 s of driver control, and the data runs to 121 s.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest

from spar_schema import (
    OBJECTS_SCHEMA,
    TRACKS_SCHEMA,
    EpisodeKind,
    EpisodeMeta,
    EpisodeWriter,
    Outcome,
    RosterEntry,
    load_scene,
)

SCENE = load_scene(Path(__file__).parents[1])

# Waypoints (t, x, y): each robot drives in straight lines between them, then holds the last.
WAYPOINTS = {
    "red_1": [
        (0, -1.5, 1.0), (3, -1.45, 0.45), (6, -1.45, 0.45),  # to R1
        (9, -1.55, 0.05), (14, -1.55, 0.05),  # to the T_red_1 wall
        (30, -1.45, 0.45), (40, -1.45, 0.45),  # back to R1
        (110, 0.0, 0.2),  # into the Midfield
    ],
    "red_2": [
        (0, -1.5, -1.0), (5, -0.6, -1.45), (15, -0.6, -1.45),  # to R3
        (60, -1.45, -1.3), (70, -1.45, -1.3),  # to the near red Loader
        (80, -0.6, -1.45),  # back to R3
    ],
    "blue_1": [(0, 1.5, 1.0), (6, 1.45, 0.6), (100, 1.45, 0.6), (108, 0.1, 0.0)],  # B2, Midfield
    "blue_2": [(0, 1.5, -1.0), (50, 1.55, -0.05)],  # to the T_blue_1 wall
}  # fmt: skip

# (t, object_id, cls, status, goal_id, stack_index, attrs, holder)
OBJECTS = [
    (4.0, "p1", "pin", "placed", "R1", 0, {"halves": ["yellow", "red"]}, None),
    (7.0, "p3", "pin", "placed", "B2", 0, {"halves": ["yellow", "blue"]}, None),
    (8.0, "p2", "pin", "placed", "R3", 0, {"halves": ["yellow", "red"]}, None),
    (31.0, "p4", "pin", "held", None, None, {"halves": ["blue", "yellow"]}, "red_1"),
    (33.0, "c1", "cup", "placed", "R1", 1, {"gray": "up"}, None),
    (35.0, "p4", "pin", "placed", "R1", 2, {"halves": ["blue", "yellow"]}, None),
    (84.0, "c2", "cup", "placed", "R3", 1, {"gray": "up"}, None),
    (85.0, "p5", "pin", "placed", "R3", 2, {"halves": ["yellow", "red"]}, None),
]
# (t, Toggle, face): all start yellow; red_1 flips T_red_1 at 12 s, blue_2 flips T_blue_1 at 55 s.
TOGGLES = [(0.0, t, "yellow") for t in ("T_red_1", "T_red_2", "T_blue_1", "T_blue_2")]
TOGGLES += [(12.0, "T_red_1", "red"), (55.0, "T_blue_1", "blue")]
DATA_END = 121.0


def position(agent, t):
    points = WAYPOINTS[agent]
    times = [p[0] for p in points]
    return float(np.interp(t, times, [p[1] for p in points])), float(
        np.interp(t, times, [p[2] for p in points])
    )


def _table(schema, rows):
    return pa.table({name: [row.get(name) for row in rows] for name in schema.names})


@pytest.fixture
def override_match(tmp_path):
    """Write the match as an episode and return its path, with the scene and the inputs above."""
    tracks = []
    for step in range(int(DATA_END * 2) + 1):
        t = step / 2
        for agent in WAYPOINTS:
            x, y = position(agent, t)
            tracks.append(
                {
                    "t": t,
                    "agent_id": agent,
                    "x": x,
                    "y": y,
                    "heading": 0.0,
                    "provenance": "detected",
                }
            )
    objects = [
        {
            "t": t, "object_id": oid, "cls": cls, "status": status, "goal_id": goal,
            "stack_index": level, "attrs": json.dumps(attrs), "holder": holder,
            "provenance": "detected",
        }
        for t, oid, cls, status, goal, level, attrs, holder in OBJECTS
    ]  # fmt: skip
    objects += [
        {"t": t, "object_id": toggle, "cls": "toggle", "status": "unknown",
         "provenance": "detected",
         "attrs": json.dumps({"face": face, "seated": True, "touched": False})}
        for t, toggle, face in TOGGLES
    ]  # fmt: skip
    meta = EpisodeMeta(
        episode_id="override_SYNTH_Q2",
        kind=EpisodeKind.MATCH,
        scene="override",
        rules_version=SCENE.rules_version,
        program="v5rc",
        roster={agent: RosterEntry(alliance=agent.split("_")[0]) for agent in WAYPOINTS},
    )
    path = tmp_path / "ep"
    # The Override scene still has TODO values (the field frame, robot limits), so this is a draft.
    with EpisodeWriter(path, SCENE, meta, allow_incomplete_scene=True) as w:
        w.write_tracks(_table(TRACKS_SCHEMA, tracks))
        w.write_objects(_table(OBJECTS_SCHEMA, objects))
        w.write_outcome(Outcome(points_red=0, points_blue=0, autonomous_winner="red"))
    return SimpleNamespace(path=path, scene=SCENE, position=position)
