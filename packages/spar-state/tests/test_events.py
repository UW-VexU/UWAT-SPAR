"""Generic event rules on the toy scene: object status changes and region crossings."""

import json
from pathlib import Path

import pyarrow as pa

from spar_schema import (
    EVENTS_SCHEMA,
    OBJECTS_SCHEMA,
    TRACKS_SCHEMA,
    EpisodeKind,
    EpisodeMeta,
    EpisodeReader,
    EpisodeWriter,
    RosterEntry,
    load_scene,
)
from spar_state.events.rules import events_table, extract_events, scene_regions

TOY_SCENE = Path(__file__).parents[2] / "spar-schema" / "tests" / "fixtures" / "toy_scene"
SCENE = load_scene(TOY_SCENE, strict=True)


def table(schema, rows):
    return pa.table({name: [row.get(name) for row in rows] for name in schema.names})


def track_rows():
    rows = []
    for t in range(11):
        # red_1 drives into the zone at t = 4 and out again at t = 8.
        x = -1.0 if t < 4 else (0.0 if t < 8 else 1.0)
        rows.append({"t": float(t), "agent_id": "red_1", "x": x, "y": 0.0})
        # blue_1 sits outside the zone, except for one blip at t = 5.
        blue = (0.0, 0.0) if t == 5 else (0.8, 0.8)
        rows.append({"t": float(t), "agent_id": "blue_1", "x": blue[0], "y": blue[1]})
    for row in rows:
        row.update(heading=0.0, provenance="detected")
    return rows


def object_rows():
    rows = [
        {"t": 0.0, "object_id": "ball_1", "status": "free", "x": -1.0, "y": 0.1},
        {"t": 1.0, "object_id": "ball_1", "status": "held", "holder": "red_1"},
        {
            "t": 2.0,
            "object_id": "ball_1",
            "status": "placed",
            "goal_id": "red",
            "x": -0.9,
            "y": 0.0,
        },
        {"t": 6.0, "object_id": "ball_1", "status": "free", "x": 0.75, "y": 0.75},
        {
            "t": 3.0,
            "object_id": "ball_2",
            "status": "placed",
            "goal_id": "blue",
            "x": 5.0,
            "y": 5.0,
        },
    ]
    for row in rows:
        row.update(cls="ball", provenance="detected", attrs=json.dumps({}))
    return rows


def write_episode(path):
    meta = EpisodeMeta(
        episode_id="toy_E2_Q1",
        kind=EpisodeKind.MATCH,
        scene="toy",
        rules_version="v1",
        program="solo",
        roster={"red_1": RosterEntry(alliance="red"), "blue_1": RosterEntry(alliance="blue")},
    )
    with EpisodeWriter(path, SCENE, meta) as w:
        w.write_tracks(table(TRACKS_SCHEMA, track_rows()))
        w.write_objects(table(OBJECTS_SCHEMA, object_rows()))
    return EpisodeReader(path)


def summary(events):
    return [(e["t"], e["type"], e["agent_id"], e["object_id"], e["goal_id"]) for e in events]


def test_object_changes_become_events_credited_to_a_robot(tmp_path):
    events = [e for e in extract_events(write_episode(tmp_path / "ep"), SCENE) if e["object_id"]]
    assert summary(events) == [
        (1.0, "pick", "red_1", "ball_1", None),  # the row names the holder
        (2.0, "place", "red_1", "ball_1", "red"),  # the robot that was holding it
        (3.0, "place", None, "ball_2", "blue"),  # nobody within reach
        (6.0, "remove", "blue_1", "ball_1", "red"),  # the nearest robot
    ]


def test_region_crossings_become_events(tmp_path):
    events = [e for e in extract_events(write_episode(tmp_path / "ep"), SCENE) if e["target_id"]]
    assert [(e["t"], e["type"], e["agent_id"]) for e in events] == [
        (4.0, "enter_zone", "red_1"),
        (8.0, "leave_zone", "red_1"),
    ]  # blue_1's one-row blip does not count


def test_events_are_written_as_events_parquet(tmp_path):
    episode = write_episode(tmp_path / "ep")
    rows = events_table(extract_events(episode, SCENE))
    assert rows.schema.equals(EVENTS_SCHEMA)
    with EpisodeWriter(tmp_path / "ep", SCENE, episode.meta, exist_ok=True) as w:
        w.write_events(rows)
    assert EpisodeReader(tmp_path / "ep").events.num_rows == len(rows)


def test_regions_nested_in_a_group_are_named_by_their_key():
    square = [[0, 0], [1, 0], [1, 1]]
    regions = {
        "midfield": square,
        "quadrants": {"a": square, "b": square},
        "line": {"points": [[0, 0], [1, 1]]},
    }
    assert sorted(scene_regions(regions)) == ["a", "b", "midfield"]
