"""Events to plan-level actions on the toy scene, and actions.parquet from spar state run."""

import json
from pathlib import Path

import pyarrow as pa
import pytest

from spar_schema import (
    OBJECTS_SCHEMA,
    TRACKS_SCHEMA,
    EpisodeKind,
    EpisodeMeta,
    EpisodeReader,
    EpisodeWriter,
    Outcome,
    RosterEntry,
    load_scene,
)
from spar_state.actions import infer_actions
from spar_state.run import run_episode

TOY_SCENE = Path(__file__).parents[2] / "spar-schema" / "tests" / "fixtures" / "toy_scene"
SCENE = load_scene(TOY_SCENE, strict=True)


def table(schema, rows):
    return pa.table({name: [row.get(name) for row in rows] for name in schema.names})


@pytest.fixture
def episode(tmp_path):
    """red_1 waits by the red goal; one ball is picked up and dropped in, then another, never seen
    picked up, is dropped in too. blue_1 drives into the zone at 8 s."""
    tracks = []
    for t in range(13):
        blue_x = 0.7 if t < 8 else 0.0
        tracks += [
            {"t": float(t), "agent_id": "red_1", "x": -0.6, "y": 0.0},
            {"t": float(t), "agent_id": "blue_1", "x": blue_x, "y": 0.0},
        ]
    for row in tracks:
        row.update(heading=0.0, provenance="detected")
    objects = [
        {"t": 0.0, "object_id": "a", "cls": "ball", "status": "free", "x": -0.5, "y": 0.0},
        {"t": 2.0, "object_id": "a", "cls": "ball", "status": "held", "holder": "red_1"},
        {"t": 4.0, "object_id": "a", "cls": "ball", "status": "placed", "goal_id": "red"},
        {"t": 6.0, "object_id": "b", "cls": "ball", "status": "placed", "goal_id": "red"},
        # c lands in the blue goal with no robot near it.
        {"t": 7.0, "object_id": "c", "cls": "ball", "status": "placed", "goal_id": "blue"},
    ]
    for row in objects:
        row["provenance"] = "detected"
    meta = EpisodeMeta(
        episode_id="toy_actions",
        kind=EpisodeKind.MATCH,
        scene="toy",
        rules_version="v1",
        program="solo",
        roster={"red_1": RosterEntry(alliance="red"), "blue_1": RosterEntry(alliance="blue")},
    )
    path = tmp_path / "ep"
    with EpisodeWriter(path, SCENE, meta) as w:
        w.write_tracks(table(TRACKS_SCHEMA, tracks))
        w.write_objects(table(OBJECTS_SCHEMA, objects))
        w.write_outcome(Outcome(points_red=2, points_blue=1, autonomous_winner="red"))
    return path


def event(t, kind, agent, **fields):
    return {"t": t, "type": kind, "agent_id": agent, "object_id": None, "goal_id": None} | fields


def test_actions_are_laid_end_to_end(episode):
    events = [
        event(2.0, "pick", "red_1", object_id="a"),
        event(4.0, "place", "red_1", object_id="a", goal_id="red"),
        event(6.0, "place", "red_1", object_id="b", goal_id="red"),
        event(7.0, "place", None, object_id="c", goal_id="blue"),
        event(8.0, "enter_zone", "blue_1", target_id="zone"),
        event(9.0, "leave_zone", "blue_1", target_id="zone"),
    ]
    found = infer_actions(EpisodeReader(episode), events, SCENE)
    rows = [(r["agent_id"], r["t_start"], r["t_end"], r["action"]) for r in found.rows]
    assert rows == [
        ("red_1", 0.0, 2.0, "pick"),
        ("red_1", 2.0, 4.0, "place"),
        ("red_1", 4.0, 5.0, "pick"),  # implied: b was never seen picked up
        ("red_1", 5.0, 6.0, "place"),
        ("blue_1", 0.0, 8.0, "park"),
    ]
    assert found.rows[1]["args"] == {"goal": "red", "_event": "place", "_object": "a"}
    assert found.rows[2]["args"]["_implied"] is True
    assert "_implied" not in found.rows[0]["args"]
    assert all(r["inferred"] for r in found.rows)
    assert (found.unattributed, found.unmapped) == (1, 1)  # c's place, and leaving the zone


def test_unknown_labels_are_left_out(episode):
    events = [event(4.0, "place", "red_1", object_id="a", goal_id="green")]
    found = infer_actions(EpisodeReader(episode), events, SCENE)
    assert found.rows == [] and found.unmapped == 1


def test_run_writes_actions(episode):
    result = run_episode(episode, SCENE)
    actions = EpisodeReader(episode).actions.to_pylist()
    assert len(actions) == len(result.actions.rows) > 0
    assert json.loads(actions[0]["args"])["_event"] == "pick"
    assert {row["action"] for row in actions} >= {"pick", "place", "park"}
