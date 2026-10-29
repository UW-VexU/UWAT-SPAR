"""The Override state plugin, and the score check on a synthetic Override match."""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from spar_schema import (
    EpisodeKind,
    EpisodeMeta,
    EpisodeReader,
    EpisodeWriter,
    Outcome,
    RosterEntry,
    Snapshot,
    load_scene,
)
from spar_state.run import run_episode

SCENE = load_scene(Path(__file__).parents[1])
SPEC = SCENE.task_state


def pin(goal, level, lower, upper, name=None):
    attrs = {"halves": [lower, upper]}
    return {
        "object_id": name or f"pin_{goal}_{level}",
        "cls": "pin",
        "goal_id": goal,
        "stack_index": level,
        "attrs": attrs,
    }


def cup(goal, level, gray, name=None):
    attrs = {"gray": gray}
    return {
        "object_id": name or f"cup_{goal}_{level}",
        "cls": "cup",
        "goal_id": goal,
        "stack_index": level,
        "attrs": attrs,
    }


def toggle(toggle_id, face, touched=False, seated=True):
    attrs = {"face": face, "touched": touched, "seated": seated}
    return {
        "object_id": toggle_id,
        "cls": "toggle",
        "goal_id": None,
        "stack_index": None,
        "attrs": attrs,
    }


def robot(agent, alliance, x, y, radius=None):
    return {"agent_id": agent, "alliance": alliance, "x": x, "y": y, "radius_m": radius}


def snap(objects=(), robots=(), checkpoint="match_end", program="v5rc", winner="none"):
    return Snapshot(
        t=0.0, checkpoint=checkpoint, program=program, autonomous_winner=winner,
        robots=list(robots), objects=list(objects),
    )  # fmt: skip


def points(state):
    result = SCENE.score(state, np)
    return float(result["red"]), float(result["blue"])


def test_stack_objects_fill_the_goal_levels():
    state = SCENE.build_state(
        snap([pin("R4", 0, "yellow", "red"), cup("R4", 1, "up"), pin("R4", 2, "blue", "red")])
    )
    g = SPEC.index("goal", "R4")
    kinds = [SPEC.fields["kind"].values[k] for k in state["kind"][g, :4]]
    assert kinds == ["pin", "cup", "pin", "empty"]
    assert not state["cup_opaque_lower"][g, 1]
    assert points(state) == (10, 0)


def test_manual_worked_example_from_objects():
    objects = []
    for i, upper in enumerate(["red", "red", "blue", "yellow", "yellow", "red", "yellow"]):
        if i:
            objects.append(cup("R1", 2 * i - 1, "up"))
        objects.append(pin("R1", 2 * i, "yellow", upper))
    for face, expected in [("yellow", (15, 5)), ("red", (45, 5)), ("blue", (15, 35))]:
        assert points(SCENE.build_state(snap([*objects, toggle("T_red_1", face)]))) == expected, (
            face
        )


def test_two_objects_at_one_level_mark_a_crowded_socket():
    objects = [
        pin("B2", 0, "yellow", "blue"),
        cup("B2", 1, "up"),
        pin("B2", 2, "yellow", "blue", name="first"),
        pin("B2", 2, "yellow", "blue", name="second"),
    ]
    state = SCENE.build_state(snap(objects))
    assert state["crowded"][SPEC.index("goal", "B2"), 2]
    assert points(state) == (0, 5)


def test_touched_toggle_is_not_set():
    state = SCENE.build_state(
        snap([pin("R1", 0, "red", "yellow"), toggle("T_red_1", "red", touched=True)])
    )
    assert not state["toggle_set"][SPEC.index("quadrant", "red_1")]
    assert points(state) == (0, 0)


def test_toggles_without_rows_count_as_yellow():
    state = SCENE.build_state(snap([pin("R1", 0, "red", "yellow")]))
    assert not state["toggle_set"].any()


@pytest.mark.parametrize(
    "x, program, radius, counted",
    [
        (0.0, "v5rc", None, True),  # center inside
        (0.75, "v5rc", None, True),  # center 0.15 m outside, within half of 18 in
        (0.95, "v5rc", None, False),  # 0.35 m outside
        (0.95, "v5rc", 0.4, True),  # a wider measured footprint reaches in
        (0.80, "vexu", None, False),  # 0.20 m outside: beyond half of 15 in
        (0.80, "v5rc", None, True),  # but within half of 18 in
    ],
)
def test_robot_counts_in_the_midfield_when_it_reaches_in(x, program, radius, counted):
    state = SCENE.build_state(
        snap(robots=[robot("blue_1", "blue", x, 0.0, radius)], program=program)
    )
    assert state["robots_in_midfield"].tolist() == [0, 1 if counted else 0]


def test_snapshot_context_becomes_state_codes():
    state = SCENE.build_state(snap(checkpoint="autonomous_end", program="vexu", winner="tie"))
    assert int(state["checkpoint"]) == SPEC.code("checkpoint", "autonomous_end")
    assert int(state["program"]) == SPEC.code("program", "vexu")
    assert int(state["autonomous_winner"]) == SPEC.code("autonomous_winner", "tie")
    SPEC.check(state)


def test_stack_taller_than_the_state_holds_raises():
    with pytest.raises(ValueError, match="level 16"):
        SCENE.build_state(snap([pin("R1", 16, "red", "red")]))


def test_states_are_scored_at_a_checkpoint():
    with pytest.raises(ValueError, match="checkpoint"):
        SCENE.build_state(snap(checkpoint=None))


# A synthetic V5RC match, through the real score check.
# Autonomous (0-15 s): red Pins on R2 and R3, a blue Pin on B2. Red 10, blue 5, so red wins.
# Driver: a gray-up Cup and a blue/yellow Pin on R2, T_red_1 set to red, a red/yellow Pin on M.
# End: red_1 in the Midfield; blue_1 inside and blue_2 partly inside; red_2 far away.
# Final, by hand: red = 5 (R2 bottom) + 10 (R2 yellow, Toggle red) + 5 (R3) + 8 (one Robot)
# + 12 (Autonomous Bonus) = 40. Blue = 5 (B2) + 10 (Midfield yellow, more Robots) + 16 = 31.
OBJECTS = [
    (5.0, pin("R2", 0, "yellow", "red")),
    (8.0, pin("R3", 0, "yellow", "red")),
    (12.0, pin("B2", 0, "yellow", "blue")),
    (40.0, cup("R2", 1, "up")),
    (45.0, pin("R2", 2, "blue", "yellow")),
    (60.0, toggle("T_red_1", "red")),
    (90.0, pin("M", 0, "red", "yellow")),
]
START = {"red_1": (-1.5, 1.0), "red_2": (-1.5, -1.0), "blue_1": (1.5, 1.0), "blue_2": (1.5, -1.0)}
END = {"red_1": (0.0, 0.0), "red_2": (-1.5, -1.5), "blue_1": (0.1, 0.1), "blue_2": (-0.7, 0.0)}
DATA_END = 121.0  # the match ends at 120 s; the video stops before the 5 s scoring delay


def objects_table():
    rows = [(0.0, toggle(t, "yellow")) for t in ("T_red_1", "T_red_2", "T_blue_1", "T_blue_2")]
    rows += OBJECTS
    n = len(rows)
    return pa.table(
        {
            "t": [r[0] for r in rows],
            "frame": [int(r[0] * 30) for r in rows],
            "object_id": [r[1]["object_id"] for r in rows],
            "cls": [r[1]["cls"] for r in rows],
            "x": [None] * n,
            "y": [None] * n,
            "z": [None] * n,
            "status": ["placed" if r[1]["goal_id"] else "unknown" for r in rows],
            "holder": [None] * n,
            "goal_id": [r[1]["goal_id"] for r in rows],
            "stack_index": [r[1]["stack_index"] for r in rows],
            "attrs": [json.dumps(r[1]["attrs"]) for r in rows],
            "conf": [None] * n,
            "provenance": ["detected"] * n,
        }
    )


def tracks_table():
    rows = []
    for step in range(int(DATA_END * 2) + 1):
        t = step / 2
        for agent in START:
            x, y = (END if t >= 100 else START)[agent]
            rows.append((t, agent, x, y))
    n = len(rows)
    return pa.table(
        {
            "t": [r[0] for r in rows],
            "frame": [int(r[0] * 30) for r in rows],
            "agent_id": [r[1] for r in rows],
            "x": [r[2] for r in rows],
            "y": [r[3] for r in rows],
            "heading": [0.0] * n,
            "vx": [0.0] * n,
            "vy": [0.0] * n,
            "omega": [0.0] * n,
            "cov": [None] * n,
            "conf": [None] * n,
            "provenance": ["detected"] * n,
        }
    )


def write_match(path, outcome):
    meta = EpisodeMeta(
        episode_id="override_SYNTH_Q1",
        kind=EpisodeKind.MATCH,
        scene="override",
        rules_version=SCENE.rules_version,
        program="v5rc",
        roster={agent: RosterEntry(alliance=agent.split("_")[0]) for agent in START},
    )
    # The Override scene still has TODO values (the field frame, robot limits), so this is a draft.
    with EpisodeWriter(path, SCENE, meta, allow_incomplete_scene=True) as w:
        w.write_tracks(tracks_table())
        w.write_objects(objects_table())
        w.write_outcome(outcome)
    return path


def test_synthetic_match_passes_the_score_check(tmp_path):
    path = write_match(
        tmp_path / "ep", Outcome(points_red=40, points_blue=31, autonomous_winner="red")
    )
    checks = run_episode(path, SCENE, allow_incomplete_scene=True).checks
    assert checks["autonomous_end"].computed == {"red": 10.0, "blue": 5.0}
    assert checks["autonomous_end"].passed is True
    assert checks["match_end"].computed == {"red": 40.0, "blue": 31.0}
    assert checks["match_end"].passed is True
    episode = EpisodeReader(path)
    assert episode.state.column("t").to_pylist() == [15.0, DATA_END]
    assert episode.meta.score_check["match_end"].passed is True


def test_wrong_official_score_fails_the_score_check(tmp_path):
    path = write_match(
        tmp_path / "ep", Outcome(points_red=45, points_blue=31, autonomous_winner="red")
    )
    checks = run_episode(path, SCENE, allow_incomplete_scene=True).checks
    assert checks["match_end"].passed is False
    assert checks["match_end"].difference == {"red": -5.0, "blue": 0.0}
