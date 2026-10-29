"""Task state and the score check, on the toy test scene: one point per ball on a goal."""

import json
from pathlib import Path

import pyarrow as pa
import pytest
from typer.testing import CliRunner

from spar_schema import (
    EpisodeKind,
    EpisodeMeta,
    EpisodeReader,
    EpisodeWriter,
    Outcome,
    RosterEntry,
    load_scene,
)
from spar_state.cli import app
from spar_state.run import run_episode
from spar_state.score_check import check_scores
from spar_state.state import latest_rows, snapshot_at

TOY_SCENE = Path(__file__).parents[2] / "spar-schema" / "tests" / "fixtures" / "toy_scene"
SCENE = load_scene(TOY_SCENE, strict=True)

# (t, ball, goal it is placed on). The toy match: 10 s autonomous, then 50 s of driver control.
PLACEMENTS = [(5.0, "ball_1", "red"), (20.0, "ball_2", "blue"), (40.0, "ball_3", "red")]
DATA_END = 61.0  # the video stops 1 s before match_end's checkpoint at 62 s


def objects_table(placements):
    rows = [(0.0, f"ball_{i}", "free", None) for i in range(1, 4)]
    rows += [(t, ball, "placed", goal) for t, ball, goal in placements]
    n = len(rows)
    return pa.table(
        {
            "t": [r[0] for r in rows],
            "frame": [int(r[0] * 30) for r in rows],
            "object_id": [r[1] for r in rows],
            "cls": ["ball"] * n,
            "x": [None] * n,
            "y": [None] * n,
            "z": [None] * n,
            "status": [r[2] for r in rows],
            "holder": [None] * n,
            "goal_id": [r[3] for r in rows],
            "stack_index": [None] * n,
            "attrs": [json.dumps({"scuffed": r[1] == "ball_2"}) for r in rows],
            "conf": [None] * n,
            "provenance": ["detected"] * n,
        }
    )


def tracks_table():
    times = [float(t) for t in range(int(DATA_END) + 1)]
    rows = [(t, agent) for t in times for agent in ("red_1", "blue_1")]
    n = len(rows)
    return pa.table(
        {
            "t": [r[0] for r in rows],
            "frame": [int(r[0] * 30) for r in rows],
            "agent_id": [r[1] for r in rows],
            "x": [r[0] / 100 for r in rows],
            "y": [0.0] * n,
            "heading": [0.0] * n,
            "vx": [0.01] * n,
            "vy": [0.0] * n,
            "omega": [0.0] * n,
            "cov": [None] * n,
            "conf": [None] * n,
            "provenance": ["detected"] * n,
        }
    )


def write_episode(path, outcome):
    meta = EpisodeMeta(
        episode_id="toy_E1_Q1",
        kind=EpisodeKind.MATCH,
        scene="toy",
        rules_version="v1",
        program="solo",
        roster={"red_1": RosterEntry(alliance="red"), "blue_1": RosterEntry(alliance="blue")},
    )
    with EpisodeWriter(path, SCENE, meta) as w:
        w.write_tracks(tracks_table())
        w.write_objects(objects_table(PLACEMENTS))
        if outcome:
            w.write_outcome(outcome)
    return path


def outcome(red=2, blue=1, winner="red"):
    return Outcome(points_red=red, points_blue=blue, autonomous_winner=winner)


def test_latest_rows_takes_the_last_row_at_or_before_t():
    rows = latest_rows(objects_table(PLACEMENTS), "object_id", 20.0)
    status = {r["object_id"]: (r["status"], r["goal_id"]) for r in rows}
    assert status == {
        "ball_1": ("placed", "red"),
        "ball_2": ("placed", "blue"),
        "ball_3": ("free", None),
    }
    assert latest_rows(objects_table(PLACEMENTS), "object_id", -1.0) == []


def test_snapshot_adds_alliances_and_parses_attrs(tmp_path):
    episode = EpisodeReader(write_episode(tmp_path / "ep", outcome()))
    snap = snapshot_at(episode, 30.0, checkpoint="match_end", autonomous_winner="red")
    assert {r["agent_id"]: r["alliance"] for r in snap.robots} == {"red_1": "red", "blue_1": "blue"}
    assert all(r["t"] == 30.0 for r in snap.robots)
    assert {o["object_id"]: o["attrs"]["scuffed"] for o in snap.objects}["ball_2"] is True
    assert snap.program == "solo"


def test_matching_outcome_passes_at_every_checkpoint(tmp_path):
    result = check_scores(EpisodeReader(write_episode(tmp_path / "ep", outcome())), SCENE)
    auton, final = result.checks["autonomous_end"], result.checks["match_end"]
    assert (auton.computed, auton.computed_winner, auton.passed) == (
        {"red": 1, "blue": 0},
        "red",
        True,
    )
    assert (final.computed, final.difference, final.passed) == (
        {"red": 2, "blue": 1},
        {"red": 0, "blue": 0},
        True,
    )


def test_wrong_final_score_fails_with_its_difference(tmp_path):
    check = check_scores(EpisodeReader(write_episode(tmp_path / "ep", outcome(red=3))), SCENE)
    assert check.checks["match_end"].passed is False
    assert check.checks["match_end"].difference == {"red": -1.0, "blue": 0.0}


def test_tolerance_allows_a_small_difference(tmp_path):
    episode = EpisodeReader(write_episode(tmp_path / "ep", outcome(red=3)))
    assert check_scores(episode, SCENE, tolerance=1).checks["match_end"].passed is True


@pytest.mark.parametrize(
    "winner, passed", [("red", True), ("blue", False), ("tie", False), ("none", None)]
)
def test_autonomous_is_compared_by_winner(tmp_path, winner, passed):
    episode = EpisodeReader(write_episode(tmp_path / "ep", outcome(winner=winner)))
    assert check_scores(episode, SCENE).checks["autonomous_end"].passed is passed


def test_checkpoint_after_the_data_uses_the_last_row(tmp_path):
    states = check_scores(EpisodeReader(write_episode(tmp_path / "ep", outcome())), SCENE).states
    assert states.column("t").to_pylist() == [10.0, DATA_END]
    assert states.column("checkpoint").to_pylist() == ["autonomous_end", "match_end"]


def test_run_episode_writes_state_and_score_check(tmp_path):
    path = write_episode(tmp_path / "ep", outcome())
    result = run_episode(path, SCENE)
    episode = EpisodeReader(path)
    assert episode.state.num_rows == 2
    assert episode.events.num_rows == result.events > 0  # balls placed on goals
    assert result.codes is None and episode.strategy is None  # the toy scene has no strategy plugin
    assert episode.meta.score_check["match_end"].passed is True
    stored = SCENE.task_state.from_json(
        episode.state.column("state")[1].as_py(), __import__("numpy")
    )
    assert stored["points"].tolist() == [2, 1]


def test_episode_without_outcome_cannot_be_checked(tmp_path):
    with pytest.raises(ValueError, match="outcome"):
        check_scores(EpisodeReader(write_episode(tmp_path / "ep", None)), SCENE)


def test_cli_run(tmp_path):
    path = write_episode(tmp_path / "ep", outcome(red=3))
    result = CliRunner().invoke(app, ["run", str(path), "--scene", str(TOY_SCENE)])
    assert result.exit_code == 0, result.output
    assert "autonomous_end: pass" in result.output
    assert "match_end: FAIL" in result.output
    assert "events:" in result.output and "no strategy plugin" in result.output
