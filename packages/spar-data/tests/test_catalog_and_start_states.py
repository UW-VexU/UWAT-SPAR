"""The episode catalog and start-state sampling, on toy-scene episodes in a temporary data root."""

import json
import shutil
from datetime import date
from pathlib import Path

import pyarrow as pa
import pytest
from typer.testing import CliRunner

from spar_data.catalog import scan
from spar_data.cli import app
from spar_data.start_states import sample_start_states
from spar_schema import (
    OBJECTS_SCHEMA,
    TRACKS_SCHEMA,
    EpisodeKind,
    EpisodeMeta,
    EpisodeWriter,
    RosterEntry,
    ScoreCheck,
    load_scene,
    read_start_states,
    write_start_states,
)

TOY_SCENE = Path(__file__).parents[2] / "spar-schema" / "tests" / "fixtures" / "toy_scene"
SCENE = load_scene(TOY_SCENE, strict=True)
MATCH_S = 60.0  # the toy match: 10 s autonomous, 50 s driver control

PASS = {"match_end": ScoreCheck(computed={"red": 1, "blue": 0}, passed=True)}
FAIL = {"match_end": ScoreCheck(computed={"red": 1, "blue": 0}, passed=False)}


def table(schema, rows):
    return pa.table({name: [row.get(name) for row in rows] for name in schema.names})


def red_x(t):
    return -1.0 + t / 60  # red_1 drives slowly to the right


def write(root, episode_id, *, event, played, checks):
    meta = EpisodeMeta(
        episode_id=episode_id,
        kind=EpisodeKind.MATCH,
        scene="toy",
        rules_version="v1",
        program="solo",
        event_sku=event,
        match_name="Q1",
        match_date=played,
        roster={"red_1": RosterEntry(alliance="red"), "blue_1": RosterEntry(alliance="blue")},
        score_check=checks,
    )
    tracks = [
        {"t": float(t), "agent_id": agent, "x": red_x(t) if agent == "red_1" else 0.5, "y": 0.0,
         "heading": 0.0, "vx": 1 / 60 if agent == "red_1" else None, "provenance": "detected"}
        for t in range(0, 62) for agent in ("red_1", "blue_1")
    ]  # fmt: skip
    objects = [
        {"t": 0.0, "object_id": "ball_1", "cls": "ball", "status": "free", "x": 0.0, "y": 0.0,
         "attrs": json.dumps({"scuffed": False}), "provenance": "detected"},
        {"t": 30.0, "object_id": "ball_1", "cls": "ball", "status": "placed", "goal_id": "red",
         "attrs": json.dumps({"scuffed": True}), "provenance": "detected"},
    ]  # fmt: skip
    path = root / "toy" / "v1" / "solo" / event / episode_id
    with EpisodeWriter(path, SCENE, meta) as w:
        w.write_tracks(table(TRACKS_SCHEMA, tracks))
        w.write_objects(table(OBJECTS_SCHEMA, objects))
    return path


@pytest.fixture
def root(tmp_path):
    episodes = tmp_path / "data" / "episodes"
    write(episodes, "toy_A_Q1", event="A", played=date(2026, 10, 3), checks=PASS)
    write(episodes, "toy_A_Q2", event="A", played=date(2026, 10, 3), checks=FAIL)
    write(episodes, "toy_B_Q1", event="B", played=date(2026, 11, 14), checks={})
    (episodes / "not_an_episode").mkdir()
    return episodes


def test_scan_finds_every_episode(root):
    found = scan(root)
    assert [e.meta.episode_id for e in found] == ["toy_A_Q1", "toy_A_Q2", "toy_B_Q1"]
    assert [e.score_check for e in found] == ["passed", "failed", "unchecked"]
    assert found.skipped == []


def test_filters(root):
    found = scan(root)
    assert len(found.filter(event_sku="A")) == 2
    assert [e.meta.episode_id for e in found.filter(score_check="passed")] == ["toy_A_Q1"]
    assert len(found.filter(played_from=date(2026, 11, 1))) == 1
    assert len(found.filter(played_to=date(2026, 10, 3))) == 2
    assert len(found.filter(scene="other")) == 0
    assert found.counts("event_sku", "score_check") == {
        ("A", "passed"): 1,
        ("A", "failed"): 1,
        ("B", "unchecked"): 1,
    }


def test_catalog_table(root):
    rows = scan(root).table().to_pylist()
    assert rows[2]["match_date"] == date(2026, 11, 14)
    assert rows[0]["score_check"] == "passed"


def test_episode_from_a_newer_schema_is_skipped(root):
    meta = root / "toy" / "v1" / "solo" / "B" / "toy_B_Q1" / "episode.json"
    meta.write_text(
        meta.read_text().replace('"schema_version": "0.1.0"', '"schema_version": "9.0.0"')
    )
    found = scan(root)
    assert len(found) == 2 and len(found.skipped) == 1


def test_start_states_are_moments_from_the_episodes(root):
    states = sample_start_states(scan(root), SCENE, 50, seed=4)
    assert len(states) == 50
    for state in states:
        assert 0.0 <= state.t <= MATCH_S - 1.0
        assert state.time_left_s == pytest.approx(MATCH_S - state.t)
        robots = {r.agent_id: r for r in state.robots}
        assert robots["red_1"].alliance == "red" and state.program == "solo"
        assert state.autonomous_winner is None  # these episodes have no outcome.json
        whole_second = int(state.t)  # tracks are once a second, so the latest row is the floor
        assert robots["red_1"].x == pytest.approx(red_x(whole_second), abs=1e-6)
        assert robots["red_1"].vx == pytest.approx(1 / 60, abs=1e-6)
        assert robots["blue_1"].vx == 0.0  # a missing velocity starts at rest
        ball = state.objects[0]
        assert ball["status"] == ("placed" if state.t >= 30 else "free")
        assert ball["attrs"]["scuffed"] == (state.t >= 30)
    assert {s.episode_id for s in states} == {"toy_A_Q1", "toy_A_Q2", "toy_B_Q1"}


def test_sampling_is_repeatable_with_a_seed(root):
    first = sample_start_states(scan(root), SCENE, 10, seed=1)
    assert first == sample_start_states(scan(root), SCENE, 10, seed=1)
    assert first != sample_start_states(scan(root), SCENE, 10, seed=2)


def test_start_states_round_trip(root, tmp_path):
    states = sample_start_states(scan(root), SCENE, 20, seed=3)
    path = tmp_path / "start_states.parquet"
    write_start_states(path, states, SCENE)
    restored = read_start_states(path)
    assert [s.episode_id for s in restored] == [s.episode_id for s in states]
    assert restored[0].objects == states[0].objects
    assert restored[0].robots[0].x == pytest.approx(states[0].robots[0].x, abs=1e-6)


def test_cli(root, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    shutil.copytree(TOY_SCENE, tmp_path / "scenes" / "toy")
    monkeypatch.setenv("SPAR_DATA", str(root.parent))
    runner = CliRunner()
    listed = runner.invoke(app, ["catalog"])
    assert listed.exit_code == 0, listed.output
    assert "3 episodes" in listed.output and "1 passed" in listed.output
    sampled = runner.invoke(app, ["start-states", "--scene", "toy", "--n", "25"])
    assert sampled.exit_code == 0, sampled.output
    path = root.parent / "start_states" / "toy-v1.parquet"
    assert len(read_start_states(path)) == 25
    assert {s.episode_id for s in read_start_states(path)} == {"toy_A_Q1"}  # passed only
