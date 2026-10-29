"""Timing fitted from toy episodes, the replay check, and the spar sim commands."""

import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
from typer.testing import CliRunner

from spar_schema import (
    ACTIONS_SCHEMA,
    EVENTS_SCHEMA,
    OBJECTS_SCHEMA,
    TRACKS_SCHEMA,
    EpisodeKind,
    EpisodeMeta,
    EpisodeReader,
    EpisodeWriter,
    Outcome,
    RosterEntry,
    ScoreCheck,
    load_scene,
)
from spar_sim.cli import app
from spar_sim.fit import Timing, fit_timing
from spar_sim.replay import replay_episode, replay_summary
from spar_sim.viz import replay_svg

TOY_SCENE = Path(__file__).parents[2] / "spar-schema" / "tests" / "fixtures" / "toy_scene"
SCENE = load_scene(TOY_SCENE, strict=True)
SPEED = 0.5  # red_1 drives at this speed between its stops


def table(schema, rows):
    return pa.table({name: [row.get(name) for row in rows] for name in schema.names})


def red_xy(t):
    """red_1 waits at x = 0 until 1 s, drives to x = -0.6 by 2.2 s, and waits there."""
    return float(np.interp(t, [0.0, 1.0, 2.2, 61.0], [0.0, 0.0, -0.6, -0.6])), 0.0


def write_episode(root, episode_id, *, place_took=3.2, removed_at=None):
    """red_1 picks a ball (1 s), drives to the red goal and drops it in, then does it again;
    blue_1 drops one ball in the blue goal. Official result 2-1, unless a removal says otherwise."""
    tracks = []
    for step in range(611):
        t = step / 10
        x, y = red_xy(t)
        speed = SPEED if 1.0 < t < 2.2 else 0.0
        tracks += [
            {"t": t, "agent_id": "red_1", "x": x, "y": y, "vx": -speed, "vy": 0.0},
            {"t": t, "agent_id": "blue_1", "x": 0.7, "y": 0.0, "vx": 0.0, "vy": 0.0},
        ]
    for row in tracks:
        row.update(heading=0.0, provenance="detected")
    s = 1.0 + place_took  # when red_1's first place ends
    actions = [
        ("red_1", 0.0, 1.0, "pick", {"item": "ball", "_event": "pick", "_object": "a"}),
        ("red_1", 1.0, s, "place", {"goal": "red", "_event": "place", "_object": "a"}),
        ("red_1", s, s + 1, "pick", {"item": "ball", "_event": "pick", "_object": "b"}),
        ("red_1", s + 1, s + 2, "place", {"goal": "red", "_event": "place", "_object": "b"}),
        ("blue_1", 0.0, 0.5, "pick", {"item": "ball", "_implied": True}),
        ("blue_1", 0.5, 1.0, "place", {"goal": "blue", "_event": "place", "_object": "c"}),
    ]  # fmt: skip
    events = []
    if removed_at is not None:
        events.append({"t": removed_at, "type": "remove", "object_id": "a", "source": "rule"})
    meta = EpisodeMeta(
        episode_id=episode_id,
        kind=EpisodeKind.MATCH,
        scene="toy",
        rules_version="v1",
        program="solo",
        roster={"red_1": RosterEntry(alliance="red"), "blue_1": RosterEntry(alliance="blue")},
        score_check={"match_end": ScoreCheck(computed={"red": 2, "blue": 1}, passed=True)},
    )
    path = root / episode_id
    with EpisodeWriter(path, SCENE, meta) as w:
        w.write_tracks(table(TRACKS_SCHEMA, tracks))
        w.write_objects(table(OBJECTS_SCHEMA, []))
        w.write_events(table(EVENTS_SCHEMA, events))
        rows = [
            {"t_start": a, "t_end": b, "agent_id": agent, "action": name,
             "args": json.dumps(args), "inferred": True}
            for agent, a, b, name, args in actions
        ]  # fmt: skip
        w.write_actions(table(ACTIONS_SCHEMA, rows))
        w.write_outcome(Outcome(points_red=2, points_blue=1, autonomous_winner="red"))
    return path


@pytest.fixture
def episodes(tmp_path):
    root = tmp_path / "data" / "episodes"
    paths = [write_episode(root, f"toy_{i}", place_took=3.0 + 0.1 * i) for i in range(6)]
    paths.append(write_episode(root, "toy_removed", removed_at=4.5))  # its first place fails
    return paths


def test_fit_speed_handling_and_failures(episodes):
    timing = fit_timing([EpisodeReader(p) for p in episodes], SCENE, min_samples=5)
    assert timing.episodes == 7
    assert timing.speed_mps == pytest.approx(SPEED)
    assert timing.detour == pytest.approx(1.0)  # red_1 drives in straight lines here
    pick = timing.actions["pick"]
    assert pick.n == 14 and pick.median_s == pytest.approx(1.0) and pick.sigma == pytest.approx(0)
    # red_1's first place: 0.6 m of driving at 0.5 m/s, so its handling is the rest.
    place = timing.actions["place"]
    assert place.n == 21
    first = [took - 1.2 for took in [3.0 + 0.1 * i for i in range(6)] + [3.2]]
    handling = first + [1.0] * 7 + [0.5] * 7  # red's second place, and blue's
    assert place.median_s == pytest.approx(np.exp(np.mean(np.log(handling))), rel=1e-5)
    assert place.p_fail == pytest.approx(1 / 21)  # nobody defends, so both rates are the same
    assert place.p_fail_defended == pytest.approx(1 / 21)
    assert timing.actions["idle"].median_s == 0.5 and timing.actions["park"].n == 0


def test_fit_a_curved_drive(tmp_path):
    """red_1 drives an L: 0.5 m along x, then 1.2 m along y, at 0.5 m/s with no velocities given,
    so its path is 1.7 m where the straight line is 1.3 m."""
    corners = ([0.0, 1.0, 2.0, 4.4, 61.0], [0.0, 0.0, 0.5, 0.5, 0.5], [0.0, 0.0, 0.0, 1.2, 1.2])
    tracks = []
    for step in range(611):
        t = step / 10
        x, y = np.interp(t, corners[0], corners[1]), np.interp(t, corners[0], corners[2])
        tracks += [
            {"t": t, "agent_id": "red_1", "x": float(x), "y": float(y)},
            {"t": t, "agent_id": "blue_1", "x": 0.7, "y": -0.8},
        ]
    for row in tracks:
        row.update(heading=0.0, provenance="detected")
    actions = [
        ("red_1", 0.0, 1.0, "pick", {"item": "ball"}),
        ("red_1", 1.0, 5.4, "place", {"goal": "red"}),  # 3.4 s of driving, then 1 s
    ]
    meta = EpisodeMeta(
        episode_id="toy_curve",
        kind=EpisodeKind.MATCH,
        scene="toy",
        rules_version="v1",
        program="solo",
        roster={"red_1": RosterEntry(alliance="red"), "blue_1": RosterEntry(alliance="blue")},
    )
    with EpisodeWriter(tmp_path / "curve", SCENE, meta) as w:
        w.write_tracks(table(TRACKS_SCHEMA, tracks))
        rows = [
            {"t_start": a, "t_end": b, "agent_id": agent, "action": name,
             "args": json.dumps(args), "inferred": True}
            for agent, a, b, name, args in actions
        ]  # fmt: skip
        w.write_actions(table(ACTIONS_SCHEMA, rows))
    timing = fit_timing([EpisodeReader(tmp_path / "curve")], SCENE, min_samples=1)
    assert timing.speed_mps == pytest.approx(0.5)
    assert timing.detour == pytest.approx(1.7 / 1.3)
    assert timing.actions["place"].median_s == pytest.approx(1.0)
    assert timing.actions["pick"].median_s == pytest.approx(1.0)


def test_timing_round_trip(episodes, tmp_path):
    timing = fit_timing([EpisodeReader(p) for p in episodes], SCENE)
    timing.save(tmp_path / "timing.json")
    assert Timing.load(tmp_path / "timing.json") == timing


def test_logged_replay_reproduces_the_score(episodes):
    result = replay_episode(episodes[0], SCENE, trace=True)
    assert result.points == {
        "autonomous_end": {"red": 2.0, "blue": 1.0},
        "match_end": {"red": 2.0, "blue": 1.0},
    }
    assert result.error == {"red": 0.0, "blue": 0.0}
    assert result.error_vs_computed == {"red": 0.0, "blue": 0.0}
    assert result.actions == 6
    svg = replay_svg(result.env, result.states, title="toy_0")
    assert svg.startswith("<svg") and "red_1" in svg and "match_end 2-1" in svg


def test_sampled_replay_uses_the_timing(episodes):
    timing = fit_timing([EpisodeReader(p) for p in episodes], SCENE)
    timing.actions["place"].p_fail = timing.actions["place"].p_fail_defended = 0.0
    results = [
        replay_episode(episodes[0], SCENE, timing, logged_times=False, seed=s, trace=True)
        for s in range(3)
    ]
    for result in results:
        assert result.final == {"red": 2.0, "blue": 1.0}  # every action ends well before the end
    # When red scored its second point: from the timing, so not the logged 6 s, and seed by seed.
    second = {
        round(next(float(s.t) for s in r.states if int(s.task["points"][0]) == 2), 3)
        for r in results
    }
    assert len(second) == 3 and 6.0 not in second
    assert replay_summary(results) == {
        "episodes": 3,
        "mae_vs_official": 0.0,
        "exact_vs_official": 1.0,
        "mae_vs_computed": 0.0,
        "exact_vs_computed": 1.0,
    }


def test_cli_fit_replay_and_bench(episodes, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    shutil.copytree(TOY_SCENE, tmp_path / "scenes" / "toy")
    monkeypatch.setenv("SPAR_DATA", str(tmp_path / "data"))
    runner = CliRunner()
    fitted = runner.invoke(app, ["fit", "--scene", "toy"])
    assert fitted.exit_code == 0, fitted.output
    assert (tmp_path / "data" / "sim" / "toy-v1.json").is_file()
    replayed = runner.invoke(
        app, ["replay", str(tmp_path / "data" / "episodes"), "--scene", "toy", "--svg", "r.svg"]
    )
    assert replayed.exit_code == 0, replayed.output
    assert "mae_vs_official 0" in replayed.output and (tmp_path / "r.svg").is_file()
    sampled = runner.invoke(app, ["replay", str(episodes[0]), "--scene", "toy", "--sampled"])
    assert sampled.exit_code == 0, sampled.output
    timed = runner.invoke(
        app, ["bench", "--scene", "toy", "--program", "solo", "--n-envs", "4", "--steps", "5"]
    )
    assert timed.exit_code == 0, timed.output
    assert "steps/s" in timed.output
