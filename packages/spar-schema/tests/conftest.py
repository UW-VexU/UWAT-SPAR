"""Shared fixtures: a small complete scene, and episode data that matches it."""

import shutil
from pathlib import Path

import pyarrow as pa
import pytest

from spar_schema import EpisodeKind, EpisodeMeta, RosterEntry, load_scene

TOY_SCENE = Path(__file__).parent / "fixtures" / "toy_scene"


def _edit(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert old in text, f"{old!r} not in {path}"
    path.write_text(text.replace(old, new), encoding="utf-8")


@pytest.fixture
def edit():
    """edit(path, old, new) replaces text in a file, failing if it is not there."""
    return _edit


@pytest.fixture
def toy_scene_path():
    """The toy scene folder. Read only: use toy_scene_dir to edit a copy."""
    return TOY_SCENE


@pytest.fixture
def toy_scene_dir(tmp_path):
    """A copy of the toy scene that a test may edit."""
    return Path(shutil.copytree(TOY_SCENE, tmp_path / "toy_scene"))


@pytest.fixture
def scene():
    return load_scene(TOY_SCENE, strict=True)


@pytest.fixture
def meta():
    return EpisodeMeta(
        episode_id="toy_E1_Q1",
        kind=EpisodeKind.MATCH,
        scene="toy",
        rules_version="v1",
        program="solo",
        event_sku="E1",
        match_name="Q1",
        roster={
            "red_1": RosterEntry(alliance="red", team="1A"),
            "blue_1": RosterEntry(alliance="blue", team="2B"),
        },
    )


@pytest.fixture
def tables():
    """One small valid table per episode table, by name, with columns in schema order."""
    return {
        "tracks": pa.table(
            {
                "t": [0.0, 0.0, 0.1],
                "frame": [0, 0, 3],
                "agent_id": ["red_1", "blue_1", "red_1"],
                "x": [-0.5, 0.5, -0.45],
                "y": [0.0, 0.0, 0.01],
                "heading": [0.0, 3.14, 0.02],
                "vx": [0.0, 0.0, 0.5],
                "vy": [0.0, 0.0, 0.1],
                "omega": [0.0, 0.0, 0.2],
                "cov": [[0.01] * 9, [0.01] * 9, None],
                "conf": [0.9, 0.8, 0.85],
                "provenance": ["detected", "detected", "interpolated"],
            }
        ),
        "objects": pa.table(
            {
                "t": [0.0, 0.1],
                "frame": [0, 3],
                "object_id": ["ball_1", "ball_1"],
                "cls": ["ball", "ball"],
                "x": [0.0, None],
                "y": [0.0, None],
                "z": [0.0, None],
                "status": ["free", "held"],
                "holder": [None, "red_1"],
                "goal_id": [None, None],
                "stack_index": [None, None],
                "attrs": ["{}", None],
                "conf": [0.7, 0.6],
                "provenance": ["detected", "detected"],
            }
        ),
        "events": pa.table(
            {
                "t": [0.1],
                "agent_id": ["red_1"],
                "type": ["pick"],
                "object_id": ["ball_1"],
                "goal_id": [None],
                "target_id": [None],
                "conf": [0.6],
                "source": ["rule"],
            }
        ),
        "state": pa.table(
            {
                "t": [0.0, 60.0],
                "checkpoint": [None, "match_end"],
                "points_red": [0.0, 1.0],
                "points_blue": [0.0, 0.0],
                "state": ['{"red": 0, "blue": 0}', '{"red": 1, "blue": 0}'],
            }
        ),
        "actions": pa.table(
            {
                "t_start": [0.0],
                "t_end": [0.1],
                "agent_id": ["red_1"],
                "action": ["pick"],
                "args": ['{"object_id": "ball_1"}'],
                "inferred": [True],
            }
        ),
    }
