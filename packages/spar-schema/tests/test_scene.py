"""load_scene: validation, TODO values, rules versions and the scoring plugin."""

import numpy as np
import pytest

from spar_schema import (
    SceneError,
    SceneIncompleteError,
    SceneVersionError,
    SchemaVersionError,
    load_scene,
)


def test_toy_scene_loads(scene):
    assert scene.name == "toy"
    assert scene.rules_version == "v1"
    assert scene.todos == ()
    assert scene.alliances == ["red", "blue"]
    assert scene.object_classes == ["ball", "goal"]
    assert scene.checkpoints == ["autonomous_end", "match_end"]
    assert scene.checkpoint_time("autonomous_end", "solo") == 10.0
    assert scene.checkpoint_time("match_end", "solo") == 62.0  # 10 + 50, plus a 2 s delay
    assert scene.landmarks["red_corner"] == (-1.0, -1.0, 0.0)
    assert list(scene.strategy_code.fields) == ["first_target"]
    state = scene.task_state.empty(np)
    state["points"][:] = [2, 1]
    assert {k: float(v) for k, v in scene.score(state, np).items()} == {"red": 2.0, "blue": 1.0}


def test_loads_from_scene_yaml_path(toy_scene_path):
    assert load_scene(toy_scene_path / "scene.yaml").name == "toy"


def test_todo_values_are_listed(toy_scene_dir, edit):
    edit(toy_scene_dir / "scene.yaml", "driver_s: 50", "driver_s: TODO")
    edit(toy_scene_dir / "robot_limits.yaml", "max_speed_mps: 1.0", "max_speed_mps: TODO_measure")
    scene = load_scene(toy_scene_dir)
    assert scene.todos == (
        "scene.yaml: programs.solo.driver_s",
        "robot_limits.yaml: robots.bot.max_speed_mps",
    )
    with pytest.raises(SceneIncompleteError, match="driver_s"):
        load_scene(toy_scene_dir, strict=True)


def test_rules_version_must_match(toy_scene_path):
    assert load_scene(toy_scene_path, rules_version="v1").rules_version == "v1"
    with pytest.raises(SceneVersionError):
        load_scene(toy_scene_path, rules_version="v2")


def test_unknown_schema_major_is_refused(toy_scene_dir, edit):
    edit(toy_scene_dir / "scene.yaml", "schema_version: 0.1.0", "schema_version: 1.0.0")
    with pytest.raises(SchemaVersionError):
        load_scene(toy_scene_dir)


def test_typo_in_scene_is_an_error(toy_scene_dir, edit):
    edit(toy_scene_dir / "scene.yaml", "timing: {}", "timing: {}\ntimng: {}")
    with pytest.raises(SceneError, match="timng"):
        load_scene(toy_scene_dir)


def test_wrong_type_is_an_error(toy_scene_dir, edit):
    edit(toy_scene_dir / "scene.yaml", "driver_s: 50", "driver_s: fifty")
    with pytest.raises(SceneError, match="driver_s"):
        load_scene(toy_scene_dir)


def test_missing_scoring_function(toy_scene_dir, edit):
    edit(toy_scene_dir / "scene.yaml", "function: score", "function: total")
    with pytest.raises(SceneError, match="total"):
        load_scene(toy_scene_dir)


def test_missing_referenced_file(toy_scene_dir):
    (toy_scene_dir / "landmarks.yaml").unlink()
    with pytest.raises(SceneError, match="landmarks.yaml"):
        load_scene(toy_scene_dir)


def test_scene_without_a_state_plugin_cannot_build_states(toy_scene_dir, edit):
    edit(toy_scene_dir / "scene.yaml", "  state: state.py:build_state\n", "")
    scene = load_scene(toy_scene_dir)
    assert not scene.has_plugin("state")
    with pytest.raises(SceneError, match="no 'state' plugin"):
        scene.build_state(None)


def test_plugin_entries_must_name_a_file_and_function(toy_scene_dir, edit):
    edit(toy_scene_dir / "scene.yaml", "state: state.py:build_state", "state: state.py")
    with pytest.raises(SceneError, match="plugins"):
        load_scene(toy_scene_dir)


def test_missing_state_plugin_file(toy_scene_dir):
    (toy_scene_dir / "state.py").unlink()
    with pytest.raises(SceneError, match="state.py"):
        load_scene(toy_scene_dir)


def test_checkpoint_times_must_cover_every_checkpoint(toy_scene_dir, edit):
    edit(toy_scene_dir / "scene.yaml", "    autonomous_end: {after: autonomous}\n", "")
    with pytest.raises(SceneError, match="checkpoint_at"):
        load_scene(toy_scene_dir)
