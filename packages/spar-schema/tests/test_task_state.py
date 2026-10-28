"""TaskStateSpec: shapes, codes, checks and the JSON form stored in state.parquet."""

import numpy as np
import pytest
from pydantic import ValidationError

from spar_schema import SceneError, TaskStateSpec, load_scene


def test_empty_state_has_declared_shapes(scene):
    state = scene.task_state.empty(np)
    assert state["points"].shape == (2,) and state["points"].dtype == np.int16
    assert state["holder"].shape == (3,) and state["holder"].dtype == np.int8
    assert state["over"].shape == () and state["over"].dtype == np.bool_
    scene.task_state.check(state)


def test_labels_and_codes(scene):
    spec = scene.task_state
    assert spec.index("alliance", "blue") == 1
    assert spec.code("holder", "red") == 1
    with pytest.raises(ValueError):
        spec.index("slot", "first")  # slot is a plain size, with no labels
    with pytest.raises(ValueError):
        spec.code("points", "red")  # points has no values


def test_json_round_trip(scene):
    spec = scene.task_state
    state = spec.empty(np)
    state["points"][:] = [12, 7]
    state["holder"][:] = [0, 2, 1]
    state["over"] = np.bool_(True)
    restored = spec.from_json(spec.to_json(state), np)
    for name in state:
        np.testing.assert_array_equal(restored[name], state[name])
        assert restored[name].dtype == state[name].dtype


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda s: s.pop("over"), "missing"),
        (lambda s: s.update(extra=np.zeros(1)), "unexpected"),
        (lambda s: s.update(points=np.zeros(3, dtype=np.int16)), "shape"),
        (lambda s: s.update(points=np.zeros(2, dtype=np.float32)), "dtype"),
        (lambda s: s.update(holder=np.array([0, 3, 1], dtype=np.int8)), "codes"),
    ],
    ids=["missing field", "unexpected field", "wrong shape", "wrong dtype", "unknown code"],
)
def test_check_rejects_bad_states(scene, change, message):
    state = scene.task_state.empty(np)
    change(state)
    with pytest.raises(ValueError, match=message):
        scene.task_state.check(state)


def test_fields_must_use_declared_dims():
    with pytest.raises(ValidationError, match="undeclared"):
        TaskStateSpec.model_validate(
            {"dims": {"goal": 2}, "fields": {"kind": {"dims": ["level"], "dtype": "int8"}}}
        )


def test_values_need_an_integer_dtype():
    with pytest.raises(ValidationError, match="integer"):
        TaskStateSpec.model_validate(
            {"dims": {}, "fields": {"done": {"dtype": "bool", "values": ["no", "yes"]}}}
        )


def test_scene_without_state_section(toy_scene_dir):
    text = (toy_scene_dir / "scene.yaml").read_text(encoding="utf-8")
    (toy_scene_dir / "scene.yaml").write_text(text.split("\nstate:")[0], encoding="utf-8")
    with pytest.raises(SceneError, match="no state section"):
        _ = load_scene(toy_scene_dir).task_state
