"""Every scene in scenes/ that has a scene.yaml loads and validates."""

from pathlib import Path

import pytest

from spar_schema import load_scene

SCENES = Path(__file__).parents[1] / "scenes"


@pytest.mark.parametrize(
    "folder", sorted(path.parent for path in SCENES.glob("*/scene.yaml")), ids=lambda p: p.name
)
def test_scene_loads(folder):
    scene = load_scene(folder)
    assert scene.name == folder.name
    assert scene.checkpoints
    assert callable(scene.score_fn)
