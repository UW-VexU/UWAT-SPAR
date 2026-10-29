"""The repo's cutoff policy names a real scene and the proposal's cutoff day."""

from datetime import date
from pathlib import Path

from spar_schema import CutoffPolicy, load_scene

ROOT = Path(__file__).parents[1]


def test_policy_points_at_a_scene():
    policy = CutoffPolicy.load(ROOT / "configs" / "eval" / "prereg.yaml")
    assert policy.cutoff == date(2026, 11, 30)
    assert load_scene(ROOT / "scenes" / policy.scene).name == policy.scene
