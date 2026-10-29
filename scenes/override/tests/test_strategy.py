"""Override strategy codes extracted from the synthetic match in conftest.py."""

import pytest

from spar_schema import EpisodeReader
from spar_state.events.rules import extract_events
from spar_state.strategy.distance import code_distance
from spar_state.strategy.extract import check_code, extract_codes


@pytest.fixture
def codes(override_match):
    episode = EpisodeReader(override_match.path)
    return extract_codes(
        episode, extract_events(episode, override_match.scene), override_match.scene
    )


def test_codes_match_the_strategy_code_definition(codes, override_match):
    for alliance, code in codes.items():
        check_code(code, override_match.scene.strategy_code, label=alliance)


def test_autonomous_route(codes):
    # Red placed on R1 (red_1, 4 s) and R3 (red_2, 8 s), then flipped T_red_1 (red_1, 12 s).
    assert codes["red"]["auton_route"] == ["R1", "R3", "T_red_1"]
    assert codes["blue"]["auton_route"] == ["B2"]


def test_goal_order_per_robot(codes):
    assert codes["red"]["goal_order"] == [["R1"], ["R3"]]  # repeats collapsed
    assert codes["blue"]["goal_order"] == [["B2"], []]


def test_toggle_share(codes):
    # T_red_1 red from 12 s to the end of the 120 s match; T_blue_1 blue from 55 s.
    assert codes["red"]["toggle_share"] == pytest.approx([108 / 120, 0, 0, 0], abs=1e-6)
    assert codes["blue"]["toggle_share"] == pytest.approx([0, 0, 65 / 120, 0], abs=1e-6)


def test_midfield_entry(codes):
    assert codes["red"]["midfield_entry"] == [94.0, None]
    assert codes["blue"]["midfield_entry"] == [106.0, None]


def test_time_split_reflects_where_robots_spent_time(codes):
    red_1, red_2 = codes["red"]["time_split"]
    blue_1, blue_2 = codes["blue"]["time_split"]
    scoring, loading, defense, toggles = range(4)
    assert red_1[toggles] > 0 and red_1[scoring] > 0  # at the T_red_1 wall and at R1
    assert red_2[loading] > 0.1  # 10 s at the near red Loader
    assert blue_2[toggles] > 0.5  # parked at the T_blue_1 wall from 50 s
    assert blue_1[scoring] > 0.5  # at B2 for most of the match


def test_the_two_alliances_played_differently(codes, override_match):
    spec = override_match.scene.strategy_code
    assert code_distance(codes["red"], codes["red"], spec, 120.0) == 0.0
    assert code_distance(codes["red"], codes["blue"], spec, 120.0) > 1.0


def test_spar_state_run_writes_events_and_strategy(override_match, codes):
    from spar_state.run import run_episode

    result = run_episode(override_match.path, override_match.scene, allow_incomplete_scene=True)
    episode = EpisodeReader(override_match.path)
    assert episode.events.num_rows == result.events
    assert "toggle_flip" in episode.events.column("type").to_pylist()
    assert episode.strategy.codes == codes == result.codes
    assert episode.strategy.coder == "pipeline"
    assert episode.strategy.code_version.startswith("override-")
    assert set(episode.meta.score_check) == {"autonomous_end", "match_end"}
