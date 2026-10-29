"""Events on a synthetic Override match: generic rules plus the Toggle flips from events.py."""

from spar_schema import EpisodeReader
from spar_state.events.rules import events_table, extract_events


def events_of(match, *types):
    found = extract_events(EpisodeReader(match.path), match.scene)
    return [
        (e["t"], e["type"], e["agent_id"], e["goal_id"] or e["target_id"])
        for e in found
        if e["type"] in types
    ]


def test_placements_are_credited_to_the_robot_at_the_goal(override_match):
    assert events_of(override_match, "pick", "place") == [
        (4.0, "place", "red_1", "R1"),
        (7.0, "place", "blue_1", "B2"),
        (8.0, "place", "red_2", "R3"),
        (31.0, "pick", "red_1", None),
        (33.0, "place", "red_1", "R1"),
        (35.0, "place", "red_1", "R1"),
        (84.0, "place", "red_2", "R3"),
        (85.0, "place", "red_2", "R3"),
    ]


def test_toggle_flips_are_credited_to_the_robot_at_the_wall(override_match):
    assert events_of(override_match, "toggle_flip") == [
        (12.0, "toggle_flip", "red_1", "T_red_1"),
        (55.0, "toggle_flip", "blue_2", "T_blue_1"),
    ]


def test_robots_entering_the_midfield(override_match):
    # red_1's center crosses the Midfield's edge at 93.6 s and blue_1's at 106.0 s; tracks are
    # sampled every 0.5 s.
    assert events_of(override_match, "enter_midfield") == [
        (94.0, "enter_midfield", "red_1", "midfield"),
        (106.0, "enter_midfield", "blue_1", "midfield"),
    ]


def test_all_events_make_valid_rows(override_match):
    rows = events_table(extract_events(EpisodeReader(override_match.path), override_match.scene))
    assert set(rows.column("source").to_pylist()) == {"rule"}
    assert {"toggle_flip", "place", "enter_midfield"} <= set(rows.column("type").to_pylist())
