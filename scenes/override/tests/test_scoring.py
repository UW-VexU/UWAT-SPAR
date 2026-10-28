"""Override scoring, with tests for each rule of game manual v2.1 that affects points.

Stacks are written bottom first, so stack(state, "R2", pin("yellow", "red"), cup("up")) puts a
Pin with its red half on top into Goal R2 and a Cup over it, gray (opaque) half up.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from spar_schema import load_scene

SCENE = load_scene(Path(__file__).parents[1])
SPEC = SCENE.task_state


def _load_scoring_module():
    path = SCENE.root / SCENE.spec.objective.module
    module_spec = importlib.util.spec_from_file_location("override_scoring_under_test", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


SCORING = _load_scoring_module()


def new_state(checkpoint="match_end", program="v5rc", winner="none"):
    state = SPEC.empty(np)
    state["checkpoint"] = np.asarray(SPEC.code("checkpoint", checkpoint), dtype=np.int8)
    state["program"] = np.asarray(SPEC.code("program", program), dtype=np.int8)
    state["autonomous_winner"] = np.asarray(SPEC.code("autonomous_winner", winner), dtype=np.int8)
    return state


def pin(lower, upper):
    return ("pin", lower, upper)


def cup(gray):
    """A Cup with its gray (opaque) half "up" or "down"."""
    return ("cup", gray)


def stack(state, goal, *items, crowded=()):
    """Put a stack on a Goal, bottom first. `crowded` lists levels whose socket is shared."""
    g = SPEC.index("goal", goal)
    for level, item in enumerate(items):
        state["kind"][g, level] = SPEC.code("kind", item[0])
        if item[0] == "pin":
            state["pin_color"][g, level] = [SPEC.code("pin_color", c) for c in item[1:]]
        else:
            state["cup_opaque_lower"][g, level] = item[1] == "down"
    for level in crowded:
        state["crowded"][g, level] = True


def toggle(state, quadrant, face, is_set=True):
    q = SPEC.index("quadrant", quadrant)
    state["toggle_face"][q] = SPEC.code("toggle_face", face)
    state["toggle_set"][q] = is_set


def midfield(state, red, blue):
    state["robots_in_midfield"][:] = [red, blue]


def points(state):
    SPEC.check(state)
    result = SCENE.score(state, np)
    return float(result["red"]), float(result["blue"])


# The plugin's constants against scene.yaml.


def test_codes_match_scene_yaml():
    codes = {
        "kind": {"empty": SCORING.EMPTY, "pin": SCORING.PIN, "cup": SCORING.CUP},
        "pin_color": {"yellow": SCORING.YELLOW, "red": SCORING.RED, "blue": SCORING.BLUE},
        "toggle_face": {"yellow": SCORING.YELLOW, "red": SCORING.RED, "blue": SCORING.BLUE},
        "checkpoint": {"autonomous_end": SCORING.AUTONOMOUS_END, "match_end": SCORING.MATCH_END},
        "program": {"v5rc": SCORING.V5RC, "vexu": SCORING.VEXU},
        "autonomous_winner": {
            "none": SCORING.WINNER_NONE,
            "red": SCORING.WINNER_RED,
            "blue": SCORING.WINNER_BLUE,
            "tie": SCORING.WINNER_TIE,
        },
    }
    for field, labels in codes.items():
        assert SPEC.fields[field].values == list(labels), field
        for label, code in labels.items():
            assert SPEC.code(field, label) == code, (field, label)
    assert SPEC.fields["checkpoint"].values == SCENE.checkpoints
    assert SPEC.fields["program"].values == list(SCENE.programs)


def test_points_match_scene_yaml():
    assert SCENE.spec.objective.points == {
        "autonomous_bonus": SCORING.AUTONOMOUS_BONUS,
        "autonomous_tie_bonus": SCORING.AUTONOMOUS_TIE_BONUS,
        "alliance_pin_half": SCORING.ALLIANCE_PIN_HALF,
        "yellow_pin_half": SCORING.YELLOW_PIN_HALF,
        "robot_in_midfield": SCORING.ROBOT_IN_MIDFIELD,
    }


def test_goal_owners_match_scene_yaml():
    goals = SCENE.spec.objects["goal"]["instances"]
    quadrants = SPEC.dims["quadrant"]
    assert SPEC.dims["goal"] == list(goals)
    expected = [
        len(quadrants) if goal["quadrant"] is None else quadrants.index(goal["quadrant"])
        for goal in goals.values()
    ]
    assert list(SCORING.GOAL_OWNER) == expected
    assert list(SCENE.spec.field.regions["quadrants"]) == quadrants
    toggles = SCENE.spec.objects["toggle"]["instances"].values()
    assert sorted(t["quadrant"] for t in toggles) == sorted(quadrants)


def test_empty_field_scores_nothing():
    assert points(new_state()) == (0, 0)


# SC2: Placed.


def test_pin_in_goal_is_placed():
    state = new_state()
    stack(state, "R2", pin("yellow", "red"))
    assert points(state) == (5, 0)


def test_stack_of_pins_and_cups_is_placed():
    state = new_state()
    stack(
        state,
        "R2",
        pin("yellow", "red"),
        cup("up"),
        pin("yellow", "red"),
        cup("up"),
        pin("yellow", "red"),
    )
    assert points(state) == (15, 0)


def test_pin_on_cup_without_pin_below_is_not_placed():
    state = new_state()
    stack(state, "R2", cup("up"), pin("red", "red"))
    assert points(state) == (0, 0)


def test_pin_resting_on_pin_is_not_placed():
    state = new_state()
    stack(state, "R2", pin("blue", "red"), pin("red", "red"))
    assert points(state) == (5, 0)  # only the bottom Pin's upper half


def test_crowded_goal_places_nothing():
    state = new_state()
    stack(state, "R2", pin("blue", "red"), crowded=[0])
    assert points(state) == (0, 0)


def test_crowded_cup_top_unplaces_the_pin_in_it():
    state = new_state()
    stack(state, "R2", pin("yellow", "red"), cup("up"), pin("yellow", "red"), crowded=[2])
    assert points(state) == (5, 0)  # the bottom Pin still counts


def test_crowded_cup_bottom_unplaces_the_pin_under_it_and_everything_above():
    state = new_state()
    stack(state, "R2", pin("yellow", "red"), cup("up"), pin("yellow", "red"), crowded=[1])
    assert points(state) == (0, 0)


# SC3: Scored halves.


def test_half_inside_goal_is_not_scored():
    state = new_state()
    stack(state, "R2", pin("red", "blue"))
    assert points(state) == (0, 5)


def test_each_alliance_color_scores_for_its_alliance():
    state = new_state()
    stack(state, "R2", pin("yellow", "red"))
    stack(state, "B2", pin("yellow", "blue"))
    stack(state, "R4", pin("red", "blue"))  # alliance colors score on any Goal
    assert points(state) == (5, 10)


def test_gray_half_up_hides_the_pin_above_and_shows_the_pin_below():
    state = new_state()
    stack(state, "R4", pin("yellow", "red"), cup("up"), pin("blue", "red"))
    assert points(state) == (10, 0)


def test_gray_half_down_hides_the_pin_below_and_shows_the_pin_above():
    state = new_state()
    stack(state, "R4", pin("yellow", "red"), cup("down"), pin("blue", "red"))
    assert points(state) == (5, 5)


def test_cup_on_top_covers_the_top_half_only_when_gray_side_down():
    covered = new_state()
    stack(covered, "R4", pin("yellow", "red"), cup("down"))
    shown = new_state()
    stack(shown, "R4", pin("yellow", "red"), cup("up"))
    assert points(covered) == (0, 0)
    assert points(shown) == (5, 0)


# SC4 and SC5a: Toggles and yellow halves in Quadrants.


def test_yellow_half_with_neutral_toggle_scores_nothing():
    state = new_state()
    stack(state, "R1", pin("red", "yellow"))
    toggle(state, "red_1", "yellow")
    assert points(state) == (0, 0)


@pytest.mark.parametrize("face, expected", [("red", (10, 0)), ("blue", (0, 10))])
def test_yellow_half_belongs_to_the_toggle_color(face, expected):
    state = new_state()
    stack(state, "B4", pin("red", "yellow"))
    toggle(state, "blue_2", face)
    assert points(state) == expected


def test_toggle_that_is_not_set_counts_as_yellow():
    state = new_state()
    stack(state, "R1", pin("red", "yellow"))
    toggle(state, "red_1", "red", is_set=False)  # touched by a Robot, or not fully seated
    assert points(state) == (0, 0)


def test_only_the_goals_own_quadrant_toggle_counts():
    state = new_state()
    stack(state, "R1", pin("red", "yellow"))  # Quadrant red_1
    toggle(state, "red_2", "red")
    toggle(state, "blue_1", "blue")
    assert points(state) == (0, 0)


def test_manual_example_stack_under_each_toggle_color():
    """The three worked examples after SC5: one stack, totals by Toggle color."""
    uppers = ["red", "red", "blue", "yellow", "yellow", "red", "yellow"]
    items = []
    for i, upper in enumerate(uppers):
        if i:
            items.append(cup("up"))
        items.append(pin("yellow", upper))
    for face, expected in [("yellow", (15, 5)), ("blue", (15, 35)), ("red", (45, 5))]:
        state = new_state()
        stack(state, "R1", *items)
        toggle(state, "red_1", face)
        assert points(state) == expected, face


# SC5b and SC6: the Midfield.


def test_midfield_yellow_goes_to_the_alliance_with_more_robots_there():
    state = new_state()
    stack(state, "M", pin("red", "yellow"))
    midfield(state, red=2, blue=1)
    assert points(state) == (10 + 16, 8)


def test_midfield_tie_means_nobody_owns_its_yellow():
    state = new_state()
    stack(state, "M", pin("red", "yellow"))
    midfield(state, red=1, blue=1)
    assert points(state) == (8, 8)


def test_midfield_ignores_toggles():
    state = new_state()
    stack(state, "M", pin("red", "yellow"))
    for quadrant in SPEC.dims["quadrant"]:
        toggle(state, quadrant, "blue")
    assert points(state) == (0, 0)


def test_each_robot_ending_in_the_midfield_scores():
    state = new_state()
    midfield(state, red=1, blue=2)
    assert points(state) == (8, 16)


# SC7 and VUG5: the Autonomous Period.


def autonomous_state(program):
    state = new_state(checkpoint="autonomous_end", program=program, winner="red")
    stack(state, "R2", pin("yellow", "red"))
    stack(state, "M", pin("red", "yellow"))
    midfield(state, red=2, blue=0)
    return state


def test_v5rc_autonomous_leaves_out_the_midfield():
    assert points(autonomous_state("v5rc")) == (5, 0)


def test_vexu_autonomous_counts_the_midfield():
    assert points(autonomous_state("vexu")) == (5 + 10 + 16, 0)


@pytest.mark.parametrize(
    "winner, expected",
    [("red", (12, 0)), ("blue", (0, 12)), ("tie", (6, 6)), ("none", (0, 0))],
)
def test_autonomous_bonus_is_added_at_match_end(winner, expected):
    assert points(new_state(checkpoint="match_end", winner=winner)) == expected


def test_autonomous_bonus_is_not_part_of_the_autonomous_score():
    assert points(new_state(checkpoint="autonomous_end", winner="red")) == (0, 0)


# The same code under JAX, as the simulator will run it.


def mixed_state():
    state = new_state(checkpoint="match_end", program="vexu", winner="tie")
    stack(state, "R1", pin("yellow", "red"), cup("up"), pin("red", "yellow"), cup("down"))
    stack(state, "B3", pin("red", "blue"), cup("down"), pin("yellow", "yellow"), crowded=[2])
    stack(state, "M", pin("blue", "yellow"), cup("up"), pin("yellow", "red"))
    toggle(state, "red_1", "red")
    toggle(state, "blue_2", "blue", is_set=False)
    midfield(state, red=1, blue=2)
    return state


def test_mixed_state_by_hand():
    # Red: R1's bottom Pin shows red (5), the Midfield top Pin shows red (5), one Robot in the
    # Midfield (8), tie bonus (6). Blue: owns the Midfield's yellow with more Robots there (10),
    # two Robots (16), tie bonus (6). B3 scores nothing: its blue half is under a gray-down Cup
    # and its top Pin is crowded. R1's top yellow half is under a gray-down Cup too.
    assert points(mixed_state()) == (24, 32)


def test_jax_jit_and_vmap_match_numpy():
    jax = pytest.importorskip("jax")
    jnp = jax.numpy
    state = mixed_state()
    expected = points(state)
    as_jax = {name: jnp.asarray(value) for name, value in state.items()}
    jitted = jax.jit(lambda s: SCORING.score(s, jnp))(as_jax)
    assert (float(jitted["red"]), float(jitted["blue"])) == expected
    batch = {name: jnp.stack([value, value]) for name, value in as_jax.items()}
    batched = jax.vmap(lambda s: SCORING.score(s, jnp))(batch)
    assert batched["red"].tolist() == [expected[0]] * 2
    assert batched["blue"].tolist() == [expected[1]] * 2
