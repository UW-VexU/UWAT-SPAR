"""The Override simulator plugin: action inference, effects, and the replay check on the synthetic
match in conftest.py."""

from pathlib import Path

import jax
import numpy as np
import pytest

from spar_schema import RobotState, StartState, action_set, load_scene
from spar_sim.bench import benchmark
from spar_sim.env import PlanEnv
from spar_sim.replay import replay_episode
from spar_state.run import run_episode

SCENE = load_scene(Path(__file__).parents[1])
SIM = SCENE.plugin("sim")
infer_action = SCENE.plugin("actions")


def start(t=0.0, objects=(), winner=None):
    robots = [
        RobotState(agent_id=name, x=x, y=y, heading=0.0, vx=0.0, vy=0.0, omega=0.0,
                   alliance=name.split("_")[0])
        for name, x, y in [("red_1", -1.5, 1.0), ("red_2", -1.5, -1.0), ("blue_1", 1.5, 1.0),
                           ("blue_2", 1.5, -1.0)]
    ]  # fmt: skip
    return StartState(
        episode_id="t",
        t=t,
        time_left_s=120.0 - t,
        robots=robots,
        objects=list(objects),
        program="v5rc",
        autonomous_winner=winner,
    )


def test_action_set():
    actions = action_set(SCENE, "v5rc")
    assert actions.names[0] == "idle" and "flip_toggle" in actions.names
    place = actions["place"]
    assert [arg.name for arg in place.args] == ["goal", "item", "flip"]
    assert place.args[0].positions[place.args[0].labels.index("M")] == (0.0, 0.0)
    assert actions["park_midfield"].to_xy == (0.0, 0.0)
    assert actions["defend"].args[0].opponents


@pytest.mark.parametrize(
    "event, obj, xy, expected",
    [
        ({"type": "place", "goal_id": "R1"}, {"cls": "pin", "attrs": {"halves": ["yellow", "red"]}},
         None, ("place", {"goal": "R1", "item": "pin_red_yellow", "flip": "yes"})),
        ({"type": "place", "goal_id": "B2"}, {"cls": "cup", "attrs": {"gray": "down"}},
         None, ("place", {"goal": "B2", "item": "cup", "flip": "yes"})),
        ({"type": "pick"}, {"cls": "pin", "attrs": {"halves": ["blue", "blue"]}},
         (0.0, 0.0), None),  # no such Pin
        ({"type": "pick"}, {"cls": "pin", "attrs": {"halves": ["yellow", "yellow"]}},
         (-1.6, 1.4), ("load", {"loader": "L_red_far", "item": "pin_yellow_yellow"})),
        ({"type": "pick"}, {"cls": "cup", "attrs": {}}, (0.0, 0.0), ("pick", {"item": "cup"})),
        ({"type": "remove", "goal_id": "M"}, None, None, ("remove", {"goal": "M"})),
        ({"type": "toggle_flip", "target_id": "T_blue_2"}, None, None,
         ("flip_toggle", {"toggle": "T_blue_2"})),
        ({"type": "enter_midfield"}, None, (0.1, 0.0), ("park_midfield", {})),
        ({"type": "leave_midfield"}, None, (0.0, -0.7), ("move_to", {"region": "red_2"})),
        ({"type": "enter_red_1"}, None, (-1.0, 0.0), None),
    ],
    ids=["pin", "cup", "unknown pin", "load", "pick", "remove", "toggle", "enter", "leave",
         "quadrant"],
)  # fmt: skip
def test_infer_action(event, obj, xy, expected):
    assert infer_action(event, obj, xy, SCENE) == expected


@pytest.fixture
def rules():
    return SIM(SCENE, "v5rc")


def ctx(agent, *args, t=10.0, carry=2):
    return {"agent": agent, "alliance": agent // 2, "args": np.array(args), "t": t, "carry": carry}


def codes(name, **args):
    return action_set(SCENE, "v5rc").encode(name, args)[1]


def points(rules, task, checkpoint=1, xy=None):
    xy = np.zeros((4, 2)) + 1.5 if xy is None else xy  # nobody in the Midfield by default
    robots = {"xy": xy, "alliance": np.array([0, 0, 1, 1])}
    scored = SCENE.score(rules.task_state(task, robots, checkpoint, np), np)
    return float(scored["red"]), float(scored["blue"])


def test_encode_an_empty_field(rules):
    task = rules.encode(start(), ["red_1", "red_2", "blue_1", "blue_2"])
    items = SCENE.spec.sim.items
    assert task["supply"].tolist() == [4, 20, 20, 19, 56]
    assert items == [
        "pin_red_blue",
        "pin_red_yellow",
        "pin_blue_yellow",
        "pin_yellow_yellow",
        "cup",
    ]
    assert task["height"].tolist() == [0] * 9 and task["held"].sum() == 0
    assert not task["winner_known"]


def test_encode_objects_on_goals_and_held(rules):
    objects = [
        {"object_id": "p", "cls": "pin", "status": "placed", "goal_id": "R2", "stack_index": 0,
         "attrs": {"halves": ["red", "yellow"]}},
        {"object_id": "c", "cls": "cup", "status": "placed", "goal_id": "R2", "stack_index": 1,
         "attrs": {"gray": "up"}},
        {"object_id": "q", "cls": "pin", "status": "held", "holder": "blue_2",
         "attrs": {"halves": ["blue", "yellow"]}},
    ]  # fmt: skip
    task = rules.encode(start(t=30.0, objects=objects, winner="blue"), ["a", "b", "c", "blue_2"])
    assert task["height"][1] == 2 and task["held"][3].tolist() == [0, 0, 1, 0, 0]
    assert task["supply"].tolist() == [4, 19, 19, 19, 55]
    assert task["winner_known"] and points(rules, task) == (0.0, 12.0)  # the red half is hidden


def test_pick_place_and_score(rules):
    task = rules.encode(start(), ["red_1", "red_2", "blue_1", "blue_2"])
    task, ok = rules.pick(task, ctx(0, *codes("pick", item="pin_red_yellow")), np)
    assert ok and task["held"][0, 1] == 1 and task["supply"][1] == 19
    flipped = codes("place", goal="R1", item="pin_red_yellow", flip="yes")
    task, ok = rules.place(task, ctx(0, *flipped), np)
    assert ok and task["height"][0] == 1 and task["held"].sum() == 0
    assert task["pin_color"][0, 0].tolist() == [0, 1]  # yellow down, red up
    assert points(rules, task) == (5.0, 0.0)
    again, ok = rules.place(task, ctx(0, *flipped), np)
    assert not ok  # nothing left in hand


def test_carry_limit(rules):
    task = rules.encode(start(), ["red_1", "red_2", "blue_1", "blue_2"])
    for expected in (True, True, False):
        task, ok = rules.pick(task, ctx(2, *codes("pick", item="cup"), carry=2), np)
        assert bool(ok) == expected
    assert task["held"][2, 4] == 2


def test_remove_returns_the_top_object(rules):
    task = rules.encode(start(), ["red_1", "red_2", "blue_1", "blue_2"])
    task, _ = rules.pick(task, ctx(1, *codes("pick", item="cup")), np)
    task, _ = rules.place(task, ctx(1, *codes("place", goal="B3", item="cup", flip="no")), np)
    assert task["kind"][6, 0] == 2 and task["supply"][4] == 55
    task, ok = rules.remove(task, ctx(3, *codes("remove", goal="B3")), np)
    assert ok and task["kind"][6, 0] == 0 and task["height"][6] == 0 and task["supply"][4] == 56
    _, ok = rules.remove(task, ctx(3, *codes("remove", goal="B3")), np)
    assert not ok


def test_no_placing_on_the_midfield_goal_in_the_endgame(rules):
    task = rules.encode(start(), ["red_1", "red_2", "blue_1", "blue_2"])
    task, _ = rules.pick(task, ctx(0, *codes("pick", item="cup")), np)
    on_m = codes("place", goal="M", item="cup", flip="no")
    _, ok = rules.place(task, ctx(0, *on_m, t=111.0), np)  # play ends at 120 s
    assert not ok
    _, ok = rules.place(task, ctx(0, *on_m, t=109.0), np)
    assert ok


def test_toggles_and_the_autonomous_winner(rules):
    task = rules.encode(start(), ["red_1", "red_2", "blue_1", "blue_2"])
    task, ok = rules.flip_toggle(task, ctx(3, *codes("flip_toggle", toggle="T_red_2")), np)
    assert ok and task["toggle_face"].tolist() == [0, 2, 0, 0] and task["toggle_set"][1]
    scores = np.array([[0.0, 0.0], [np.nan, np.nan]])
    task = rules.on_checkpoint(task, 0, scores, np)
    assert task["winner_known"] and task["autonomous_winner"] == 3  # 0-0 is a tie
    assert points(rules, task) == (6.0, 6.0)
    in_midfield = np.array([[0.0, 0.1], [1.5, 1.5], [1.5, 1.5], [1.5, 1.5]])
    assert points(rules, task, xy=in_midfield) == (14.0, 6.0)
    assert points(rules, task, checkpoint=0, xy=in_midfield) == (0.0, 0.0)  # V5RC autonomous


def test_replay_reproduces_the_synthetic_match(override_match):
    result = run_episode(override_match.path, SCENE, allow_incomplete_scene=True)
    computed = {name: check.computed for name, check in result.checks.items()}
    assert computed == {
        "autonomous_end": {"red": 10.0, "blue": 5.0},
        "match_end": {"red": 45.0, "blue": 13.0},
    }
    assert result.actions.unattributed == 0
    replayed = replay_episode(override_match.path, SCENE)
    assert replayed.points == computed
    assert replayed.error_vs_computed == {"red": 0.0, "blue": 0.0}


def test_override_runs_under_jit_and_vmap():
    result = benchmark(PlanEnv(SCENE, program="vexu"), n_envs=8, n_steps=40, seed=1)
    assert result["env_steps_per_s"] > 0
    env = PlanEnv(SCENE, program="v5rc")
    state = env.reset_from(start(), jax.random.key(0))
    assert env.observe(state).shape[0] == 4
