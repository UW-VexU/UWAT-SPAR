"""PlanEnv on the toy scene: one robot per alliance, balls picked up and dropped in goals."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spar_schema import RobotState, StartState, load_scene
from spar_sim.actions import random_actions
from spar_sim.env import PlanEnv
from spar_sim.fit import ActionTiming, Timing

TOY_SCENE = Path(__file__).parents[2] / "spar-schema" / "tests" / "fixtures" / "toy_scene"
SCENE = load_scene(TOY_SCENE, strict=True)
PLAY_END, MATCH_END = 60.0, 62.0  # 10 s autonomous, 50 s driver control, scored 2 s later


def start_at(t=0.0, red=(-0.5, 0.0), blue=(0.5, 0.0), objects=()):
    robots = [
        RobotState(agent_id=name, x=xy[0], y=xy[1], heading=0.0, vx=0.0, vy=0.0, omega=0.0,
                   alliance=alliance)
        for name, alliance, xy in (("r", "red", red), ("b", "blue", blue))
    ]  # fmt: skip
    return StartState(
        episode_id="toy", t=t, time_left_s=PLAY_END - t, robots=robots, objects=list(objects)
    )


@pytest.fixture
def env():
    return PlanEnv(SCENE, program="solo", speeds={"red_1": 1.0, "blue_1": 1.0})


def act(env, *chosen, durations=None):
    """Actions for both robots, as (name, args) pairs, with logged durations if given."""
    codes = [env.actions.encode(name, args) for name, args in chosen]
    actions = {
        "action": np.array([code for code, _ in codes], np.int32),
        "args": np.array([args for _, args in codes], np.int32),
    }
    if durations is not None:
        actions["duration"] = np.array(durations, np.float32)
    return actions


IDLE = ("idle", {})
PICK = ("pick", {"item": "ball"})


def place(goal):
    return ("place", {"goal": goal})


def test_reset_from_a_start_state(env):
    state = env.reset_from(start_at(), jax.random.key(0))
    assert float(state.t) == 0.0
    assert np.allclose(state.xy, [[-0.5, 0.0], [0.5, 0.0]])
    assert np.asarray(state.task["supply"]).tolist() == [4]
    assert np.asarray(env.ready(state)).all()
    assert env.agents == ["red_1", "blue_1"]


def test_start_states_keep_placed_and_held_balls(env):
    objects = [
        {"object_id": "a", "cls": "ball", "status": "placed", "goal_id": "blue", "attrs": {}},
        {"object_id": "c", "cls": "ball", "status": "held", "holder": "r", "attrs": {}},
    ]
    state = env.reset_from(start_at(objects=objects), jax.random.key(0))
    assert np.asarray(state.task["points"]).tolist() == [0, 1]
    assert np.asarray(state.task["held"]).tolist() == [[1], [0]]
    assert np.asarray(state.task["supply"]).tolist() == [2]
    assert float(state.lead) == -1.0


def test_pick_then_place_scores_and_rewards(env):
    step = jax.jit(env.step)
    key = jax.random.key(0)
    state = env.reset_from(start_at(), key)
    state, _, reward, _ = step(state, act(env, PICK, IDLE, durations=[1.0, 5.0]), key)
    assert float(state.t) == 1.0 and np.asarray(state.task["held"]).tolist() == [[1], [0]]
    assert np.asarray(env.ready(state)).tolist() == [True, False]
    state, _, reward, _ = step(state, act(env, place("red"), IDLE, durations=[2.0, -1]), key)
    assert float(state.t) == 3.0  # blue's idle ends at 5 s, the place at 3 s
    assert np.asarray(state.task["points"]).tolist() == [1, 0]
    assert np.asarray(reward).tolist() == [1.0, -1.0]
    # The red goal is at (-0.8, 0) and robots stop 0.2 m short.
    assert np.allclose(state.xy[0], [-0.6, 0.0], atol=1e-6)


def test_placing_without_a_ball_changes_nothing(env):
    key = jax.random.key(0)
    state = env.reset_from(start_at(), key)
    state, _, reward, _ = env.step(state, act(env, place("red"), IDLE, durations=[1.0, 1.0]), key)
    assert np.asarray(state.task["points"]).tolist() == [0, 0]
    assert np.asarray(reward).tolist() == [0.0, 0.0]


def test_checkpoints_are_scored_and_rewards_add_up(env):
    step = jax.jit(env.step)
    key = jax.random.key(1)
    state = env.reset_from(start_at(), key)
    # Red scores for itself before the end of autonomous at 10 s, then once more for each alliance.
    plan = [PICK, place("red"), ("idle", {}), PICK, place("red"), PICK, place("blue")]
    took = [1.0, 1.0, 9.0, 1.0, 1.0, 1.0, 1.0]
    total, times = np.zeros(2), []
    for _ in range(400):
        red = (plan.pop(0), took.pop(0)) if plan and bool(env.ready(state)[0]) else (IDLE, -1.0)
        state, _, reward, done = step(state, act(env, red[0], IDLE, durations=[red[1], -1.0]), key)
        total += np.asarray(reward)
        times.append(float(state.t))
        if bool(done):
            break
    assert bool(done) and float(state.t) == MATCH_END
    assert 10.0 in times and PLAY_END in times  # autonomous_end, then the end of play
    assert np.asarray(state.scores).tolist() == [[1.0, 0.0], [2.0, 1.0]]
    assert total.tolist() == [1.0, -1.0]  # the final lead, from 0-0
    after, _, reward, _ = step(state, act(env, PICK, PICK), key)  # a finished match stays put
    assert float(after.t) == MATCH_END and np.asarray(reward).tolist() == [0.0, 0.0]


def test_actions_still_going_when_play_stops_are_dropped(env):
    objects = [{"object_id": "c", "cls": "ball", "status": "held", "holder": "r", "attrs": {}}]
    state = env.reset_from(start_at(t=59.0, objects=objects), jax.random.key(0))
    key = jax.random.key(0)
    state, _, _, _ = env.step(state, act(env, place("red"), IDLE, durations=[2.0, 2.0]), key)
    assert float(state.t) == PLAY_END
    assert np.asarray(state.task["points"]).tolist() == [0, 0]
    assert not np.asarray(env.ready(state)).any()  # nothing starts after play stops


@pytest.mark.parametrize("detour", [1.0, 1.5])
def test_durations_come_from_the_timing(detour):
    timing = Timing({"pick": ActionTiming(median_s=2.5, sigma=0.0)}, speed_mps=0.5, detour=detour)
    env = PlanEnv(SCENE, timing, program="solo")
    key = jax.random.key(0)
    state = env.reset_from(start_at(), key)
    state, _, _, _ = env.step(state, act(env, PICK, PICK), key)
    assert np.asarray(state.t_end).tolist() == [2.5, 2.5]
    state, _, _, _ = env.step(state, act(env, place("red"), IDLE), key)
    # Driving 0.1 m in a straight line, times the detour, at 0.5 m/s; then the default 1 s
    # handling (sigma 0.5, so it varies).
    assert float(state.t_arrive[0]) == pytest.approx(2.5 + 0.2 * detour)
    assert np.allclose(state.to_xy[0], [-0.6, 0.0])  # the detour costs time, not distance


def test_failures_come_from_the_timing():
    timing = Timing({"place": ActionTiming(median_s=1.0, sigma=0.0, p_fail=1.0)})
    env = PlanEnv(SCENE, timing, program="solo")
    key = jax.random.key(0)
    state = env.reset_from(start_at(), key)
    for chosen in (PICK, place("red")):
        state, _, _, _ = env.step(state, act(env, chosen, IDLE), key)
        state, _, _, _ = env.step(state, act(env, IDLE, IDLE), key)
    assert np.asarray(state.task["points"]).tolist() == [0, 0]
    assert np.asarray(state.task["held"]).tolist() == [[1], [0]]  # still holding it


def test_encode_checks_the_start_state(env):
    with pytest.raises(ValueError, match="red robots"):
        env.encode(start_at().model_copy(update={"robots": start_at().robots[1:]}))
    with pytest.raises(ValueError, match="not solo"):
        env.encode(start_at().model_copy(update={"program": "other"}))


def test_observations(env):
    state = env.reset_from(start_at(), jax.random.key(0))
    obs = env.observe(state)
    assert obs.shape[0] == 2 and np.isfinite(obs).all()
    code = jnp.ones((2, 3))
    assert env.observe(state, code).shape == (2, obs.shape[1] + 3)


def test_jit_and_vmap_match_one_at_a_time(env):
    n = 8
    starts = [env.encode(start_at(red=(-0.5 + 0.05 * i, 0.0))) for i in range(n)]
    batch = jax.tree.map(lambda *leaves: np.stack(leaves), *starts)
    keys = jax.random.split(jax.random.key(3), n)
    reset = jax.jit(jax.vmap(env.reset_from))
    step = jax.jit(jax.vmap(env.step))
    states = reset(batch, keys)
    one = env.reset_from(starts[5], keys[5])
    for i in range(30):
        key = jax.random.key(100 + i)
        actions = jax.vmap(lambda k: random_actions(k, env.tables, 2))(jax.random.split(key, n))
        step_keys = jax.random.split(jax.random.fold_in(key, 1), n)
        states, obs, reward, done = step(states, actions, step_keys)
        one, _, one_reward, _ = env.step(one, jax.tree.map(lambda a: a[5], actions), step_keys[5])
        assert np.allclose(states.t[5], one.t) and np.allclose(reward[5], one_reward)
    assert obs.shape[:2] == (n, 2) and np.isfinite(obs).all()
    assert np.asarray(states.task["supply"] >= 0).all()
    assert np.allclose(states.xy[5], one.xy, atol=1e-5)
