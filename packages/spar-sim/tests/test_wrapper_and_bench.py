"""The PettingZoo wrapper and the throughput benchmark, on the toy scene."""

from pathlib import Path

import numpy as np
from pettingzoo.test import parallel_api_test

from spar_schema import RobotState, StartState, load_scene
from spar_sim.bench import benchmark, spread_start
from spar_sim.env import PlanEnv
from spar_sim.pettingzoo_wrapper import ParallelPlanEnv

TOY_SCENE = Path(__file__).parents[2] / "spar-schema" / "tests" / "fixtures" / "toy_scene"
SCENE = load_scene(TOY_SCENE, strict=True)


def start():
    robots = [
        RobotState(agent_id=name, x=x, y=0.0, heading=0.0, vx=0.0, vy=0.0, omega=0.0,
                   alliance=alliance)
        for name, alliance, x in (("r", "red", -0.5), ("b", "blue", 0.5))
    ]  # fmt: skip
    return StartState(episode_id="toy", t=0.0, time_left_s=60.0, robots=robots)


def test_pettingzoo_api():
    parallel_api_test(ParallelPlanEnv(PlanEnv(SCENE, program="solo"), [start()]), num_cycles=200)


def test_pettingzoo_match_runs_to_the_end():
    env = ParallelPlanEnv(PlanEnv(SCENE, program="solo"), [start()])
    observations, infos = env.reset(seed=0)
    assert set(observations) == {"red_1", "blue_1"} and infos["red_1"]["ready"]
    pick, place = env.env.actions.index("pick"), env.env.actions.index("place")
    total = {agent: 0.0 for agent in env.possible_agents}
    for i in range(1000):
        if not env.agents:
            break
        chosen = np.array([pick if i % 2 == 0 else place, 0])  # red drops balls in its goal
        _, rewards, terminations, _, infos = env.step({"red_1": chosen})
        for agent, reward in rewards.items():
            total[agent] += reward
    assert not env.agents and all(terminations.values())
    assert total["red_1"] > 0 and total["blue_1"] == -total["red_1"]


def test_benchmark_runs():
    env = PlanEnv(SCENE, program="solo")
    assert len(spread_start(env).robots) == 2
    result = benchmark(env, n_envs=16, n_steps=50)
    assert result["env_steps_per_s"] > 0 and result["envs"] == 16
    assert 0.0 <= result["share_done"] <= 1.0
