"""PettingZoo ParallelEnv wrapper for non-JAX users.

    env = ParallelPlanEnv(PlanEnv(scene, timing, program="vexu"), start_states)
    observations, infos = env.reset(seed=0)
    observations, rewards, terminations, truncations, infos = env.step(actions)

Each robot's action is a MultiDiscrete vector: the action index, then one code per argument
slot. Codes past an action's labels wrap around, and slots it does not use are ignored. A robot
still busy with its last action has infos[agent]["ready"] False, and its new action is ignored.
reset() starts from one of the start states, chosen at random from the seed, or the one given as
options={"start": i}.
"""

import functools

import jax
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from spar_schema import StartState
from spar_sim.env import PlanEnv


class ParallelPlanEnv(ParallelEnv):
    metadata = {"name": "spar_plan_v0"}

    def __init__(self, env: PlanEnv, starts: list[StartState]):
        if not starts:
            raise ValueError("ParallelPlanEnv needs at least one start state")
        self.env = env
        self.starts = [env.encode(start) for start in starts]
        self.possible_agents = list(env.agents)
        self.agents: list[str] = []
        self._step = jax.jit(env.step)
        self._reset = jax.jit(env.reset_from)
        self._obs_size = int(
            env.observe(env.reset_from(self.starts[0], jax.random.key(0))).shape[1]
        )
        self._sizes = [len(env.tables.moves), *env.tables.arg_sizes.max(axis=0).tolist()]
        self._rng = np.random.default_rng()
        self.state = None

    @functools.cache  # noqa: B019 - the same space object each call, as PettingZoo asks
    def observation_space(self, agent: str) -> spaces.Box:
        return spaces.Box(-np.inf, np.inf, (self._obs_size,), np.float32)

    @functools.cache  # noqa: B019
    def action_space(self, agent: str) -> spaces.MultiDiscrete:
        return spaces.MultiDiscrete(self._sizes)

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        options = options or {}
        index = options.get("start", int(self._rng.integers(len(self.starts))))
        key = jax.random.key(int(self._rng.integers(2**31)))
        self.state = self._reset(self.starts[index], key)
        self.agents = list(self.possible_agents)
        return self._observations(self.env.observe(self.state)), self._infos()

    def step(self, actions: dict[str, np.ndarray]):
        n = len(self.possible_agents)
        chosen = np.zeros((n, len(self._sizes)), np.int32)
        chosen[:, 0] = self.env.idle
        for i, agent in enumerate(self.possible_agents):
            if agent in actions:
                chosen[i] = np.asarray(actions[agent], np.int32)
        key = jax.random.key(int(self._rng.integers(2**31)))
        self.state, obs, reward, done = self._step(
            self.state, {"action": chosen[:, 0], "args": chosen[:, 1:]}, key
        )
        agents = list(self.agents)
        reward = np.asarray(reward)
        rewards = {a: float(reward[self.possible_agents.index(a)]) for a in agents}
        terminations = {a: bool(done) for a in agents}
        truncations = {a: False for a in agents}
        observations, infos = self._observations(obs), self._infos()
        if bool(done):
            self.agents = []
        return observations, rewards, terminations, truncations, infos

    def _observations(self, obs) -> dict[str, np.ndarray]:
        obs = np.asarray(obs, np.float32)
        return {a: obs[self.possible_agents.index(a)] for a in self.agents}

    def _infos(self) -> dict[str, dict]:
        ready = np.asarray(self.env.ready(self.state))
        t = float(self.state.t)
        return {
            a: {"ready": bool(ready[self.possible_agents.index(a)]), "t": t} for a in self.agents
        }
