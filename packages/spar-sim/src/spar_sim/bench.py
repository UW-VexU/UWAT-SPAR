"""Throughput benchmark: how many simulator steps per second, many matches at once.

benchmark() runs n_envs copies of one start state under jax.jit(jax.vmap(env.step)), choosing
random actions, for n_steps steps inside one lax.scan, and times it after a first run that
compiles. spec section 6.7 asks for this number on the lab GPU.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np

from spar_schema import RobotState, StartState
from spar_sim.actions import random_actions
from spar_sim.env import PlanEnv


def benchmark(
    env: PlanEnv,
    start: StartState | None = None,
    *,
    n_envs: int = 1024,
    n_steps: int = 200,
    seed: int = 0,
) -> dict[str, float | int | str]:
    encoded = env.encode(start or spread_start(env))
    batch = jax.tree.map(lambda leaf: np.broadcast_to(leaf, (n_envs, *np.shape(leaf))), encoded)
    n = len(env.agents)
    reset = jax.jit(jax.vmap(env.reset_from))
    step = jax.vmap(env.step)
    choose = jax.vmap(lambda key: random_actions(key, env.tables, n))

    @jax.jit
    def run(states, key):
        def body(states, key):
            pick, move = jax.random.split(key)
            actions = choose(jax.random.split(pick, n_envs))
            states, _, _, done = step(states, actions, jax.random.split(move, n_envs))
            return states, jnp.mean(done)

        return jax.lax.scan(body, states, jax.random.split(key, n_steps))

    key = jax.random.key(seed)
    states = reset(batch, jax.random.split(key, n_envs))
    jax.block_until_ready(run(states, key))  # compile
    began = time.perf_counter()
    final, done = jax.block_until_ready(run(states, key))
    seconds = time.perf_counter() - began
    return {
        "device": jax.devices()[0].platform,
        "envs": n_envs,
        "steps": n_steps,
        "seconds": seconds,
        "env_steps_per_s": n_envs * n_steps / seconds,
        "share_done": float(done[-1]),
        "mean_match_time_s": float(jnp.mean(final.t)),
    }


def spread_start(env: PlanEnv) -> StartState:
    """An empty field at time 0 with the robots on a circle around the center, for benchmarks."""
    n = len(env.agents)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    robots = [
        RobotState(
            agent_id=agent,
            x=float(np.cos(a)),
            y=float(np.sin(a)),
            heading=0.0,
            vx=0.0,
            vy=0.0,
            omega=0.0,
            alliance=env.scene.alliances[int(env.alliance[i])],
        )
        for i, (agent, a) in enumerate(zip(env.agents, angles, strict=True))
    ]
    return StartState(
        episode_id="benchmark",
        t=0.0,
        time_left_s=env.match_length,
        robots=robots,
        program=env.program,
    )
