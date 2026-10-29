"""Observations: positions, object states, time left, plus a strategy code when conditioned.

observe() gives one row per robot, all the same length:
    time left in play, as a share of the match
    which robot this is (one-hot over the env's agents)
    every robot's position (m), whether it is free to choose, and seconds until it is
    the scene's task arrays, flattened (goal contents, Toggle owners, what robots hold, ...)
    the strategy code row for this robot, when code [agent, n] is given
"""

import jax
import jax.numpy as jnp


def observe(env, state, code=None) -> jax.Array:
    n = len(env.agents)
    time_left = jnp.maximum(env.play_end - state.t, 0.0) / env.match_length
    free = ~state.pending
    wait = jnp.where(state.pending, jnp.maximum(state.t_end - state.t, 0.0), 0.0)
    task = jnp.concatenate(
        [jnp.ravel(leaf).astype(jnp.float32) for leaf in jax.tree.leaves(state.task)]
    )
    shared = jnp.concatenate(
        [
            jnp.ravel(state.xy),
            free.astype(jnp.float32),
            wait,
            task,
        ]
    )
    rows = jnp.concatenate(
        [
            jnp.full((n, 1), time_left),
            jnp.eye(n, dtype=jnp.float32),
            jnp.broadcast_to(shared, (n, shared.shape[0])),
        ],
        axis=1,
    )
    if code is not None:
        rows = jnp.concatenate([rows, jnp.asarray(code, jnp.float32)], axis=1)
    return rows
