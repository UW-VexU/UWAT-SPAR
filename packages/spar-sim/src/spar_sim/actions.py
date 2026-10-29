"""The scene's plan-level action set (spar_schema.ActionSet) as arrays, for the JAX environment.

An action chosen by a robot is an action index plus one code per argument, padded to max_args.
ActionTables says, for each action, how many labels each argument has and where the robot drives:
to the place named by one argument, to a fixed place, or to an opponent robot.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from spar_schema import ActionSet


@dataclass(frozen=True)
class ActionTables:
    arg_sizes: np.ndarray  # [action, arg] labels per argument, 1 where the action has none
    moves: np.ndarray  # [action] whether it drives somewhere first
    to_arg: np.ndarray  # [action] the argument naming the destination, or -1
    to_opponent: np.ndarray  # [action] the destination is an opponent robot
    fixed_xy: np.ndarray  # [action, 2] a fixed destination
    places: np.ndarray  # [action, label, 2] the destination for each label of to_arg
    standoff_m: np.ndarray  # [action] how far short of the destination it stops


def action_tables(actions: ActionSet) -> ActionTables:
    n, k = len(actions.actions), max(actions.max_args, 1)
    labels = max([len(arg.labels) for action in actions.actions for arg in action.args] or [1])
    arg_sizes = np.ones((n, k), np.int32)
    moves = np.zeros(n, bool)
    to_arg = np.full(n, -1, np.int32)
    to_opponent = np.zeros(n, bool)
    fixed_xy = np.zeros((n, 2), np.float32)
    places = np.zeros((n, labels, 2), np.float32)
    standoff = np.zeros(n, np.float32)
    for i, action in enumerate(actions.actions):
        for j, arg in enumerate(action.args):
            arg_sizes[i, j] = len(arg.labels)
        moves[i] = action.moves
        standoff[i] = actions.standoff_m if action.standoff else 0.0
        if action.to_xy is not None:
            fixed_xy[i] = action.to_xy
        if action.to_arg is not None:
            to_arg[i] = action.to_arg
            arg = action.args[action.to_arg]
            to_opponent[i] = arg.opponents
            if arg.positions is not None:
                places[i, : len(arg.positions)] = arg.positions
    return ActionTables(arg_sizes, moves, to_arg, to_opponent, fixed_xy, places, standoff)


def random_actions(key: jax.Array, tables: ActionTables, n_agents: int) -> dict[str, jax.Array]:
    """A uniformly random action and arguments for each robot, for tests and benchmarks."""
    pick, fill = jax.random.split(key)
    action = jax.random.randint(pick, (n_agents,), 0, len(tables.moves))
    sizes = jnp.asarray(tables.arg_sizes)[action]
    args = jnp.floor(jax.random.uniform(fill, sizes.shape) * sizes).astype(jnp.int32)
    return {"action": action, "args": args}
