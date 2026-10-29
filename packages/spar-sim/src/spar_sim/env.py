"""PlanEnv: the plan-level simulator, as pure functions reset_from(start, key) and step(state,
actions, key) (spec section 6.7).

    env = PlanEnv(scene, timing, program="vexu")
    state = env.reset_from(start_state, key)
    state, obs, reward, done = env.step(state, actions, key)

Both are pure, so jax.jit(env.step) and jax.jit(jax.vmap(env.step)) work; for a batch, encode the
start states first with env.encode() and stack them. The scene declares the actions (its `sim`
section) and says what each does to its task arrays (its `sim` plugin); this module knows nothing
about any game.

Time. Each robot does one action at a time, and is ready when it has none. step() starts the
chosen action of each ready robot (the others' choices are ignored), then moves time on to the
next thing that happens: an action ends, a checkpoint is scored, or play stops. An action drives
to its destination, stopping standoff_m short of an object or a robot, then takes a handling time
drawn from the fitted log-normal. Real robots curve and turn, so the drive takes the straight-line
distance times the fitted detour ratio, at the robot's average driving speed; positions along the
way are drawn on the straight line, and nothing steers around field elements. An action can fail
with the fitted probability, a different one when an opponent is within defend_radius_m as it
starts; a failed action takes its time but changes nothing. Actions still going when play stops
are dropped, and nothing starts after that.

actions is {"action": int[agent], "args": int[agent, arg]}, with arguments as codes into each
argument's labels (spar_schema.ActionSet.encode), and optionally "duration": float[agent]. A robot
given a duration of 0 or more takes exactly that long and does not fail; replay.py uses this to
rerun logged actions.

Rewards. Each robot's reward is the change in its alliance's lead, scored as if the match ended
now (the last checkpoint), so a match's rewards add up to the final lead less the lead at the
start. Two alliances; the first one's lead is positive.
"""

import functools
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from spar_schema import Scene, StartState, action_set
from spar_sim.actions import action_tables
from spar_sim.fit import Timing
from spar_sim.observations import observe

MIN_DURATION_S = 1e-3


class State(NamedTuple):
    t: jax.Array  # seconds from match start
    task: dict[str, jax.Array]  # the scene's task arrays
    xy: jax.Array  # [agent, 2] where each robot is at t
    from_xy: jax.Array  # [agent, 2] where its current action started
    to_xy: jax.Array  # [agent, 2] where it drives to
    t_start: jax.Array  # [agent]
    t_arrive: jax.Array  # [agent] when it gets there
    t_end: jax.Array  # [agent] when the action ends
    pending: jax.Array  # [agent] an action is under way
    action: jax.Array  # [agent] int32
    args: jax.Array  # [agent, arg] int32
    fails: jax.Array  # [agent] the action under way will fail
    speed: jax.Array  # [agent] m/s
    scores: jax.Array  # [checkpoint, alliance] points, NaN until scored
    next_checkpoint: jax.Array  # index of the next checkpoint to score
    lead: jax.Array  # the first alliance's lead, scored as if the match ended now
    done: jax.Array


class PlanEnv:
    def __init__(
        self,
        scene: Scene,
        timing: Timing | None = None,
        *,
        program: str,
        carry: int | None = None,
        speeds: dict[str, float] | None = None,
    ):
        """carry overrides the scene's sim.carry, and speeds fixes some robots' average driving
        speeds (m/s), which are lower than their top speeds."""
        if len(scene.alliances) != 2:
            raise ValueError("PlanEnv supports two alliances")
        self.scene, self.program = scene, program
        self.actions = action_set(scene, program)
        self.timing = timing or Timing.default(self.actions)
        self.tables = action_tables(self.actions)
        self.rules = scene.plugin("sim")(scene, program)
        periods = scene.programs[program]
        per_alliance = periods.robots_per_alliance
        self.agents = [f"{a}_{i + 1}" for a in scene.alliances for i in range(per_alliance)]
        self.alliance = np.repeat(np.arange(2, dtype=np.int32), per_alliance)
        # The other alliance's robots, for each robot, by number.
        self.opponents = np.array(
            [[(1 - k) * per_alliance + i for i in range(per_alliance)] for k in self.alliance]
        )
        self.carry = carry if carry is not None else self.actions.carry
        fixed = speeds or {}
        self.fixed_speed = np.array([fixed.get(a, np.nan) for a in self.agents], np.float32)
        self.timing_arrays = self.timing.arrays(self.actions)
        self.play_end = float(periods.autonomous_s) + float(periods.driver_s)
        self.match_length = self.play_end
        self.checkpoint_times = np.array(
            [scene.checkpoint_time(c, program) for c in scene.checkpoints], np.float32
        )
        self.idle = self.actions.index("idle") if "idle" in self.actions.names else 0
        self._branches = [self._branch(name) for name in self.actions.names]

    # Starting.

    def slot_agents(self, start: StartState) -> list[str]:
        """The start state's robot in each of this env's slots (self.agents), by alliance."""
        found = []
        for k, alliance in enumerate(self.scene.alliances):
            robots = sorted(r.agent_id for r in start.robots if r.alliance == alliance)
            wanted = int(np.sum(self.alliance == k))
            if len(robots) != wanted:
                raise ValueError(
                    f"start state {start.episode_id} at {start.t:.1f} s has {len(robots)} "
                    f"{alliance} robots with a pose; the {self.program} program has {wanted}"
                )
            found += robots
        return found

    def encode(self, start: StartState) -> dict[str, Any]:
        """A StartState as arrays, ready for reset_from (and for stacking into a batch)."""
        if start.program is not None and start.program != self.program:
            raise ValueError(f"start state is from {start.program}, not {self.program}")
        slots = self.slot_agents(start)
        poses = {r.agent_id: r for r in start.robots}
        return {
            "t": np.float32(start.t),
            "xy": np.array([(poses[a].x, poses[a].y) for a in slots], np.float32),
            "task": self.rules.encode(start, slots),
        }

    def reset_from(self, start: StartState | dict[str, Any], key: jax.Array) -> State:
        if isinstance(start, StartState):
            start = self.encode(start)
        t = jnp.asarray(start["t"], jnp.float32)
        xy = jnp.asarray(start["xy"], jnp.float32)
        task = jax.tree.map(jnp.asarray, start["task"])
        n, k = len(self.agents), self.tables.arg_sizes.shape[1]
        spread = self.timing.speed_sigma * jax.random.normal(key, (n,))
        sampled = self.timing.speed_mps * jnp.exp(spread)
        speed = jnp.where(jnp.isnan(self.fixed_speed), sampled, self.fixed_speed)
        times = jnp.asarray(self.checkpoint_times)
        next_checkpoint = jnp.sum(times < t).astype(jnp.int32)
        state = State(
            t=t,
            task=task,
            xy=xy,
            from_xy=xy,
            to_xy=xy,
            t_start=jnp.full(n, t),
            t_arrive=jnp.full(n, t),
            t_end=jnp.full(n, t),
            pending=jnp.zeros(n, bool),
            action=jnp.full(n, self.idle, jnp.int32),
            args=jnp.zeros((n, k), jnp.int32),
            fails=jnp.zeros(n, bool),
            speed=speed.astype(jnp.float32),
            scores=jnp.full((len(times), 2), jnp.nan, jnp.float32),
            next_checkpoint=next_checkpoint,
            lead=jnp.float32(0.0),
            done=next_checkpoint >= len(times),
        )
        return state._replace(lead=self._lead(state.task, xy))

    # Stepping.

    def step(self, state: State, actions: dict[str, jax.Array], key: jax.Array):
        tb, n = self.tables, len(self.agents)
        timing = self.timing_arrays
        k_handle, k_fail = jax.random.split(key)

        # Start the chosen action of every ready robot.
        ready = ~state.pending & (state.t < self.play_end) & ~state.done
        chosen = jnp.asarray(actions["action"], jnp.int32) % len(tb.moves)
        args = jnp.asarray(actions["args"], jnp.int32) % jnp.asarray(tb.arg_sizes)[chosen]
        target = self._destination(state.xy, chosen, args)
        gap = target - state.xy
        distance = jnp.sqrt(jnp.sum(gap * gap, axis=-1))
        drive = jnp.maximum(distance - jnp.asarray(tb.standoff_m)[chosen], 0.0)
        to_xy = state.xy + gap * (drive / jnp.maximum(distance, 1e-6))[:, None]
        drive_s = drive * self.timing.detour / state.speed
        mu, sigma = jnp.asarray(timing["mu"])[chosen], jnp.asarray(timing["sigma"])[chosen]
        handle_s = jnp.exp(mu + sigma * jax.random.normal(k_handle, (n,)))
        forced = jnp.asarray(actions.get("duration", jnp.full(n, -1.0)), jnp.float32)
        logged = forced >= 0
        duration = jnp.maximum(jnp.where(logged, forced, drive_s + handle_s), MIN_DURATION_S)
        defended = self._defended(state.xy)
        p_fail = jnp.where(
            defended,
            jnp.asarray(timing["p_fail_defended"])[chosen],
            jnp.asarray(timing["p_fail"])[chosen],
        )
        fails = (jax.random.uniform(k_fail, (n,)) < p_fail) & ~logged

        def pick(new, old):
            mask = ready.reshape(ready.shape + (1,) * (new.ndim - 1))
            return jnp.where(mask, new, old)

        state = state._replace(
            from_xy=pick(state.xy, state.from_xy),
            to_xy=pick(to_xy, state.to_xy),
            t_start=pick(jnp.full(n, state.t), state.t_start),
            t_arrive=pick(state.t + jnp.minimum(drive_s, duration), state.t_arrive),
            t_end=pick(state.t + duration, state.t_end),
            pending=state.pending | ready,
            action=pick(chosen, state.action),
            args=pick(args, state.args),
            fails=pick(fails, state.fails),
        )

        # Move time on to the next action end, checkpoint or the end of play.
        times = jnp.asarray(self.checkpoint_times)
        c = jnp.minimum(state.next_checkpoint, len(times) - 1)
        checkpoint_t = jnp.where(state.next_checkpoint < len(times), times[c], jnp.inf)
        ends = jnp.where(state.pending & (state.t_end <= self.play_end), state.t_end, jnp.inf)
        stop = jnp.where(state.t < self.play_end, self.play_end, jnp.inf)
        t_next = jnp.minimum(jnp.minimum(jnp.min(ends), checkpoint_t), stop)
        t_next = jnp.where(jnp.isfinite(t_next), jnp.maximum(t_next, state.t), state.t)
        xy = self._positions(state, t_next)

        # Finish the actions that end now, in robot order. At the end of play, drop the rest.
        task = state.task
        finished = state.pending & (state.t_end <= t_next)
        for i in range(n):
            ctx = {
                "agent": i,
                "alliance": int(self.alliance[i]),
                "args": state.args[i],
                "t": state.t_end[i],
                "carry": self.carry,
                "xy": xy,
            }
            new, ok = jax.lax.switch(state.action[i], self._branches, task, ctx)
            apply = finished[i] & ~state.fails[i] & (state.t_end[i] <= self.play_end) & ok
            task = jax.tree.map(functools.partial(jnp.where, apply), new, task)
        pending = state.pending & ~finished & (t_next < self.play_end)

        # Score a checkpoint that falls now.
        hit = (state.next_checkpoint < len(times)) & (t_next >= checkpoint_t)
        points = self._score(task, xy, c)
        scores = jnp.where(hit, state.scores.at[c].set(points), state.scores)
        marked = self.rules.on_checkpoint(task, c, scores, jnp)
        task = jax.tree.map(lambda a, b: jnp.where(hit, jnp.asarray(a, b.dtype), b), marked, task)
        next_checkpoint = state.next_checkpoint + hit.astype(jnp.int32)

        lead = self._lead(task, xy)
        sign = jnp.where(jnp.asarray(self.alliance) == 0, 1.0, -1.0)
        reward = sign * (lead - state.lead)
        state = state._replace(
            t=t_next,
            task=task,
            xy=xy,
            pending=pending,
            scores=scores,
            next_checkpoint=next_checkpoint,
            lead=lead,
            done=next_checkpoint >= len(times),
        )
        return state, self.observe(state), reward, state.done

    def observe(self, state: State, code: jax.Array | None = None) -> jax.Array:
        return observe(self, state, code)

    def ready(self, state: State) -> jax.Array:
        """Which robots choose an action at the next step."""
        return ~state.pending & (state.t < self.play_end) & ~state.done

    # Helpers.

    def _branch(self, name: str):
        effect = self.rules.effects.get(name)

        def run(task, ctx):
            if effect is None:
                return task, jnp.asarray(True)
            new, ok = effect(task, ctx, jnp)
            new = {key: jnp.asarray(new[key], task[key].dtype) for key in task}
            return new, jnp.asarray(ok, bool)

        return run

    def _destination(self, xy, chosen, args):
        tb, n = self.tables, len(self.agents)
        to_arg = jnp.asarray(tb.to_arg)[chosen]
        label = jnp.take_along_axis(args, jnp.maximum(to_arg, 0)[:, None], axis=1)[:, 0]
        places = jnp.asarray(tb.places)
        place_xy = places[chosen, jnp.minimum(label, places.shape[1] - 1)]
        opponents = jnp.asarray(self.opponents)
        opponent = opponents[jnp.arange(n), label % opponents.shape[1]]
        dest = jnp.where((to_arg >= 0)[:, None], place_xy, jnp.asarray(tb.fixed_xy)[chosen])
        dest = jnp.where(jnp.asarray(tb.to_opponent)[chosen][:, None], xy[opponent], dest)
        return jnp.where(jnp.asarray(tb.moves)[chosen][:, None], dest, xy)

    def _positions(self, state: State, t):
        t = jnp.minimum(t, self.play_end)
        span = jnp.maximum(state.t_arrive - state.t_start, 1e-6)
        along = jnp.clip((t - state.t_start) / span, 0.0, 1.0)[:, None]
        moving = state.from_xy + (state.to_xy - state.from_xy) * along
        return jnp.where(state.pending[:, None], moving, state.xy)

    def _defended(self, xy):
        gap = xy[:, None, :] - xy[jnp.asarray(self.opponents)]
        near = jnp.sqrt(jnp.sum(gap * gap, axis=-1)) <= self.timing.defend_radius_m
        return jnp.any(near, axis=1)

    def _score(self, task, xy, checkpoint):
        robots = {"xy": xy, "alliance": jnp.asarray(self.alliance)}
        points = self.scene.score(self.rules.task_state(task, robots, checkpoint, jnp), jnp)
        return jnp.stack([points[a] for a in self.scene.alliances]).astype(jnp.float32)

    def _lead(self, task, xy):
        points = self._score(task, xy, len(self.checkpoint_times) - 1)
        return points[0] - points[1]
