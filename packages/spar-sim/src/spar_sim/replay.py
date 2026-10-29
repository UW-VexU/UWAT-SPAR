"""The replay check: rerun an episode's actions from its start and compare the score.

replay_episode() starts the simulator from the episode's first moment with every robot tracked,
and feeds each robot its actions from actions.parquet in order, until the match is scored. With
logged_times (the default) each action takes as long as it did in the episode and never fails, so
the check tests what the actions do; without, durations and failures come from the timing, so it
also tests the fitted timing. Carrying is not limited in a logged replay, since perception can
miss a place and leave a robot holding something it no longer has.

The score error, the simulator's final points less the official ones, is one of the paper's
metrics (spec section 6.7). The points spar-state computed from the episode's objects are
reported too: the difference from those isolates the simulator and the actions, without
perception's errors.
"""

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import jax
import numpy as np

from spar_schema import EpisodeMoments, EpisodeReader, Scene
from spar_sim.env import PlanEnv
from spar_sim.fit import Timing

LOGGED_CARRY = 99


@dataclass(frozen=True)
class ReplayResult:
    episode_id: str
    points: dict[str, dict[str, float]]  # the simulator's, by checkpoint, then alliance
    official: dict[str, float] | None  # final points from outcome.json
    computed: dict[str, float] | None  # final points spar-state computed, from the score check
    actions: int  # actions replayed
    steps: int
    env: PlanEnv | None = field(default=None, repr=False)
    states: list | None = field(default=None, repr=False)  # each step's State, with trace=True

    @property
    def final(self) -> dict[str, float]:
        return list(self.points.values())[-1]

    @property
    def error(self) -> dict[str, float] | None:
        """Simulated less official final points, by alliance."""
        if self.official is None:
            return None
        return {a: self.final[a] - self.official[a] for a in self.official}

    @property
    def error_vs_computed(self) -> dict[str, float] | None:
        if self.computed is None:
            return None
        return {a: self.final[a] - self.computed[a] for a in self.computed}


def replay_episode(
    path: str | Path,
    scene: Scene,
    timing: Timing | None = None,
    *,
    logged_times: bool = True,
    seed: int = 0,
    env: PlanEnv | None = None,
    trace: bool = False,
) -> ReplayResult:
    episode = EpisodeReader(path)
    if episode.actions is None or episode.tracks is None:
        raise ValueError(f"{path}: needs tracks and actions.parquet (run spar state run first)")
    meta = episode.meta
    if env is None:
        carry = LOGGED_CARRY if logged_times else None
        env = PlanEnv(scene, timing, program=meta.program, carry=carry)
    start = EpisodeMoments(episode, scene).at(_all_tracked(episode))
    slots = env.slot_agents(start)
    queues = {i: [] for i in range(len(slots))}
    for row in sorted(episode.actions.to_pylist(), key=lambda r: (r["t_start"], r["t_end"])):
        if row["agent_id"] in slots:
            code, args = env.actions.encode(row["action"], _args(row["args"]))
            queues[slots.index(row["agent_id"])].append((code, args, row["t_end"]))
    total = sum(len(q) for q in queues.values())

    step = jax.jit(env.step)
    key = jax.random.key(seed)
    key, first = jax.random.split(key)
    state = env.reset_from(env.encode(start), first)
    states = [state] if trace else None
    n, width = len(slots), env.tables.arg_sizes.shape[1]
    steps = 0
    while not bool(state.done):
        ready = np.asarray(env.ready(state))
        t = float(state.t)
        action = np.full(n, env.idle, np.int32)
        args = np.zeros((n, width), np.int32)
        duration = np.full(n, -1.0, np.float32)
        for i in np.flatnonzero(ready):
            if queues[i]:
                action[i], args[i], end = queues[i].pop(0)
                if logged_times:
                    duration[i] = max(end - t, 0.0)
            else:
                duration[i] = env.play_end - t + 1.0  # nothing left to do: wait out the match
        key, sub = jax.random.split(key)
        state, _, _, _ = step(state, {"action": action, "args": args, "duration": duration}, sub)
        steps += 1
        if trace:
            states.append(state)
        if steps > 10 * (total + len(env.checkpoint_times) + n) + 100:
            raise RuntimeError(f"{path}: the replay did not finish")

    scores = np.asarray(state.scores)
    points = {
        checkpoint: {a: float(scores[c, k]) for k, a in enumerate(scene.alliances)}
        for c, checkpoint in enumerate(scene.checkpoints)
    }
    official = None
    if episode.outcome is not None:
        official = {"red": episode.outcome.points_red, "blue": episode.outcome.points_blue}
    last = meta.score_check.get(scene.checkpoints[-1])
    computed = dict(last.computed) if last is not None else None
    return ReplayResult(meta.episode_id, points, official, computed, total, steps, env, states)


def replay_summary(results: Iterable[ReplayResult]) -> dict[str, float]:
    """Mean absolute error of the final points, per alliance and over both, against the official
    points and against spar-state's computed ones."""
    results = list(results)
    summary: dict[str, float] = {"episodes": len(results)}
    for name, errors in (
        ("official", [r.error for r in results if r.error]),
        ("computed", [r.error_vs_computed for r in results if r.error_vs_computed]),
    ):
        if errors:
            values = np.array([list(e.values()) for e in errors], float)
            summary[f"mae_vs_{name}"] = float(np.mean(np.abs(values)))
            summary[f"exact_vs_{name}"] = float(np.mean(np.all(values == 0, axis=1)))
    return summary


def _all_tracked(episode: EpisodeReader) -> float:
    """The first time every robot in the episode has a track row, and not before 0."""
    tracks = episode.tracks
    agents = np.asarray(tracks.column("agent_id").to_pylist())
    times = tracks.column("t").to_numpy()
    return max(0.0, max(float(times[agents == a].min()) for a in set(agents)))


def _args(raw: str | None) -> dict:
    return json.loads(raw) if raw else {}
