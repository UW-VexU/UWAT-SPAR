"""Action timing for the simulator, fitted from real episodes (spar sim fit).

fit_timing() reads each episode's actions.parquet (written by spar state run), events and tracks,
and fits:
    speed_mps, speed_sigma    robot speed: the median over robots of each robot's average speed
                              while driving (so slowing for turns counts), and the log-normal
                              spread between robots
    detour                    how much longer robots' tracked paths are than straight lines, over
                              drives of at least MIN_DRIVE_M; the simulator drives the straight
                              distance times this
    median_s, sigma           per action, the handling time: how long the action took, less the
                              time to drive its tracked path at the robot's speed, as a log-normal
    p_fail, p_fail_defended   for actions that put an object somewhere (from place events): the
                              share whose object was taken off again within fail_window_s, without
                              and with an opponent within defend_radius_m as it ended
Implied actions (_implied) are not fitted. An action with fewer than min_samples examples keeps
its default: a 1 s median, sigma 0.5 and no failures; idle is always a fixed 0.5 s.
"""

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from spar_schema import ActionSet, EpisodeReader, Scene, action_set

IDLE = "idle"
MIN_HANDLING_S = 0.05
MOVING_MPS = 0.05  # slower than this counts as standing still
PATH_STEP_S = 0.5  # a path is measured between positions this far apart, so jitter adds no length
MIN_DRIVE_M = 0.3  # shorter drives are mostly turning in place, and are left out of the detour


@dataclass
class ActionTiming:
    median_s: float = 1.0  # handling time, after any driving
    sigma: float = 0.5  # log-normal spread of the handling time
    p_fail: float = 0.0
    p_fail_defended: float = 0.0
    n: int = 0  # actions it was fitted from


@dataclass
class Timing:
    actions: dict[str, ActionTiming] = field(default_factory=dict)
    speed_mps: float = 1.0  # average speed while driving
    speed_sigma: float = 0.0
    detour: float = 1.0  # tracked path length over straight-line distance
    defend_radius_m: float = 0.6
    episodes: int = 0  # episodes it was fitted from

    @classmethod
    def default(cls, actions: ActionSet) -> "Timing":
        return cls({name: _default(name) for name in actions.names})

    def get(self, name: str) -> ActionTiming:
        return self.actions.get(name) or _default(name)

    def arrays(self, actions: ActionSet) -> dict[str, np.ndarray]:
        """Per action, in action-set order: mu and sigma of the log handling time, and p_fail."""
        rows = [self.get(name) for name in actions.names]
        return {
            "mu": np.log([max(row.median_s, MIN_HANDLING_S) for row in rows]).astype(np.float32),
            "sigma": np.array([row.sigma for row in rows], np.float32),
            "p_fail": np.array([row.p_fail for row in rows], np.float32),
            "p_fail_defended": np.array([row.p_fail_defended for row in rows], np.float32),
        }

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> "Timing":
        raw = json.loads(text)
        raw["actions"] = {name: ActionTiming(**row) for name, row in raw["actions"].items()}
        return cls(**raw)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Timing":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


def _default(name: str) -> ActionTiming:
    return ActionTiming(median_s=0.5, sigma=0.0) if name == IDLE else ActionTiming()


def fit_timing(
    episodes: Iterable[EpisodeReader],
    scene: Scene,
    *,
    min_samples: int = 5,
    fail_window_s: float = 2.0,
    defend_radius_m: float = 0.6,
) -> Timing:
    speeds: list[float] = []
    drives: list[tuple[float, float]] = []  # (straight-line distance, tracked path length)
    handling: dict[str, list[float]] = {}
    placed: dict[str, list[tuple[bool, bool]]] = {}  # (failed, defended), by action
    names: set[str] = set()
    used = 0
    for episode in episodes:
        if episode.actions is None or episode.tracks is None:
            continue
        names |= set(action_set(scene, episode.meta.program).names)
        used += 1
        paths = _Paths(episode.tracks)
        robot_speeds = paths.speeds()
        speeds += list(robot_speeds.values())
        fallback = float(np.median(list(robot_speeds.values()))) if robot_speeds else 1.0
        removed = _removals(episode)
        roster = episode.meta.roster
        for row in episode.actions.to_pylist():
            args = json.loads(row["args"]) if row["args"] else {}
            if args.get("_implied"):
                continue
            agent, start, end = row["agent_id"], row["t_start"], row["t_end"]
            if agent not in paths.agents:
                continue
            straight = float(np.hypot(*(paths.at(agent, end) - paths.at(agent, start))))
            path = paths.length(agent, start, end)
            if straight >= MIN_DRIVE_M:
                drives.append((straight, path))
            took = (end - start) - path / robot_speeds.get(agent, fallback)
            handling.setdefault(row["action"], []).append(max(took, MIN_HANDLING_S))
            if args.get("_event") == "place" and args.get("_object"):
                later = removed.get(args["_object"], [])
                failed = any(end < t <= end + fail_window_s for t in later)
                defended = _defended(paths, roster, agent, end, defend_radius_m)
                placed.setdefault(row["action"], []).append((failed, defended))

    timing = Timing(
        {name: _default(name) for name in sorted(names)},
        defend_radius_m=defend_radius_m,
        episodes=used,
    )
    if speeds:
        timing.speed_mps = float(np.median(speeds))
        timing.speed_sigma = float(np.std(np.log(speeds))) if len(speeds) > 1 else 0.0
    if drives:
        straight, path = np.sum(drives, axis=0)
        timing.detour = max(float(path / straight), 1.0)
    for name, values in handling.items():
        if name == IDLE or len(values) < min_samples:
            continue
        logs = np.log(values)
        median, sigma = float(np.exp(logs.mean())), float(logs.std())
        timing.actions[name] = ActionTiming(median, sigma, n=len(values))
    for name, outcomes in placed.items():
        if timing.actions[name].n:
            row = timing.actions[name]
            row.p_fail, row.p_fail_defended = _fail_rates(outcomes, min_samples)
    return timing


class _Paths:
    """Each robot's track in one episode, as arrays, to ask where it was and how far it drove."""

    def __init__(self, tracks):
        by_agent: dict[str, list[dict[str, Any]]] = {}
        for row in sorted(tracks.to_pylist(), key=lambda r: r["t"]):
            by_agent.setdefault(row["agent_id"], []).append(row)
        self.rows = by_agent
        self.agents = set(by_agent)
        self.t = {a: np.array([r["t"] for r in rows]) for a, rows in by_agent.items()}
        self.xy = {a: np.array([(r["x"], r["y"]) for r in rows]) for a, rows in by_agent.items()}

    def at(self, agent: str, t) -> np.ndarray:
        """Where the robot was at t (or each of an array of times), between track rows."""
        times, xy = self.t[agent], self.xy[agent]
        return np.stack([np.interp(t, times, xy[:, 0]), np.interp(t, times, xy[:, 1])], axis=-1)

    def length(self, agent: str, start: float, end: float) -> float:
        """How far the robot drove from start to end, along its track."""
        if end <= start:
            return 0.0
        points = self.at(agent, np.append(np.arange(start, end, PATH_STEP_S), end))
        return float(np.sum(np.hypot(*np.diff(points, axis=0).T)))

    def speeds(self) -> dict[str, float]:
        """Each robot's average speed (m/s) while driving, from its velocities if it has them,
        otherwise from its positions."""
        found = {}
        for agent, rows in self.rows.items():
            if all(r["vx"] is not None and r["vy"] is not None for r in rows):
                speed = np.hypot([r["vx"] for r in rows], [r["vy"] for r in rows])
            else:
                step = np.diff(self.t[agent])
                keep = step > 0
                speed = np.hypot(*np.diff(self.xy[agent], axis=0)[keep].T) / step[keep]
            moving = speed[speed > MOVING_MPS]
            if len(moving) >= 3:
                found[agent] = float(np.mean(moving))
        return found


def _removals(episode: EpisodeReader) -> dict[str, list[float]]:
    found: dict[str, list[float]] = {}
    if episode.events is not None:
        for row in episode.events.to_pylist():
            if row["type"] == "remove" and row["object_id"]:
                found.setdefault(row["object_id"], []).append(row["t"])
    return found


def _defended(paths: _Paths, roster, agent: str, t: float, radius: float) -> bool:
    if agent not in roster:
        return False
    me = paths.at(agent, t)
    for other, entry in roster.items():
        if entry.alliance == roster[agent].alliance or other not in paths.agents:
            continue
        if np.hypot(*(paths.at(other, t) - me)) <= radius:
            return True
    return False


def _fail_rates(placed: list[tuple[bool, bool]], min_samples: int) -> tuple[float, float]:
    """The failure rate without and with defense, each falling back to the overall rate."""
    overall = float(np.mean([failed for failed, _ in placed]))
    rates = []
    for defended in (False, True):
        group = [failed for failed, d in placed if d == defended]
        rates.append(float(np.mean(group)) if len(group) >= min_samples else overall)
    return rates[0], rates[1]
