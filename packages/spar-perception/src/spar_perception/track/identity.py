"""Closed-set identity from the match roster, using motion and, when robot sizes differ, size.

Each roster robot is at most one track at a time, and a robot that drops out of view comes back as
a new track. assign_identities() takes the tracks in the order they start and gives each one to a
roster robot of its alliance that is free at that moment: the one whose last known position it
could have driven from, and, when the roster gives sizes, whose size matches the track's footprint
width. Tracks that start in the same frame are assigned together.

In V5RC both robots of an alliance are 18 in or smaller, so partners that start together cannot be
told apart, and they get confidence 0.5. Swapping partners never changes the score, so the score
check cannot catch it either.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import groupby

import numpy as np
from scipy.optimize import linear_sum_assignment

from spar_perception.calib.homography import image_to_field
from spar_perception.track.bytetrack import Track


@dataclass(frozen=True)
class RosterRobot:
    agent_id: str  # for example "red_1"
    alliance: str
    size_m: float | None = None  # robot width when known, for example 15 in or 24 in in VEX U


@dataclass(frozen=True)
class Assignment:
    agent_id: str | None  # None when no roster robot fits, for example a false detection
    confidence: float  # 0 to 1


@dataclass(frozen=True)
class IdentityConfig:
    max_speed_mps: float = 3.0  # faster than robots drive, so it only rules out the impossible
    slack_m: float = 0.3  # position noise allowed on top of max speed
    size_weight: float = 1.0


@dataclass
class _Robot:
    last_t: float | None = None
    last_xy: np.ndarray | None = None
    busy_until: float = -np.inf
    tracks: list[int] = field(default_factory=list)


def footprint(H: np.ndarray, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The field point under each box's bottom middle (n, 2), and the bottom edge's width (n,) in m.

    From a raised camera the bottom middle is the robot's near edge, not its center, so these
    positions sit about half a robot toward the camera.
    """
    b = np.asarray(boxes, dtype=float).reshape(-1, 4)
    left = image_to_field(H, np.c_[b[:, 0], b[:, 3]])
    right = image_to_field(H, np.c_[b[:, 2], b[:, 3]])
    middle = image_to_field(H, np.c_[(b[:, 0] + b[:, 2]) / 2, b[:, 3]])
    return middle, np.linalg.norm(right - left, axis=1)


def assign_identities(
    tracks: dict[int, Track],
    positions: dict[int, np.ndarray],
    roster: list[RosterRobot],
    *,
    widths: dict[int, np.ndarray] | None = None,
    config: IdentityConfig | None = None,
) -> dict[int, Assignment]:
    """Give each track to a roster robot. positions holds each track's field points (n, 2)."""
    config = config or IdentityConfig()
    widths = widths or {}
    result: dict[int, Assignment] = {}
    by_alliance: dict[str, list[int]] = defaultdict(list)
    for track_id, track in tracks.items():
        alliance = track.majority_alliance()
        if alliance is None or not any(robot.alliance == alliance for robot in roster):
            result[track_id] = Assignment(None, 0.0)
        else:
            by_alliance[alliance].append(track_id)

    for alliance, track_ids in by_alliance.items():
        robots = [robot for robot in roster if robot.alliance == alliance]
        state = {robot.agent_id: _Robot() for robot in robots}
        track_ids.sort(key=lambda track_id: (tracks[track_id].start, track_id))
        for _, starting in groupby(track_ids, key=lambda track_id: tracks[track_id].start):
            group = list(starting)
            cost = np.array(
                [
                    [
                        _cost(
                            tracks[tid],
                            positions[tid],
                            widths.get(tid),
                            robot,
                            state[robot.agent_id],
                            config,
                        )
                        for robot in robots
                    ]
                    for tid in group
                ]
            )
            # Extra columns let a track go unassigned when no robot is free for it.
            padded = np.hstack(
                [np.where(np.isinf(cost), 1e6, cost), np.full((len(group), len(group)), 1e5)]
            )
            for row, col in zip(*linear_sum_assignment(padded), strict=True):
                track_id = group[row]
                if col >= len(robots) or np.isinf(cost[row, col]):
                    result[track_id] = Assignment(None, 0.0)
                    continue
                robot, track = robots[col], tracks[track_id]
                result[track_id] = Assignment(robot.agent_id, _confidence(cost[row], col))
                busy = state[robot.agent_id]
                busy.last_t, busy.last_xy, busy.busy_until = (
                    track.end,
                    positions[track_id][-1],
                    track.end,
                )
                busy.tracks.append(track_id)
    return result


def tracks_by_agent(assignments: dict[int, Assignment]) -> dict[str, list[int]]:
    """The track IDs given to each roster robot."""
    found: dict[str, list[int]] = defaultdict(list)
    for track_id, assignment in sorted(assignments.items()):
        if assignment.agent_id is not None:
            found[assignment.agent_id].append(track_id)
    return dict(found)


def _cost(track, xy, width, robot: RosterRobot, state: _Robot, config: IdentityConfig) -> float:
    if track.start <= state.busy_until:
        return np.inf  # the robot is already a track that is still going
    if state.last_t is None:
        cost = 0.5  # a robot not seen yet
    else:
        reach = config.max_speed_mps * (track.start - state.last_t) + config.slack_m
        distance = float(np.linalg.norm(xy[0] - state.last_xy))
        if distance > reach:
            return np.inf
        cost = distance / reach  # 0 where it vanished, 1 at the edge of what it could drive
    if robot.size_m is not None and width is not None and len(width):
        cost += config.size_weight * abs(float(np.median(width)) - robot.size_m) / robot.size_m
    return cost


def _confidence(costs: np.ndarray, chosen: int, scale: float = 0.2) -> float:
    """0.5 when another robot fits as well, rising toward 1 as the gap to the next best grows."""
    others = np.delete(costs, chosen)
    others = others[np.isfinite(others)]
    if not len(others):
        return 1.0  # the only robot it could be
    gap = float(others.min()) - float(costs[chosen])
    return float(np.clip(0.5 + 0.5 * np.tanh(gap / scale), 0.0, 1.0))
