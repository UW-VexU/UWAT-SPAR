"""Runs the scene's score() at each checkpoint and compares it with the official result.

A checkpoint scored after the autonomous period is compared by winner, because only the winner is
published there (read from the stream overlay). One scored after the driver period is compared
with the official final points (from the Events API). Checkpoint times past the end of the
episode's data are moved back to the last row.

The check is necessary but not sufficient: errors can cancel inside a total, and swapping two
partner robots never changes it. A pass means the end state is plausible, not that every track is
right (spec section 6.4).
"""

from dataclasses import dataclass

import numpy as np
import pyarrow as pa

from spar_schema import EpisodeReader, Outcome, Scene, ScoreCheck
from spar_state.state import data_end, snapshot_at, states_table


@dataclass(frozen=True)
class CheckResult:
    checks: dict[str, ScoreCheck]  # by checkpoint
    states: pa.Table  # state.parquet rows, one per checkpoint


def check_scores(episode: EpisodeReader, scene: Scene, *, tolerance: float = 0.0) -> CheckResult:
    """Score the episode's state at each checkpoint and compare it with outcome.json.

    tolerance is how many points each alliance's final score may differ by and still pass.
    """
    outcome = episode.outcome
    if outcome is None:
        raise ValueError(f"{episode.path} has no outcome.json to check against")
    end = data_end(episode)
    checks, rows = {}, []
    for checkpoint in scene.checkpoints:
        t = min(scene.checkpoint_time(checkpoint, episode.meta.program), end)
        snapshot = snapshot_at(
            episode, t, checkpoint=checkpoint, autonomous_winner=outcome.autonomous_winner
        )
        state = scene.build_state(snapshot)
        scene.task_state.check(state)
        points = {alliance: float(value) for alliance, value in scene.score(state, np).items()}
        if scene.spec.objective.checkpoint_at[checkpoint].after == "autonomous":
            checks[checkpoint] = _compare_winner(points, outcome.autonomous_winner)
        else:
            checks[checkpoint] = _compare_points(points, outcome, tolerance)
        rows.append((t, checkpoint, points, scene.task_state.to_json(state)))
    return CheckResult(checks=checks, states=states_table(rows, scene))


def _compare_winner(points: dict[str, float], official: str) -> ScoreCheck:
    red, blue = points["red"], points["blue"]
    computed = "red" if red > blue else "blue" if blue > red else "tie"
    # "none" means both alliances committed Violations, which the video cannot show.
    passed = None if official == "none" else computed == official
    return ScoreCheck(
        computed=points, computed_winner=computed, official_winner=official, passed=passed
    )


def _compare_points(points: dict[str, float], outcome: Outcome, tolerance: float) -> ScoreCheck:
    official = {"red": float(outcome.points_red), "blue": float(outcome.points_blue)}
    difference = {alliance: points[alliance] - official[alliance] for alliance in official}
    passed = all(abs(value) <= tolerance for value in difference.values())
    return ScoreCheck(computed=points, official=official, difference=difference, passed=passed)
