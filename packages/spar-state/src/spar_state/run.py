"""What `spar state run` does to one episode: score check, events and strategy codes.

run_episode() reads the episode's tracks, objects and outcome, then writes back:
    state.parquet    the task state and points at each checkpoint
    events.parquet   events from the generic rules and the scene's events plugin
    strategy.json    each alliance's strategy code, when the scene has a strategy plugin
    episode.json     the score check per checkpoint
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spar_schema import EpisodeReader, EpisodeWriter, Scene, ScoreCheck
from spar_state.events.rules import events_table, extract_events
from spar_state.score_check import check_scores
from spar_state.strategy.extract import extract_codes, strategy_record


@dataclass(frozen=True)
class RunResult:
    checks: dict[str, ScoreCheck]  # by checkpoint
    events: int  # how many events were written
    codes: dict[str, dict[str, Any]] | None  # by alliance, or None without a strategy plugin


def run_episode(
    path: str | Path,
    scene: Scene,
    *,
    tolerance: float = 0.0,
    allow_incomplete_scene: bool = False,
) -> RunResult:
    episode = EpisodeReader(path)
    result = check_scores(episode, scene, tolerance=tolerance)
    events = extract_events(episode, scene)
    codes = None
    if scene.has_plugin("strategy") and scene.strategy_code is not None:
        codes = extract_codes(episode, events, scene)
    meta = episode.meta.model_copy(update={"score_check": result.checks})
    with EpisodeWriter(
        path, scene, meta, exist_ok=True, allow_incomplete_scene=allow_incomplete_scene
    ) as writer:
        writer.write_state(result.states)
        writer.write_events(events_table(events))
        if codes is not None:
            writer.write_strategy(strategy_record(codes, scene))
    return RunResult(checks=result.checks, events=len(events), codes=codes)
