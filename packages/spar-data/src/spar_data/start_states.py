"""Samples moments from real episodes into start_states.parquet.

sample_start_states() picks episodes and times at random, from a seed, and records each robot's
pose and velocity and every object's state at that moment: the latest track and object row at or
before it (spar_schema.EpisodeMoments). Simulators start from these moments: spar-sim, and the
physics repo through env.reset_from(). Times are drawn from the part of each episode that has
tracks, and from the match itself, leaving at least min_time_left_s on the clock.
"""

from collections.abc import Iterable

import numpy as np

from spar_data.catalog import Entry
from spar_schema import EpisodeMoments, EpisodeReader, Scene, StartState


def sample_start_states(
    entries: Iterable[Entry],
    scene: Scene,
    n: int,
    *,
    seed: int = 0,
    min_time_left_s: float = 1.0,
) -> list[StartState]:
    """n moments, drawn at random (seeded) from the given episodes."""
    episodes = []
    for entry in entries:
        reader = EpisodeReader(entry.path)
        if reader.tracks is None or reader.tracks.num_rows == 0:
            continue
        periods = scene.programs[reader.meta.program]
        match_length = float(periods.autonomous_s + periods.driver_s)
        times = reader.tracks.column("t").to_numpy()
        latest = min(float(times.max()), match_length - min_time_left_s)
        earliest = max(float(times.min()), 0.0)
        if latest > earliest:
            episodes.append((reader, earliest, latest))
    if not episodes:
        raise ValueError("no episode has tracks inside its match to sample from")

    rng = np.random.default_rng(seed)
    moments: dict[int, EpisodeMoments] = {}
    states = []
    for _ in range(n):
        k = int(rng.integers(len(episodes)))
        reader, earliest, latest = episodes[k]
        if k not in moments:
            moments[k] = EpisodeMoments(reader, scene)
        states.append(moments[k].at(float(rng.uniform(earliest, latest))))
    return states
