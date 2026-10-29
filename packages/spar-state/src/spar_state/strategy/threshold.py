"""Coder threshold: 95th percentile of distance between two human coders on the same match.

Two strategy codes count as different strategies only when they are farther apart than two people
coding the same match tend to be (spec section 5.4). The threshold is measured on the coder study's
matches and sealed with the candidate list.
"""

from typing import Any

import numpy as np

from spar_schema.scene import StrategyCodeSpec
from spar_state.strategy.distance import code_distance


def coder_threshold(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    spec: StrategyCodeSpec,
    match_length_s: float,
    percentile: float = 95.0,
) -> float:
    """The percentile of code distances between two coders' codes for the same alliances."""
    if not pairs:
        raise ValueError("the coder threshold needs at least one pair of codes")
    distances = [code_distance(a, b, spec, match_length_s) for a, b in pairs]
    return float(np.percentile(distances, percentile))
