"""Strategy codes: checking, distances, the coder threshold and kappa, with hand-worked values."""

import math
from pathlib import Path

import pytest

from spar_schema import load_scene
from spar_schema.scene import StrategyCodeSpec
from spar_state.strategy.agreement import category, cohens_kappa, field_kappas
from spar_state.strategy.distance import code_distance, field_distances
from spar_state.strategy.extract import check_code, code_version
from spar_state.strategy.threshold import coder_threshold

SPEC = StrategyCodeSpec.model_validate(
    {
        "fields": {
            "route": {"type": "sequence", "distance": "edit", "weight": 1.0},
            "orders": {"type": "per_robot_sequence", "distance": "edit", "weight": 1.0},
            "shares": {"type": "vector", "length": 4, "distance": "l1", "weight": 1.0},
            "split": {
                "type": "per_robot_simplex",
                "parts": ["scoring", "loading", "defense", "toggles"],
                "distance": "l1",
                "weight": 2.0,
            },
            "entry": {
                "type": "per_robot_time",
                "distance": "abs_seconds",
                "missing_penalty": "TODO",
                "weight": 1.0,
            },
        },
        "threshold": {"method": "coder_p95", "matches": 50},
    }
)
MATCH_S = 120.0
CODE = {
    "route": ["R1", "R3"],
    "orders": [["R1"], ["R3"]],
    "shares": [0.9, 0.0, 0.0, 0.0],
    "split": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
    "entry": [94.0, None],
}


def test_identical_codes_are_zero_apart():
    assert code_distance(CODE, CODE, SPEC, MATCH_S) == 0.0


def test_field_distances_by_hand():
    other = {
        "route": ["R1", "B2"],  # one edit in two
        "orders": [["R3"], ["R1"]],  # the same robots, listed the other way round
        "shares": [0.5, 0.0, 0.0, 0.0],  # 0.4 apart in one of four entries
        "split": [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],  # nothing in common
        "entry": [100.0, None],  # 6 s apart in a 120 s match, for one of two robots
    }
    assert field_distances(CODE, other, SPEC, MATCH_S) == pytest.approx(
        {"route": 0.5, "orders": 0.0, "shares": 0.1, "split": 1.0, "entry": 0.025}
    )
    # Weighted sum: split counts twice.
    assert code_distance(CODE, other, SPEC, MATCH_S) == pytest.approx(0.5 + 0.1 + 2 * 1.0 + 0.025)


def test_a_missing_time_costs_the_penalty_and_todo_counts_as_one():
    other = {**CODE, "entry": [None, None]}
    assert field_distances(CODE, other, SPEC, MATCH_S)["entry"] == pytest.approx(0.5)


def test_a_missing_robot_counts_as_empty():
    other = {**CODE, "orders": [["R1"]], "split": [[1.0, 0.0, 0.0, 0.0]]}
    distances = field_distances(CODE, other, SPEC, MATCH_S)
    assert distances["orders"] == pytest.approx(0.5)  # ["R3"] against nothing, for one of two
    # [0, 1, 0, 0] against even shares is 1.5 / 2 = 0.75 apart, for one of two robots.
    assert distances["split"] == pytest.approx(0.375)


def test_coder_threshold_is_the_95th_percentile():
    pairs = []
    for k in range(1, 21):
        other = {**CODE, "shares": [0.9 - 0.04 * k, 0.0, 0.0, 0.0]}  # 0.01 * k apart
        pairs.append((CODE, other))
    expected = 0.01 * (19 + 0.05 * (20 - 19))  # numpy's linear interpolation at 95%
    assert coder_threshold(pairs, SPEC, MATCH_S) == pytest.approx(expected)
    with pytest.raises(ValueError):
        coder_threshold([], SPEC, MATCH_S)


def test_cohens_kappa_textbook_example():
    a = [1, 1, 0, 0, 1, 0, 1, 1, 0, 1]
    b = [1, 0, 0, 0, 1, 1, 1, 1, 0, 1]
    assert cohens_kappa(a, b) == pytest.approx((0.8 - 0.52) / 0.48)
    assert cohens_kappa(a, a) == pytest.approx(1.0)
    assert math.isnan(cohens_kappa([1, 1, 1], [1, 1, 1]))


def test_field_kappas_ignore_robot_order():
    swapped = {**CODE, "orders": [["R3"], ["R1"]], "entry": [None, 94.0]}
    other = {**CODE, "route": ["B2"], "orders": [["B2"], []], "entry": [None, None]}
    kappas = field_kappas([CODE, other], [swapped, other], SPEC)
    assert kappas["orders"] == pytest.approx(1.0)
    assert kappas["entry"] == pytest.approx(1.0)
    assert category(CODE["split"], SPEC.fields["split"]) == ("loading", "scoring")


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda c: c.pop("entry"), "fields"),
        (lambda c: c.update(shares=[0.1, 0.2]), "4 numbers"),
        (lambda c: c.update(split=[[0.5, 0.2, 0.0, 0.0]]), "add to 1"),
        (lambda c: c.update(route="R1"), "list of IDs"),
    ],
    ids=["missing field", "wrong vector length", "shares not adding to 1", "not a list"],
)
def test_check_code_rejects_bad_codes(change, message):
    code = {k: (list(v) if isinstance(v, list) else v) for k, v in CODE.items()}
    change(code)
    with pytest.raises(ValueError, match=message):
        check_code(code, SPEC)


def test_check_code_accepts_a_good_code():
    check_code(CODE, SPEC)


def test_code_version_changes_with_the_definition(tmp_path):
    toy = Path(__file__).parents[2] / "spar-schema" / "tests" / "fixtures" / "toy_scene"
    first = code_version(load_scene(toy))
    assert first == code_version(load_scene(toy)) and first.startswith("toy-")
    copy = tmp_path / "toy"
    copy.mkdir()
    for path in toy.iterdir():
        if path.is_file():
            (copy / path.name).write_bytes(path.read_bytes())
    code = copy / "strategy_code.yaml"
    code.write_text(code.read_text().replace("weight: 1.0", "weight: 2.0"))
    assert code_version(load_scene(copy)) != first
