"""Schema version checks and unit helpers."""

import math

import pytest

from spar_schema import SCHEMA_VERSION, SchemaVersionError, check_compatible
from spar_schema.frames import feet, inches, wrap_angle
from spar_schema.version import parse_version


def test_parse_version():
    assert parse_version("0.1.0") == (0, 1, 0)
    for bad in ["", "0.1", "0.1.0.1", "a.b.c", "v0.1.0"]:
        with pytest.raises(ValueError):
            parse_version(bad)


def test_same_major_is_compatible():
    check_compatible(SCHEMA_VERSION)
    check_compatible("0.99.0")


def test_unknown_major_is_refused():
    with pytest.raises(SchemaVersionError):
        check_compatible("1.0.0")


def test_units():
    assert math.isclose(feet(12), 3.6576)
    assert math.isclose(inches(15), 0.381)


@pytest.mark.parametrize(
    "theta, wrapped",
    [
        (0.0, 0.0),
        (1.5 * math.pi, -0.5 * math.pi),
        (-1.5 * math.pi, 0.5 * math.pi),
        (math.pi, -math.pi),
    ],
)
def test_wrap_angle(theta, wrapped):
    assert math.isclose(wrap_angle(theta), wrapped, abs_tol=1e-12)


SQUARE = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]


@pytest.mark.parametrize(
    "point, expected",
    [
        ((0.0, 0.0), -1.0),
        ((0.5, 0.0), -0.5),
        ((1.0, 0.3), 0.0),
        ((3.0, 0.0), 2.0),
        ((4.0, 5.0), 5.0),
    ],
    ids=["center", "inside", "on an edge", "outside", "past a corner"],
)
def test_polygon_distance(point, expected):
    from spar_schema.frames import polygon_distance

    assert math.isclose(polygon_distance(*point, SQUARE), expected, abs_tol=1e-12)


def test_polygon_distances_match_one_at_a_time():
    import numpy as np

    from spar_schema.frames import polygon_distance, polygon_distances

    diamond = [[-0.6, 0.0], [0.0, -0.6], [0.6, 0.0], [0.0, 0.6]]
    points = np.random.default_rng(0).uniform(-1.5, 1.5, size=(50, 2))
    for polygon in (SQUARE, diamond):
        expected = [polygon_distance(x, y, polygon) for x, y in points]
        assert np.allclose(polygon_distances(points, polygon, np), expected, atol=1e-5)
    assert polygon_distances(points.reshape(5, 10, 2), SQUARE, np).shape == (5, 10)
