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
