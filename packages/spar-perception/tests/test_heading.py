"""Heading from mask orientation, disambiguated by motion."""

import math

import numpy as np
import pytest

from spar_perception.heading import (
    axis_to_field,
    estimate_heading,
    heading_measurements,
    mask_axis,
)
from spar_schema.frames import wrap_angle

PX_PER_M = 200.0
# Image to field for a camera straight above the field center: u right, v down.
H = np.array([[1 / PX_PER_M, 0, -960 / PX_PER_M], [0, -1 / PX_PER_M, 540 / PX_PER_M], [0, 0, 1]])


def rectangle_mask(angle, length=60, width=20, size=101):
    """A filled rectangle rotated by `angle` in image coordinates, centered in a square image."""
    v, u = np.mgrid[0:size, 0:size] - size // 2
    along = u * math.cos(angle) + v * math.sin(angle)
    across = -u * math.sin(angle) + v * math.cos(angle)
    return (np.abs(along) <= length / 2) & (np.abs(across) <= width / 2)


@pytest.mark.parametrize("degrees", [0, 30, 60, 90, 135, 170])
def test_mask_axis_finds_the_long_axis(degrees):
    centroid, angle, elongation = mask_axis(
        rectangle_mask(math.radians(degrees)), origin=(100, 200)
    )
    assert abs(wrap_angle(2 * (angle - math.radians(degrees)))) / 2 < math.radians(2)
    assert elongation > 2
    np.testing.assert_allclose(centroid, [150, 250], atol=0.6)


def test_square_mask_has_no_long_axis():
    _, _, elongation = mask_axis(rectangle_mask(0.3, length=40, width=40))
    assert elongation < 1.3


def test_empty_mask():
    assert mask_axis(np.zeros((5, 5), dtype=bool)) is None


@pytest.mark.parametrize("image_degrees", [0, 30, 90, 150])
def test_axis_maps_onto_the_field(image_degrees):
    # With v pointing down, an image angle a is the field angle -a.
    field = axis_to_field(H, np.array([960.0, 540.0]), math.radians(image_degrees))
    assert abs(wrap_angle(2 * (field + math.radians(image_degrees)))) / 2 < 1e-6


def test_motion_picks_which_end_is_the_front():
    velocity = np.array([[0.5, 0.0], [-0.5, 0.0], [0.0, 0.0], [0.0, 0.0]])
    axis = np.array([0.0, 0.0, 0.1, np.nan])
    heading, variance = heading_measurements(velocity, axis)
    assert heading[0] == pytest.approx(0.0)
    assert abs(wrap_angle(heading[1] - math.pi)) < 1e-9  # same axis, driving the other way
    # Stopped: the end of the axis nearest the last heading.
    assert abs(wrap_angle(heading[2] - math.pi - 0.1)) < 1e-9
    assert math.isnan(heading[3])  # stopped and no axis: nothing to go on
    assert variance[0] == pytest.approx(0.15**2)


def test_square_robot_moving_uses_its_direction_of_travel():
    heading, variance = heading_measurements(np.array([[0.0, 0.4]]), np.array([np.nan]))
    assert heading[0] == pytest.approx(math.pi / 2)
    assert variance[0] == pytest.approx(0.35**2)


def test_still_robot_without_history_has_no_heading():
    heading, _ = heading_measurements(np.zeros((3, 2)), np.array([0.5, 0.5, 0.5]))
    assert np.isnan(heading).all()


def test_estimate_heading_on_an_arc():
    rng = np.random.default_rng(5)
    t = np.arange(0, 6, 1 / 30)
    xy_true = np.c_[np.cos(0.5 * t), np.sin(0.5 * t)]
    heading_true = wrap_angle(0.5 * t + math.pi / 2)  # driving counterclockwise, facing forward
    xy = xy_true + rng.normal(0, 0.01, xy_true.shape)
    cov = np.broadcast_to(1e-4 * np.eye(2), (len(t), 2, 2)).copy()
    axis = (heading_true + rng.normal(0, 0.08, len(t))) % math.pi  # the mask's axis, front unknown
    track = estimate_heading(t, xy, cov, axis)
    error = np.degrees(np.abs(wrap_angle(track.heading - heading_true)))
    assert np.median(error) < 3
    assert error.max() < 10


def test_estimate_heading_without_any_cue_returns_no_heading():
    t = np.arange(0, 1, 1 / 30)
    still = np.zeros((len(t), 2))
    cov = np.broadcast_to(1e-4 * np.eye(2), (len(t), 2, 2)).copy()
    assert estimate_heading(t, still, cov, np.full(len(t), np.nan)).heading is None
