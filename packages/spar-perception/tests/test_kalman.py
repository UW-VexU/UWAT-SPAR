"""Kalman filter and RTS smoother on synthetic robot tracks with known truth."""

import numpy as np
import pytest

from spar_perception.filter.kalman import _cv_smooth, smooth_track, to_tracks_table
from spar_schema import TRACKS_SCHEMA
from spar_schema.frames import wrap_angle

RNG = np.random.default_rng(11)
FPS = 30
T = np.arange(0, 4, 1 / FPS)


def straight_line(sigma=0.03):
    truth = np.c_[-1.0 + 0.5 * T, 0.2 * T]
    measured = truth + RNG.normal(0, sigma, truth.shape)
    cov = np.broadcast_to(sigma**2 * np.eye(2), (len(T), 2, 2)).copy()
    return truth, measured, cov


def rms(error):
    return float(np.sqrt(np.mean(np.sum(np.atleast_2d(error.T).T ** 2, axis=-1))))


def test_smoothing_reduces_position_error():
    truth, measured, cov = straight_line()
    track = smooth_track(T, measured, cov)
    assert rms(track.xy - truth) < 0.5 * rms(measured - truth)
    np.testing.assert_allclose(track.velocity.mean(axis=0), [0.5, 0.2], atol=0.05)


def test_smoother_beats_the_forward_filter():
    truth, measured, cov = straight_line()
    smooth, _, forward, _ = _cv_smooth(T, measured, cov, accel_std=2.0, initial_rate_std=2.0)
    assert rms(smooth[:, :2] - truth) < rms(forward[:, :2] - truth)


def test_gap_is_filled_with_wider_covariance():
    truth, measured, cov = straight_line()
    gap = (T > 1.5) & (T < 2.5)
    measured[gap] = np.nan
    track = smooth_track(T, measured, cov)
    assert not track.measured[gap].any() and track.measured[~gap].all()
    assert np.max(np.linalg.norm(track.xy[gap] - truth[gap], axis=1)) < 0.1
    middle = np.flatnonzero(gap)[gap.sum() // 2]
    assert np.trace(track.cov[middle, :2, :2]) > 5 * np.trace(track.cov[5, :2, :2])


def test_heading_is_smoothed_across_the_wrap():
    truth = wrap_angle(np.pi - 0.5 + 0.5 * T)  # turns through +-pi at t = 1 s
    measured = wrap_angle(truth + RNG.normal(0, 0.05, len(T)))
    _, xy, cov = straight_line()
    track = smooth_track(T, xy, cov, heading=measured, heading_var=0.05**2)
    assert np.all((track.heading >= -np.pi) & (track.heading < np.pi))
    assert np.max(np.abs(wrap_angle(track.heading - truth))) < 0.1
    assert np.mean(track.omega) == pytest.approx(0.5, abs=0.1)


def test_no_heading_leaves_heading_out():
    _, xy, cov = straight_line()
    track = smooth_track(T, xy, cov)
    assert track.heading is None and track.omega is None
    assert np.isnan(track.cov[:, 2, 2]).all()


def test_time_must_increase():
    _, xy, cov = straight_line()
    with pytest.raises(ValueError, match="increasing"):
        smooth_track(T[::-1], xy, cov)


def test_tracks_table_is_valid_and_marks_interpolated_frames():
    _, xy, cov = straight_line()
    xy[30:45] = np.nan
    heading = np.full(len(T), 0.3)
    track = smooth_track(T, xy, cov, heading=heading, heading_var=0.01)
    table = to_tracks_table("red_1", np.arange(len(T)), track)
    assert table.schema.equals(TRACKS_SCHEMA)
    provenance = table.column("provenance").to_pylist()
    assert provenance[30:45] == ["interpolated"] * 15
    assert provenance[:30] == ["detected"] * 30
    assert len(table.column("cov")[0].as_py()) == 9


def test_tracks_table_needs_heading():
    _, xy, cov = straight_line()
    with pytest.raises(ValueError, match="heading"):
        to_tracks_table("red_1", np.arange(len(T)), smooth_track(T, xy, cov))
