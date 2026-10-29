"""Field calibration on a synthetic livestream camera whose true homography is known."""

import numpy as np
import pytest

from spar_perception.calib.homography import (
    field_to_image,
    fit_homography,
    floor_landmarks,
    image_to_field,
    point_covariance,
    smooth_homographies,
)

RNG = np.random.default_rng(7)

# Points on the floor of a 3.57 m field, in meters: corners, a centered square and a few in between.
FLOOR_POINTS = np.array(
    [
        [-1.78, -1.78], [1.78, -1.78], [1.78, 1.78], [-1.78, 1.78],
        [-0.6, 0.0], [0.0, -0.6], [0.6, 0.0], [0.0, 0.6],
        [-1.2, 0.6], [1.2, -0.6], [0.6, 1.2], [-0.6, -1.2],
    ]
)  # fmt: skip


def camera_field_to_image(position, target, focal_px=1400.0, size=(1920, 1080)):
    """The field-to-image homography of a pinhole camera at `position` looking at `target` (m)."""
    position, target = np.asarray(position, float), np.asarray(target, float)
    forward = (target - position) / np.linalg.norm(target - position)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    R = np.vstack([right, down, forward])  # world to camera
    t = -R @ position
    K = np.array([[focal_px, 0, size[0] / 2], [0, focal_px, size[1] / 2], [0, 0, 1]])
    G = K @ np.column_stack([R[:, 0], R[:, 1], t])  # (x, y, 1) on the floor to pixels
    return G / G[2, 2]


def project(G, points):
    q = np.c_[points, np.ones(len(points))] @ G.T
    return q[:, :2] / q[:, 2:3]


@pytest.fixture
def camera():
    """A raised camera behind the audience side, like a livestream: returns (G, H_true)."""
    G = camera_field_to_image(position=[0.3, -4.0, 2.6], target=[0.0, 0.2, 0.0])
    H = np.linalg.inv(G)
    return G, H / H[2, 2]


def field_error(H, H_true, points=FLOOR_POINTS):
    """Largest distance on the floor, in meters, between where H and H_true put the same pixels."""
    pixels = field_to_image(H_true, points)
    return float(np.max(np.linalg.norm(image_to_field(H, pixels) - points, axis=1)))


def test_floor_landmarks_keeps_points_at_z_zero():
    landmarks = {"corner": [1.0, 1.0, 0.0], "corner_top": [1.0, 1.0, 0.29], "center": [0, 0, 0]}
    assert floor_landmarks(landmarks) == {"corner": (1.0, 1.0), "center": (0.0, 0.0)}


def test_fit_recovers_the_camera(camera):
    G, H_true = camera
    pixels = project(G, FLOOR_POINTS) + RNG.normal(0, 0.5, (len(FLOOR_POINTS), 2))
    fit = fit_homography(pixels, FLOOR_POINTS)
    assert fit.inliers.all()
    assert fit.rms_px < 1.5
    assert field_error(fit.H, H_true) < 0.02


def test_ransac_drops_a_mislabeled_landmark(camera):
    G, H_true = camera
    pixels = project(G, FLOOR_POINTS)
    pixels[3] += [80.0, -45.0]  # a landmark clicked in the wrong place
    fit = fit_homography(pixels, FLOOR_POINTS)
    assert not fit.inliers[3]
    assert fit.inliers.sum() == len(FLOOR_POINTS) - 1
    assert field_error(fit.H, H_true) < 0.005


def test_fit_needs_four_pairs(camera):
    G, _ = camera
    with pytest.raises(ValueError, match="at least 4"):
        fit_homography(project(G, FLOOR_POINTS[:3]), FLOOR_POINTS[:3])


def test_mappings_are_inverses(camera):
    _, H = camera
    pixels = np.array([[100.0, 900.0], [960.0, 540.0], [1800.0, 300.0]])
    np.testing.assert_allclose(field_to_image(H, image_to_field(H, pixels)), pixels, atol=1e-6)


def test_point_covariance_matches_sampling(camera):
    _, H = camera
    pixel = np.array([[1500.0, 380.0]])  # toward the far side, where one pixel covers more floor
    predicted = point_covariance(H, pixel, pixel_sigma=1.0)[0]
    samples = pixel + RNG.normal(0, 1.0, (40000, 2))
    measured = np.cov(image_to_field(H, samples).T)
    np.testing.assert_allclose(predicted, measured, rtol=0.05, atol=1e-9)


def test_far_points_are_less_certain(camera):
    G, H = camera
    near, far = project(G, np.array([[0.0, -1.5], [0.0, 1.5]]))
    cov = point_covariance(H, np.array([near, far]), pixel_sigma=1.0)
    assert np.trace(cov[1]) > 2 * np.trace(cov[0])


def test_smoothing_beats_per_frame_fits_and_fills_gaps(camera):
    G, H_true = camera
    raw = []
    for frame in range(60):
        if frame in (10, 11, 40):
            raw.append(None)  # frames where too few landmarks were seen
            continue
        pixels = project(G, FLOOR_POINTS) + RNG.normal(0, 1.5, (len(FLOOR_POINTS), 2))
        raw.append(fit_homography(pixels, FLOOR_POINTS).H)
    anchors = FLOOR_POINTS[:4]
    smoothed = smooth_homographies(raw, anchors, window=15)
    assert len(smoothed) == len(raw)
    raw_error = np.median([field_error(H, H_true) for H in raw if H is not None])
    smooth_error = np.median([field_error(H, H_true) for H in smoothed])
    assert smooth_error < 0.6 * raw_error
    assert field_error(smoothed[10], H_true) < 0.02


def test_pixels_to_tracks_end_to_end(camera, tmp_path):
    """A robot seen by the camera: fit, map pixels to the field, smooth, write tracks.parquet."""
    import pyarrow.parquet as pq

    from spar_perception.filter.kalman import smooth_track, to_tracks_table
    from spar_schema import TRACKS_SCHEMA, read_table
    from spar_schema.version import file_metadata

    G, _ = camera
    fit = fit_homography(
        project(G, FLOOR_POINTS) + RNG.normal(0, 0.5, (len(FLOOR_POINTS), 2)), FLOOR_POINTS
    )
    t = np.arange(0, 5, 1 / 30)
    truth = np.c_[1.2 * np.cos(0.6 * t), 0.9 * np.sin(0.6 * t)]  # an arc across the field
    heading_truth = np.arctan2(0.9 * np.cos(0.6 * t), -1.2 * np.sin(0.6 * t))
    pixels = project(G, truth) + RNG.normal(0, 1.0, truth.shape)
    xy = image_to_field(fit.H, pixels)
    xy_cov = point_covariance(fit.H, pixels, pixel_sigma=1.0)
    xy[60:75] = np.nan  # hidden behind another robot for half a second
    heading = heading_truth + RNG.normal(0, 0.05, len(t))

    track = smooth_track(t, xy, xy_cov, heading=heading, heading_var=0.05**2)
    error = np.linalg.norm(track.xy - truth, axis=1)
    assert np.sqrt(np.mean(error**2)) < 0.02
    assert error[60:75].max() < 0.05

    table = to_tracks_table("blue_2", np.arange(len(t)), track)
    path = tmp_path / "tracks.parquet"
    pq.write_table(table.replace_schema_metadata(file_metadata("toy", "v1")), path)
    assert read_table(path, TRACKS_SCHEMA).num_rows == len(t)


def test_smoothing_needs_a_fit_somewhere():
    with pytest.raises(ValueError, match="no frame"):
        smooth_homographies([None, None], FLOOR_POINTS[:4])
