"""Heading from mask orientation, disambiguated by motion.

A robot's mask gives the direction of its long axis, but not which end is the front. mask_axis()
finds the axis in the image from the mask's second moments, axis_to_field() maps it onto the field
through the homography, and heading_measurements() picks the end that points the way the robot is
moving or, when it is nearly still, the end nearest its last heading. When the mask is too square
to have a long axis, a moving robot's direction of travel is used instead, with more uncertainty.
estimate_heading() runs it all between two passes of the Kalman smoother.

Two assumptions: robots mostly drive forward, so a robot reversing for a long time can have its
heading flipped; and a mask seen from a raised camera includes the robot's sides and top, so its
axis only approximates the footprint's. Heading error on scrimmage ground truth will show how much
these matter.
"""

import math

import numpy as np

from spar_perception.calib.homography import image_to_field
from spar_perception.filter.kalman import KalmanConfig, SmoothedTrack, smooth_track
from spar_schema.frames import wrap_angle

MIN_ELONGATION = 1.3  # long over short spread; a squarer mask has no usable axis


def mask_axis(mask: np.ndarray, origin=(0.0, 0.0)) -> tuple[np.ndarray, float, float] | None:
    """The mask's centroid (u, v), long axis angle (radians, mod pi) and elongation, or None.

    mask is a bool array; origin is where its top-left pixel sits in the full image, for a mask
    cropped to its box. Angles are in image coordinates, with v pointing down.
    """
    vs, us = np.nonzero(mask)
    if len(us) < 3:
        return None
    u, v = us + float(origin[0]), vs + float(origin[1])
    centroid = np.array([u.mean(), v.mean()])
    spread, axes = np.linalg.eigh(np.cov(np.vstack([u - centroid[0], v - centroid[1]])))
    major = axes[:, 1]  # eigh sorts ascending
    angle = math.atan2(major[1], major[0]) % math.pi
    elongation = math.sqrt(spread[1] / max(spread[0], 1e-12))
    return centroid, angle, elongation


def axis_to_field(
    H: np.ndarray, centroid: np.ndarray, angle: float, step_px: float = 10.0
) -> float:
    """An image axis through centroid, as an angle on the field (radians, mod pi)."""
    tip = centroid + step_px * np.array([math.cos(angle), math.sin(angle)])
    start, end = image_to_field(H, np.vstack([centroid, tip]))
    return math.atan2(end[1] - start[1], end[0] - start[0]) % math.pi


def heading_measurements(
    velocity: np.ndarray,
    axis: np.ndarray,
    *,
    min_speed: float = 0.15,
    axis_std: float = 0.15,
    motion_std: float = 0.35,
) -> tuple[np.ndarray, np.ndarray]:
    """Heading (n,) and its variance (n,) per frame, NaN where there is nothing to go on.

    velocity is (n, 2) in m/s; axis is (n,) field axis angles mod pi, NaN where the mask has no
    long axis. axis_std and motion_std are the heading noise, in radians, from each source.
    """
    heading = np.full(len(axis), np.nan)
    variance = np.full(len(axis), np.nan)
    last = None
    for k, (vx, vy) in enumerate(np.asarray(velocity, dtype=float)):
        moving = math.hypot(vx, vy) >= min_speed
        reference = math.atan2(vy, vx) if moving else last
        if not math.isnan(axis[k]) and reference is not None:
            ends = (axis[k], axis[k] + math.pi)
            heading[k] = wrap_angle(min(ends, key=lambda end: abs(wrap_angle(end - reference))))
            variance[k] = axis_std**2
        elif moving:
            heading[k] = wrap_angle(math.atan2(vy, vx))
            variance[k] = motion_std**2
        if not math.isnan(heading[k]):
            last = heading[k]
    return heading, variance


def estimate_heading(
    t, xy, xy_cov, axis, *, config: KalmanConfig | None = None, **measurement_options
) -> SmoothedTrack:
    """Smooth a track twice: first for velocity, then with headings measured from it.

    axis is each frame's field axis angle (mod pi), NaN where unknown. Without any heading to go
    on, the first pass comes back, and its heading is None.
    """
    first = smooth_track(t, xy, xy_cov, config=config)
    heading, variance = heading_measurements(
        first.velocity, np.asarray(axis, dtype=float), **measurement_options
    )
    if np.isnan(heading).all():
        return first
    return smooth_track(t, xy, xy_cov, heading=heading, heading_var=variance, config=config)


def field_axes(H: np.ndarray, boxes: np.ndarray, masks: list[np.ndarray | None]) -> np.ndarray:
    """The field axis angle (mod pi) of each box's mask, NaN where the mask is missing or square."""
    axes = np.full(len(boxes), np.nan)
    for k, (box, mask) in enumerate(zip(boxes, masks, strict=True)):
        if mask is None:
            continue
        found = mask_axis(mask, origin=(box[0], box[1]))
        if found is not None and found[2] >= MIN_ELONGATION:
            axes[k] = axis_to_field(H, found[0], found[1])
    return axes
