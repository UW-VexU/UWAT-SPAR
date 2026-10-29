"""Field frame helpers and unit conversions (spec section 5.1).

SPAR uses SI units everywhere: meters, radians, seconds, meters per second. Each scene has one
field frame following ROS REP 103 (x forward, y left, z up, right-handed), with its origin and x
direction set in scene.yaml. These helpers work on floats and on NumPy or JAX arrays alike.
"""

import math

INCH = 0.0254  # meters
FOOT = 0.3048  # meters


def inches(value):
    """Inches to meters."""
    return value * INCH


def feet(value):
    """Feet to meters."""
    return value * FOOT


def wrap_angle(theta):
    """Wrap an angle in radians to [-pi, pi)."""
    return (theta + math.pi) % (2 * math.pi) - math.pi


def polygon_distance(x: float, y: float, polygon) -> float:
    """Signed distance in meters from (x, y) to a polygon's edge: negative inside, positive outside.

    polygon is a list of [x, y] corners in order. A robot of reach r overlaps the polygon when
    the distance from its center is below r.
    """
    corners = [(float(p[0]), float(p[1])) for p in polygon]
    nearest = math.inf
    inside = False
    for (x1, y1), (x2, y2) in zip(corners, corners[1:] + corners[:1], strict=True):
        dx, dy = x2 - x1, y2 - y1
        along = ((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy)
        along = min(1.0, max(0.0, along))
        nearest = min(nearest, math.hypot(x - (x1 + along * dx), y - (y1 + along * dy)))
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * dx / dy:
            inside = not inside
    return -nearest if inside else nearest


def polygon_distances(xy, polygon, xp):
    """polygon_distance for many points at once: xy is [..., 2], and the result is [...].

    xp is the array module (numpy or jax.numpy); the call also works under jax.jit.
    """
    a = xp.asarray(polygon, dtype=xp.float32)  # [n, 2] corners
    b = xp.roll(a, -1, axis=0)
    edge = b - a
    p = xp.asarray(xy, dtype=xp.float32)[..., None, :]  # [..., 1, 2]
    along = xp.clip(xp.sum((p - a) * edge, axis=-1) / xp.sum(edge * edge, axis=-1), 0.0, 1.0)
    gap = p - (a + along[..., None] * edge)
    nearest = xp.min(xp.sqrt(xp.sum(gap * gap, axis=-1)), axis=-1)
    px, py = p[..., 0], p[..., 1]
    dy = xp.where(edge[:, 1] == 0, 1.0, edge[:, 1])
    crosses = ((a[:, 1] > py) != (b[:, 1] > py)) & (px < a[:, 0] + (py - a[:, 1]) * edge[:, 0] / dy)
    inside = xp.sum(crosses, axis=-1) % 2 == 1
    return xp.where(inside, -nearest, nearest)
