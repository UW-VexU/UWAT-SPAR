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
