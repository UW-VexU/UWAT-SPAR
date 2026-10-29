"""Constant-velocity Kalman filter with an RTS smoother: position, heading, velocity, covariance.

A robot's field positions from the homography are noisy, and missing where the detector lost it.
smooth_track() runs a constant-velocity Kalman filter forward over the match and a
Rauch-Tung-Striebel smoother backward, which gives position, heading, velocity and covariance at
every frame, including frames without a measurement. Position and heading are filtered separately,
since a constant-velocity model does not couple them. Heading is an angle, so its corrections are
wrapped to [-pi, pi).
"""

from dataclasses import dataclass

import numpy as np
import pyarrow as pa

from spar_schema import TRACKS_SCHEMA, Provenance, validate_table
from spar_schema.frames import wrap_angle


@dataclass(frozen=True)
class KalmanConfig:
    accel_std: float = 2.0  # unmodeled acceleration, m/s^2
    angular_accel_std: float = 8.0  # unmodeled angular acceleration, rad/s^2
    initial_speed_std: float = 2.0  # prior on each velocity component at the start, m/s
    initial_turn_rate_std: float = 4.0  # prior on the turn rate at the start, rad/s


@dataclass(frozen=True)
class SmoothedTrack:
    t: np.ndarray  # (N,) seconds from match start
    xy: np.ndarray  # (N, 2) field position, m
    velocity: np.ndarray  # (N, 2) m/s
    heading: np.ndarray | None  # (N,) rad in [-pi, pi), None when no heading was measured
    omega: np.ndarray | None  # (N,) rad/s
    cov: np.ndarray  # (N, 3, 3) covariance of (x, y, heading); heading terms NaN without heading
    measured: np.ndarray  # (N,) True where a position was measured


def smooth_track(
    t,
    xy,
    xy_cov,
    heading=None,
    heading_var=None,
    config: KalmanConfig | None = None,
) -> SmoothedTrack:
    """Smooth one robot's track.

    t is (N,) seconds, strictly increasing. xy is (N, 2) measured positions with NaN rows where the
    robot was not measured, and xy_cov is their (N, 2, 2) covariance, for example from
    calib.homography.point_covariance. heading (N,) and heading_var (scalar or (N,)) are optional,
    with NaN where not measured.
    """
    config = config or KalmanConfig()
    t = np.asarray(t, dtype=np.float64)
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    xy_cov = np.asarray(xy_cov, dtype=np.float64).reshape(-1, 2, 2)
    if not (len(t) == len(xy) == len(xy_cov)):
        raise ValueError("t, xy and xy_cov need the same length")
    if len(t) > 1 and np.any(np.diff(t) <= 0):
        raise ValueError("t must be strictly increasing")

    pos, pos_cov, _, _ = _cv_smooth(t, xy, xy_cov, config.accel_std, config.initial_speed_std)
    cov = np.full((len(t), 3, 3), np.nan)
    cov[:, :2, :2] = pos_cov[:, :2, :2]
    heading_out = omega = None
    if heading is not None:
        z = np.asarray(heading, dtype=np.float64).reshape(-1, 1)
        r = np.broadcast_to(np.asarray(heading_var, dtype=np.float64), (len(t),))
        ang, ang_cov, _, _ = _cv_smooth(
            t,
            z,
            r.reshape(-1, 1, 1),
            config.angular_accel_std,
            config.initial_turn_rate_std,
            wrap=True,
        )
        heading_out, omega = ang[:, 0], ang[:, 1]
        cov[:, :2, 2] = cov[:, 2, :2] = 0.0
        cov[:, 2, 2] = ang_cov[:, 0, 0]
    return SmoothedTrack(
        t=t,
        xy=pos[:, :2],
        velocity=pos[:, 2:],
        heading=heading_out,
        omega=omega,
        cov=cov,
        measured=~np.isnan(xy).any(axis=1),
    )


def to_tracks_table(agent_id: str, frames, track: SmoothedTrack, conf=None) -> pa.Table:
    """Rows of tracks.parquet for one agent: detected where measured, interpolated elsewhere."""
    if track.heading is None:
        raise ValueError("tracks.parquet needs a heading; pass heading to smooth_track")
    n = len(track.t)
    provenance = np.where(track.measured, Provenance.DETECTED.value, Provenance.INTERPOLATED.value)
    table = pa.table(
        {
            "t": track.t,
            "frame": np.asarray(frames, dtype=np.int32),
            "agent_id": [agent_id] * n,
            "x": track.xy[:, 0],
            "y": track.xy[:, 1],
            "heading": track.heading,
            "vx": track.velocity[:, 0],
            "vy": track.velocity[:, 1],
            "omega": track.omega,
            "cov": track.cov.reshape(n, 9).tolist(),
            "conf": pa.nulls(n, pa.float32())
            if conf is None
            else np.asarray(conf, dtype=np.float32),
            "provenance": provenance.tolist(),
        }
    )
    return validate_table(
        table, TRACKS_SCHEMA, name="tracks", allowed={"provenance": {p.value for p in Provenance}}
    )


def _cv_smooth(t, z, R, accel_std, initial_rate_std, wrap=False):
    """Kalman filter and RTS smoother for a constant-velocity model of d coordinates.

    z is (N, d) with NaN rows where not measured, and R is (N, d, d). The state is [position (d),
    rate (d)]. Returns the smoothed means (N, 2d) and covariances (N, 2d, 2d), then the forward
    filter's.
    """
    n_steps, d = z.shape
    eye_d, eye = np.eye(d), np.eye(2 * d)
    obs = np.hstack([eye_d, np.zeros((d, d))])
    measured = ~np.isnan(z).any(axis=1)
    if not measured.any():
        raise ValueError("no measurements to smooth")

    def transition(dt):
        F = eye.copy()
        F[:d, d:] = dt * eye_d
        Q = accel_std**2 * np.block(
            [[dt**4 / 4 * eye_d, dt**3 / 2 * eye_d], [dt**3 / 2 * eye_d, dt**2 * eye_d]]
        )
        return F, Q

    def wrapped(v):
        if wrap:
            v = v.copy()
            v[:d] = wrap_angle(v[:d])
        return v

    # Start at the first measurement, with a loose prior so it carries little weight.
    x = np.zeros(2 * d)
    x[:d] = z[np.argmax(measured)]
    P = np.zeros((2 * d, 2 * d))
    P[:d, :d] = 100.0 * eye_d
    P[d:, d:] = initial_rate_std**2 * eye_d

    x_pred = np.empty((n_steps, 2 * d))
    P_pred = np.empty((n_steps, 2 * d, 2 * d))
    x_filt = np.empty_like(x_pred)
    P_filt = np.empty_like(P_pred)
    for k in range(n_steps):
        if k:
            F, Q = transition(t[k] - t[k - 1])
            x = wrapped(F @ x)
            P = F @ P @ F.T + Q
        x_pred[k], P_pred[k] = x, P
        if measured[k]:
            innovation = z[k] - obs @ x
            if wrap:
                innovation = wrap_angle(innovation)
            S = obs @ P @ obs.T + R[k]
            K = P @ obs.T @ np.linalg.inv(S)
            x = wrapped(x + K @ innovation)
            A = eye - K @ obs
            P = A @ P @ A.T + K @ R[k] @ K.T  # Joseph form keeps P symmetric
        x_filt[k], P_filt[k] = x, P

    x_smooth, P_smooth = x_filt.copy(), P_filt.copy()
    for k in range(n_steps - 2, -1, -1):
        F, _ = transition(t[k + 1] - t[k])
        C = P_filt[k] @ F.T @ np.linalg.inv(P_pred[k + 1])
        correction = x_smooth[k + 1] - x_pred[k + 1]
        if wrap:
            correction[:d] = wrap_angle(correction[:d])
        x_smooth[k] = wrapped(x_filt[k] + C @ correction)
        P_smooth[k] = P_filt[k] + C @ (P_smooth[k + 1] - P_pred[k + 1]) @ C.T
    return x_smooth, P_smooth, x_filt, P_filt
