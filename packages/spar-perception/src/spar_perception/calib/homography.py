"""Per-frame homography with OpenCV RANSAC, smoothed over time per camera.

The homography H maps image pixels to field-frame meters on the floor (z = 0). It is fit from
landmarks: pixel positions in one frame paired with the scene's named field points. A livestream
camera barely moves, so per-frame fits are smoothed over time: smooth_homographies() follows where
a few anchor points on the floor land in the image, takes a running median, and refits.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class HomographyFit:
    """An image-to-field homography and how well it fits its landmarks."""

    H: np.ndarray  # 3x3, image (u, v, 1) to field (x, y, 1)
    inliers: np.ndarray  # bool per point pair, False where RANSAC rejected it
    rms_px: float  # reprojection error over the inliers, in pixels


def floor_landmarks(
    landmarks: Mapping[str, Sequence[float]], tol_m: float = 1e-6
) -> dict[str, tuple[float, float]]:
    """The landmarks on the floor (z = 0), as (x, y). Only these can fit a floor homography."""
    return {name: (float(p[0]), float(p[1])) for name, p in landmarks.items() if abs(p[2]) <= tol_m}


def fit_homography(image_points, field_points, *, ransac_px: float = 3.0) -> HomographyFit:
    """Fit the image-to-field homography from at least 4 pixel and field point pairs.

    RANSAC drops pairs that reproject more than ransac_px pixels away, so one mislabeled landmark
    does not bend the fit. Raises ValueError with fewer than 4 pairs or no consistent fit.
    """
    image = _points(image_points)
    field = _points(field_points)
    if len(image) != len(field):
        raise ValueError(f"{len(image)} image points but {len(field)} field points")
    if len(image) < 4:
        raise ValueError(f"a homography needs at least 4 point pairs, got {len(image)}")
    # Fit field to image, so RANSAC measures its error in pixels, then invert.
    G, mask = cv2.findHomography(field, image, cv2.RANSAC, ransac_px)
    if G is None:
        raise ValueError("no consistent homography for these points")
    inliers = mask.ravel().astype(bool)
    H = np.linalg.inv(G)
    H /= H[2, 2]
    error = field_to_image(H, field[inliers]) - image[inliers]
    rms = float(np.sqrt(np.mean(np.sum(error**2, axis=1))))
    return HomographyFit(H=H, inliers=inliers, rms_px=rms)


def image_to_field(H: np.ndarray, pixels) -> np.ndarray:
    """Pixels (N, 2) to field points (N, 2) in meters."""
    return _apply(H, _points(pixels))


def field_to_image(H: np.ndarray, field_points) -> np.ndarray:
    """Field points (N, 2) in meters to pixels (N, 2)."""
    return _apply(np.linalg.inv(H), _points(field_points))


def point_covariance(H: np.ndarray, pixels, pixel_sigma) -> np.ndarray:
    """Covariance (N, 2, 2) in m^2 of the field points under pixel noise of pixel_sigma.

    Linearizes the homography at each pixel: cov = J (sigma^2 I) J^T. Points far from the camera
    get larger covariance, because one pixel covers more floor there. pixel_sigma is a scalar or
    one value per pixel.
    """
    p = _points(pixels)
    q = np.c_[p, np.ones(len(p))] @ H.T  # (N, 3)
    w = q[:, 2]
    jac = np.empty((len(p), 2, 2))
    for row in range(2):
        for col in range(2):
            jac[:, row, col] = (H[row, col] * w - q[:, row] * H[2, col]) / w**2
    sigma2 = np.broadcast_to(np.asarray(pixel_sigma, dtype=np.float64) ** 2, (len(p),))
    return jac @ jac.transpose(0, 2, 1) * sigma2[:, None, None]


def smooth_homographies(
    homographies: Sequence[np.ndarray | None], anchors, *, window: int = 15
) -> list[np.ndarray]:
    """Smooth per-frame homographies over time, filling frames that have none.

    anchors are at least 4 field points spread over the floor, such as the field's corners. Where
    each lands in the image is interpolated across frames without a fit, smoothed with a centered
    running median over `window` frames, and the homography is refit from them. The median keeps
    one bad frame from pulling its neighbors.
    """
    anchors = _points(anchors)
    if len(anchors) < 4:
        raise ValueError(f"need at least 4 anchors, got {len(anchors)}")
    have = np.array([H is not None for H in homographies])
    if not have.any():
        raise ValueError("no frame has a homography")
    frames = np.arange(len(homographies))
    tracks = np.full((len(homographies), len(anchors), 2), np.nan)
    for i in frames[have]:
        tracks[i] = field_to_image(homographies[i], anchors)
    for k in range(len(anchors)):
        for axis in range(2):
            series = tracks[:, k, axis]
            series[~have] = np.interp(frames[~have], frames[have], series[have])

    half = window // 2
    padded = np.pad(tracks, ((half, half), (0, 0), (0, 0)), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1, axis=0)
    smoothed = np.median(windows, axis=-1)

    out = []
    for image in smoothed:
        G, _ = cv2.findHomography(anchors, image, 0)
        H = np.linalg.inv(G)
        out.append(H / H[2, 2])
    return out


def _points(points) -> np.ndarray:
    return np.asarray(points, dtype=np.float64).reshape(-1, 2)


def _apply(M: np.ndarray, p: np.ndarray) -> np.ndarray:
    q = np.c_[p, np.ones(len(p))] @ M.T
    return q[:, :2] / q[:, 2:3]
