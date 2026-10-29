"""Tracking and identity on synthetic detections from a top-down camera, where truth is known."""

import math

import numpy as np
import pytest

from spar_perception.calib.homography import image_to_field, point_covariance
from spar_perception.filter.kalman import to_tracks_table
from spar_perception.heading import estimate_heading, field_axes
from spar_perception.track.bytetrack import (
    FrameDetections,
    join,
    split_on_alliance_flip,
    track_detections,
)
from spar_perception.track.identity import (
    RosterRobot,
    assign_identities,
    footprint,
    tracks_by_agent,
)
from spar_schema import TRACKS_SCHEMA
from spar_schema.frames import wrap_angle

FPS = 30
PX_PER_M = 200.0
# Image to field for a camera straight above the field center: u right, v down.
H = np.array([[1 / PX_PER_M, 0, -960 / PX_PER_M], [0, -1 / PX_PER_M, 540 / PX_PER_M], [0, 0, 1]])
RNG = np.random.default_rng(3)


def to_pixels(x, y):
    return 960 + PX_PER_M * x, 540 - PX_PER_M * y


def box(x, y, size_m, jitter=0.5):
    u, v = to_pixels(x, y)
    half = size_m / 2 * PX_PER_M
    noise = RNG.normal(0, jitter, 4)
    return np.array([u - half, v - half, u + half, v + half]) + noise


def frames_from(paths, n_frames, missing=None, labels=None):
    """paths: agent -> (function of t giving (x, y), size_m, alliance)."""
    missing = missing or {}
    labels = labels or {}
    out = []
    for f in range(n_frames):
        t = f / FPS
        boxes, scores, alliance = [], [], []
        for agent, (path, size, color) in paths.items():
            if f in missing.get(agent, ()):
                continue
            boxes.append(box(*path(t), size))
            scores.append(0.9)
            alliance.append(labels.get(agent, lambda f, c=color: c)(f))
        out.append(
            FrameDetections(
                f, t, np.array(boxes).reshape(-1, 4), np.array(scores), np.array(alliance)
            )
        )
    return out


def line(start, end, duration):
    def path(t):
        s = min(t / duration, 1.0)
        return start[0] + s * (end[0] - start[0]), start[1] + s * (end[1] - start[1])

    return path


V5RC = {
    "red_1": (line((-1.2, 0.8), (-0.2, 0.8), 4), 0.45, "red"),
    "red_2": (line((-1.2, -0.8), (-1.2, 0.0), 4), 0.45, "red"),
    "blue_1": (line((1.2, 0.8), (0.2, 0.8), 4), 0.45, "blue"),
    "blue_2": (line((1.2, -0.8), (0.4, -0.8), 4), 0.45, "blue"),
}
ROSTER = [RosterRobot(agent, color) for agent, (_, _, color) in V5RC.items()]


def identify(tracks, roster=ROSTER):
    positions, widths = {}, {}
    for track_id, track in tracks.items():
        positions[track_id], widths[track_id] = footprint(H, track.boxes)
    return assign_identities(tracks, positions, roster, widths=widths)


def test_each_robot_becomes_one_track():
    tracks = track_detections(frames_from(V5RC, 120), fps=FPS)
    assert len(tracks) == 4
    assert all(len(track.t) == 120 for track in tracks.values())
    assert sorted(track.majority_alliance() for track in tracks.values()) == [
        "blue",
        "blue",
        "red",
        "red",
    ]


def test_robot_lost_for_two_seconds_returns_as_the_same_robot():
    frames = frames_from(V5RC, 150, missing={"red_1": range(40, 100)})
    tracks = track_detections(frames, fps=FPS)
    assert len(tracks) == 5  # red_1 comes back under a new ID
    agents = tracks_by_agent(identify(tracks))
    assert sorted(agents) == ["blue_1", "blue_2", "red_1", "red_2"]
    assert len(agents["red_1"]) == 2


def test_partners_lost_together_come_back_by_position():
    paths = {
        "red_1": (line((-1.2, 0.8), (-1.0, 0.8), 5), 0.45, "red"),
        "red_2": (line((-1.2, -0.8), (-1.0, -0.8), 5), 0.45, "red"),
    }
    roster = [RosterRobot("red_1", "red"), RosterRobot("red_2", "red")]
    frames = frames_from(paths, 150, missing={"red_1": range(40, 100), "red_2": range(40, 100)})
    tracks = track_detections(frames, fps=FPS)
    assignments = identify(tracks, roster)
    for track_id, track in tracks.items():
        x, y = image_to_field(H, track.boxes[:1, :2] + 0.0)[0]
        expected = "red_1" if y > 0 else "red_2"
        assert assignments[track_id].agent_id == expected
    later = [tid for tid, track in tracks.items() if track.start > 1.0]
    assert all(assignments[tid].confidence > 0.9 for tid in later)


def test_vexu_robots_are_told_apart_by_size():
    paths = {
        "red_1": (line((-1.2, 0.8), (-0.5, 0.8), 3), 0.381, "red"),
        "red_2": (line((-1.2, -0.8), (-0.5, -0.8), 3), 0.610, "red"),
    }
    roster = [RosterRobot("red_1", "red", size_m=0.381), RosterRobot("red_2", "red", size_m=0.610)]
    tracks = track_detections(frames_from(paths, 60), fps=FPS)
    assignments = identify(tracks, roster)
    for track_id, track in tracks.items():
        width = np.median(track.boxes[:, 2] - track.boxes[:, 0]) / PX_PER_M
        assert assignments[track_id].agent_id == ("red_1" if width < 0.5 else "red_2")
        assert assignments[track_id].confidence > 0.8


def test_v5rc_partners_starting_together_are_a_coin_flip():
    tracks = track_detections(frames_from(V5RC, 30), fps=FPS)
    assignments = identify(tracks)
    assert {round(a.confidence, 2) for a in assignments.values()} == {0.5}


def test_extra_detection_gets_no_robot():
    paths = {**V5RC, "ghost": (line((0.0, 0.0), (0.1, 0.0), 4), 0.45, "red")}
    assignments = identify(track_detections(frames_from(paths, 60), fps=FPS))
    assert sum(a.agent_id is None for a in assignments.values()) == 1


def test_track_without_a_color_gets_no_robot():
    paths = {"red_1": (line((0, 0), (0.5, 0), 2), 0.45, "")}
    assignments = identify(track_detections(frames_from(paths, 30), fps=FPS))
    assert [a.agent_id for a in assignments.values()] == [None]


def test_track_is_cut_where_its_color_flips():
    paths = {"red_1": (line((0, 0), (1.0, 0), 3), 0.45, "red")}
    flipped = {"red_1": lambda f: "red" if f < 40 else "blue"}  # a swap with an opponent
    tracks = track_detections(frames_from(paths, 90, labels=flipped), fps=FPS)
    assert len(tracks) == 1
    pieces = split_on_alliance_flip(tracks, min_run=5)
    assert sorted(track.majority_alliance() for track in pieces.values()) == ["blue", "red"]
    assert sorted(len(track.t) for track in pieces.values()) == [40, 50]


def test_brief_color_noise_does_not_cut_a_track():
    paths = {"red_1": (line((0, 0), (1.0, 0), 3), 0.45, "red")}
    noisy = {"red_1": lambda f: "blue" if f in (20, 21, 50) else ("" if f % 7 == 0 else "red")}
    tracks = split_on_alliance_flip(track_detections(frames_from(paths, 90, labels=noisy), fps=FPS))
    assert len(tracks) == 1


# Masks and the whole path from detections to tracks.parquet rows.


def rotated_rect_detection(x, y, heading, length=0.5, width=0.3):
    """The box and the box-cropped mask of a rectangular robot seen from above."""
    u, v = to_pixels(x, y)
    corners = []
    for along, across in [(1, 1), (1, -1), (-1, -1), (-1, 1)]:
        cx = along * length / 2 * math.cos(heading) - across * width / 2 * math.sin(heading)
        cy = along * length / 2 * math.sin(heading) + across * width / 2 * math.cos(heading)
        corners.append(to_pixels(x + cx, y + cy))
    corners = np.array(corners)
    x1, y1 = np.floor(corners.min(axis=0))
    x2, y2 = np.ceil(corners.max(axis=0))
    gu, gv = np.meshgrid(np.arange(x1, x2) + 0.5, np.arange(y1, y2) + 0.5)
    fx, fy = (gu - u) / PX_PER_M, (v - gv) / PX_PER_M
    along = fx * math.cos(heading) + fy * math.sin(heading)
    across = -fx * math.sin(heading) + fy * math.cos(heading)
    mask = (np.abs(along) <= length / 2) & (np.abs(across) <= width / 2)
    return np.array([x1, y1, x2, y2]), mask


def test_detections_to_tracks_parquet_rows_with_heading():
    def arc(radius, cx, cy, phase):
        return lambda t: (
            cx + radius * math.cos(0.5 * t + phase),
            cy + radius * math.sin(0.5 * t + phase),
        )

    robots = {
        "red_1": (arc(0.6, -0.9, 0.0, 0.0), "red"),
        "red_2": (arc(0.5, -0.4, -1.0, 1.0), "red"),
        "blue_1": (arc(0.6, 0.9, 0.0, 2.0), "blue"),
        "blue_2": (arc(0.5, 0.4, 1.0, 3.0), "blue"),
    }
    n = 150
    frames, truth = [], {agent: [] for agent in robots}
    for f in range(n):
        t = f / FPS
        boxes, masks, labels = [], [], []
        for agent, (path, color) in robots.items():
            x, y = path(t)
            dx, dy = (np.array(path(t + 1e-3)) - (x, y)) / 1e-3
            heading = math.atan2(dy, dx)  # driving forward along the arc
            b, m = rotated_rect_detection(x, y, heading)
            boxes.append(b)
            masks.append(m)
            labels.append(color)
            truth[agent].append((x, y, heading))
        frames.append(
            FrameDetections(f, t, np.array(boxes), np.full(4, 0.9), np.array(labels), masks)
        )

    tracks = split_on_alliance_flip(track_detections(frames, fps=FPS))
    assignments = identify(
        tracks, [RosterRobot(agent, color) for agent, (_, color) in robots.items()]
    )
    for agent, track_ids in tracks_by_agent(assignments).items():
        series = join([tracks[tid] for tid in track_ids])
        centers = np.c_[
            (series.boxes[:, 0] + series.boxes[:, 2]) / 2,
            (series.boxes[:, 1] + series.boxes[:, 3]) / 2,
        ]
        xy = image_to_field(H, centers)  # straight down, a box center is the robot's center
        cov = point_covariance(H, centers, pixel_sigma=1.0) + np.eye(2) * 1e-4
        masks = [frames[f].masks[d] for f, d in zip(series.frames, series.detection, strict=True)]
        track = estimate_heading(series.t, xy, cov, field_axes(H, series.boxes, masks))

        expected = np.array(truth[agent])[series.frames]
        assert np.sqrt(np.mean(np.sum((track.xy - expected[:, :2]) ** 2, axis=1))) < 0.02
        error = np.abs(wrap_angle(track.heading - expected[:, 2]))
        assert np.degrees(np.median(error)) < 3 and np.degrees(error.max()) < 10, agent

        table = to_tracks_table(agent, series.frames, track)
        assert table.schema.equals(TRACKS_SCHEMA) and table.num_rows == n


@pytest.mark.parametrize("missing", [range(0), range(10, 12)])
def test_empty_frames_do_not_break_tracking(missing):
    frames = frames_from({"red_1": V5RC["red_1"]}, 40, missing={"red_1": missing})
    frames.insert(
        5, FrameDetections(99, 5.5 / FPS, np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=str))
    )
    assert len(track_detections(frames, fps=FPS)) == 1
