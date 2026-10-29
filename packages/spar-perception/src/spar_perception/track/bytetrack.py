"""ByteTrack from the trackers package: per-frame robot detections to tracks.

FrameDetections is what the detector (detect/infer.py) hands over for each frame.
track_detections() links detections across frames with ByteTrack and returns one Track per ID.
ByteTrack matches boxes by overlap and motion only, so robots that cross can swap IDs. A swap
between alliances shows up as a flip in the detector's color label, and split_on_alliance_flip()
cuts a track there. A swap between partners does not; track/identity.py says more.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import supervision as sv
from trackers import ByteTrackTracker


@dataclass(frozen=True)
class FrameDetections:
    """One frame of robot detections from the detector."""

    frame: int
    t: float  # seconds from match start
    boxes: np.ndarray  # (N, 4) x1, y1, x2, y2 in pixels
    scores: np.ndarray  # (N,) detector confidence
    alliance: np.ndarray  # (N,) "red", "blue", or "" where the color is unclear
    masks: list[np.ndarray] | None = (
        None  # bool mask cropped to each box, from a segmenting detector
    )


@dataclass(frozen=True)
class Track:
    """The detections linked under one track ID, in frame order."""

    track_id: int
    frames: np.ndarray  # (n,) frame index
    t: np.ndarray  # (n,) seconds
    boxes: np.ndarray  # (n, 4) pixels
    scores: np.ndarray  # (n,)
    alliance: np.ndarray  # (n,) color label per detection
    detection: np.ndarray  # (n,) index of each detection within its frame

    @property
    def start(self) -> float:
        return float(self.t[0])

    @property
    def end(self) -> float:
        return float(self.t[-1])

    def majority_alliance(self) -> str | None:
        """The most common color label, or None if the detector never saw a color."""
        labels = [label for label in self.alliance if label]
        if not labels:
            return None
        values, counts = np.unique(labels, return_counts=True)
        return str(values[np.argmax(counts)])

    def take(self, rows: np.ndarray, track_id: int) -> "Track":
        return Track(
            track_id=track_id,
            frames=self.frames[rows],
            t=self.t[rows],
            boxes=self.boxes[rows],
            scores=self.scores[rows],
            alliance=self.alliance[rows],
            detection=self.detection[rows],
        )


@dataclass(frozen=True)
class TrackConfig:
    lost_track_buffer_s: float = 1.0  # how long a lost robot's track waits to be found again
    activation_score: float = 0.5  # detections below this cannot start a track
    high_score: float = 0.6  # ByteTrack matches detections above this first
    min_iou: float = 0.1  # least box overlap for a match


def track_detections(
    frames: Iterable[FrameDetections], *, fps: float, config: TrackConfig | None = None
) -> dict[int, Track]:
    """Run ByteTrack over a match, frame by frame, and return its tracks by ID."""
    config = config or TrackConfig()
    tracker = ByteTrackTracker(
        lost_track_buffer=max(1, round(config.lost_track_buffer_s * fps)),
        frame_rate=fps,
        track_activation_threshold=config.activation_score,
        minimum_consecutive_frames=1,  # give a track its ID on its first frame
        minimum_iou_threshold=config.min_iou,
        high_conf_det_threshold=config.high_score,
    )
    rows: dict[int, list] = {}
    unconfirmed: list[tuple[FrameDetections, int]] = []  # last frame's detections without an ID
    for fd in frames:
        n = len(fd.boxes)
        detections = (
            sv.Detections(
                xyxy=np.asarray(fd.boxes, dtype=float).reshape(-1, 4),
                confidence=np.asarray(fd.scores, dtype=float),
                class_id=np.zeros(n, dtype=int),
                data={"index": np.arange(n)},
            )
            if n
            else sv.Detections.empty()
        )
        out = tracker.update(detections)
        waiting = []
        for track_id, index in zip(out.tracker_id, out.data.get("index", []), strict=True):
            if track_id < 0:
                waiting.append((fd, int(index)))
                continue
            if int(track_id) not in rows:
                # ByteTrack names a track one frame after it starts; recover that first frame.
                rows[int(track_id)] = _first_detection(unconfirmed, fd.boxes[index], config.min_iou)
            rows[int(track_id)].append((fd, int(index)))
        unconfirmed = waiting
    return {track_id: _build(track_id, found) for track_id, found in rows.items()}


def split_on_alliance_flip(tracks: dict[int, Track], min_run: int = 5) -> dict[int, Track]:
    """Cut each track where its color label changes and stays changed for min_run detections.

    Shorter runs of the other color count as noise. Pieces after the first get new IDs.
    """
    next_id = max(tracks, default=-1) + 1
    out = {}
    for track_id, track in sorted(tracks.items()):
        cuts = [*_flip_points(track.alliance, min_run), len(track.t)]
        start = 0
        for piece, cut in enumerate(cuts):
            new_id = track_id if piece == 0 else next_id
            next_id += piece > 0
            out[new_id] = track.take(np.arange(start, cut), new_id)
            start = cut
    return out


def join(tracks: Sequence[Track], track_id: int = -1) -> Track:
    """Several tracks of one robot as one, in time order. They must not overlap in time."""
    ordered = sorted(tracks, key=lambda track: track.start)
    return Track(
        track_id=track_id,
        frames=np.concatenate([track.frames for track in ordered]),
        t=np.concatenate([track.t for track in ordered]),
        boxes=np.concatenate([track.boxes for track in ordered]),
        scores=np.concatenate([track.scores for track in ordered]),
        alliance=np.concatenate([track.alliance for track in ordered]),
        detection=np.concatenate([track.detection for track in ordered]),
    )


def _first_detection(
    unconfirmed: list[tuple[FrameDetections, int]], box: np.ndarray, min_iou: float
) -> list[tuple[FrameDetections, int]]:
    """The unconfirmed detection that overlaps box most, taken out of the list, if any."""
    overlaps = [_iou(fd.boxes[index], box) for fd, index in unconfirmed]
    if not overlaps or max(overlaps) < min_iou:
        return []
    return [unconfirmed.pop(int(np.argmax(overlaps)))]


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = width * height
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def _build(track_id: int, found: list[tuple[FrameDetections, int]]) -> Track:
    return Track(
        track_id=track_id,
        frames=np.array([fd.frame for fd, _ in found]),
        t=np.array([fd.t for fd, _ in found], dtype=float),
        boxes=np.array([fd.boxes[i] for fd, i in found], dtype=float).reshape(-1, 4),
        scores=np.array([fd.scores[i] for fd, i in found], dtype=float),
        alliance=np.array([fd.alliance[i] for fd, i in found], dtype=str),
        detection=np.array([i for _, i in found]),
    )


def _flip_points(labels: np.ndarray, min_run: int) -> list[int]:
    runs: list[list] = []  # [label, first index, how many detections carry it]
    for i, label in enumerate(labels):
        if not label:
            continue
        if runs and runs[-1][0] == label:
            runs[-1][2] += 1
        else:
            runs.append([label, i, 1])
    solid: list[list] = []
    for run in runs:
        if run[2] >= min_run and (not solid or solid[-1][0] != run[0]):
            solid.append(run)
    return [run[1] for run in solid[1:]]
