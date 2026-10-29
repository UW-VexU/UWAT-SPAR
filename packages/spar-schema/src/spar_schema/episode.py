"""PyArrow schemas for every episode table, EpisodeMeta, Provenance and other enums (spec 5.5).

An episode is one folder:

    <episode_id>/
    ├── episode.json      EpisodeMeta
    ├── tracks.parquet    robots over time                  TRACKS_SCHEMA
    ├── objects.parquet   objects over time                 OBJECTS_SCHEMA
    ├── events.parquet    discrete events                   EVENTS_SCHEMA
    ├── state.parquet     task state and points over time   STATE_SCHEMA
    ├── actions.parquet   plan-level actions                ACTIONS_SCHEMA
    ├── strategy.json     StrategyRecord
    └── outcome.json      Outcome

Enum columns are stored as strings. Values that depend on the game (object classes, checkpoints,
programs, alliances) come from the scene, and EpisodeWriter checks them against it.
"""

from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Any, Literal, Self

import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field

from spar_schema.version import SCHEMA_VERSION


class EpisodeKind(StrEnum):
    MATCH = "match"
    RENDER = "render"
    SCRIMMAGE = "scrimmage"
    SIMULATED = "simulated"


class Provenance(StrEnum):
    """Where a track or object row came from."""

    DETECTED = "detected"
    INTERPOLATED = "interpolated"
    RENDERED = "rendered"
    LOGGED = "logged"
    SIMULATED = "simulated"
    HUMAN = "human"


class ObjectStatus(StrEnum):
    FREE = "free"
    HELD = "held"
    PLACED = "placed"
    UNKNOWN = "unknown"


class EventSource(StrEnum):
    RULE = "rule"
    MODEL = "model"
    HUMAN = "human"
    LOGGED = "logged"
    SIMULATED = "simulated"


# One row per robot per frame.
TRACKS_SCHEMA = pa.schema(
    [
        pa.field("t", pa.float64(), nullable=False),  # seconds from match start
        pa.field("frame", pa.int32()),  # frame index in the source video
        pa.field("agent_id", pa.string(), nullable=False),  # key in EpisodeMeta.roster
        pa.field("x", pa.float32(), nullable=False),  # field position (m)
        pa.field("y", pa.float32(), nullable=False),
        pa.field("heading", pa.float32(), nullable=False),  # radians
        pa.field("vx", pa.float32()),  # m/s
        pa.field("vy", pa.float32()),
        pa.field("omega", pa.float32()),  # rad/s
        pa.field("cov", pa.list_(pa.float32(), 9)),  # 3x3 covariance of (x, y, heading), row major
        pa.field("conf", pa.float32()),  # detection or track confidence
        pa.field("provenance", pa.string(), nullable=False),  # Provenance
    ]
)

# One row per tracked object per frame or per change.
OBJECTS_SCHEMA = pa.schema(
    [
        pa.field("t", pa.float64(), nullable=False),
        pa.field("frame", pa.int32()),
        pa.field("object_id", pa.string(), nullable=False),  # stable within the episode
        pa.field("cls", pa.string(), nullable=False),  # object class from the scene
        pa.field("x", pa.float32()),  # position (m) when known
        pa.field("y", pa.float32()),
        pa.field("z", pa.float32()),
        pa.field("status", pa.string(), nullable=False),  # ObjectStatus
        pa.field("holder", pa.string()),  # agent ID when held
        pa.field("goal_id", pa.string()),  # goal when placed
        pa.field("stack_index", pa.int16()),  # position in the stack
        pa.field("attrs", pa.string()),  # JSON, scene-specific state
        pa.field("conf", pa.float32()),
        pa.field("provenance", pa.string(), nullable=False),  # Provenance
    ]
)

EVENTS_SCHEMA = pa.schema(
    [
        pa.field("t", pa.float64(), nullable=False),
        pa.field("agent_id", pa.string()),  # who did it, if known
        pa.field("type", pa.string(), nullable=False),  # scene-defined, for example "place"
        pa.field("object_id", pa.string()),
        pa.field("goal_id", pa.string()),
        pa.field("target_id", pa.string()),
        pa.field("conf", pa.float32()),
        pa.field("source", pa.string(), nullable=False),  # EventSource
    ]
)

# One row per state change, plus one per checkpoint.
STATE_SCHEMA = pa.schema(
    [
        pa.field("t", pa.float64(), nullable=False),
        pa.field("checkpoint", pa.string()),  # a checkpoint from the scene's objective
        pa.field("points_red", pa.float32(), nullable=False),  # from the scene's score()
        pa.field("points_blue", pa.float32(), nullable=False),
        pa.field("state", pa.string(), nullable=False),  # JSON, the full task state
    ]
)

ACTIONS_SCHEMA = pa.schema(
    [
        pa.field("t_start", pa.float64(), nullable=False),
        pa.field("t_end", pa.float64(), nullable=False),
        pa.field("agent_id", pa.string(), nullable=False),
        pa.field("action", pa.string(), nullable=False),  # from the simulator's action set
        pa.field("args", pa.string()),  # JSON, for example {"goal_id": "N3"}
        pa.field("inferred", pa.bool_(), nullable=False),  # True if recovered from video
    ]
)


@dataclass(frozen=True)
class TableSpec:
    """One Parquet table in an episode folder."""

    name: str
    schema: pa.Schema
    enums: dict[str, type[StrEnum]]  # columns whose values must come from an enum

    @property
    def filename(self) -> str:
        return f"{self.name}.parquet"


TABLES = {
    spec.name: spec
    for spec in [
        TableSpec("tracks", TRACKS_SCHEMA, {"provenance": Provenance}),
        TableSpec("objects", OBJECTS_SCHEMA, {"status": ObjectStatus, "provenance": Provenance}),
        TableSpec("events", EVENTS_SCHEMA, {"source": EventSource}),
        TableSpec("state", STATE_SCHEMA, {}),
        TableSpec("actions", ACTIONS_SCHEMA, {}),
    ]
}


class _Record(BaseModel):
    # Unknown keys are kept, so files from a newer minor schema version still load.
    model_config = ConfigDict(extra="allow")


class RosterEntry(_Record):
    """One agent in an episode."""

    alliance: str
    team: str | None = None  # team number, for example "1234A"


class VideoRef(_Record):
    """Where the episode's video is. Video is never copied, only referenced."""

    stream_id: str
    start_s: float  # episode start, in seconds into the stream
    end_s: float
    fps: float
    resolution: tuple[int, int]  # width, height


class PipelineInfo(_Record):
    """What produced the episode."""

    git_commit: str | None = None
    package_versions: dict[str, str] = {}
    config_hash: str | None = None


class ScoreCheck(_Record):
    """Score check result at one checkpoint (spec section 6.4)."""

    computed: dict[str, float]  # points per alliance from the scene's score()
    computed_winner: str | None = None  # red, blue or tie, when compared by winner
    official: dict[str, float] | None = None  # official points, when published
    official_winner: str | None = None  # when only the winner is published
    difference: dict[str, float] | None = None  # computed minus official
    passed: bool | None  # None when the official result cannot be compared


class EpisodeMeta(_Record):
    """episode.json."""

    episode_id: str  # for example "<scene>_<event_sku>_<match_name>", or a render or scrimmage ID
    kind: EpisodeKind
    scene: str
    rules_version: str
    schema_version: str = SCHEMA_VERSION
    program: str  # a program from the scene
    event_sku: str | None = None
    match_name: str | None = None
    match_date: date | None = None  # the day the match was played; the cutoff guard reads it
    roster: dict[str, RosterEntry]  # agent ID, for example "red_1", to its team and alliance
    video: VideoRef | None = None
    pipeline: PipelineInfo = Field(default_factory=PipelineInfo)
    score_check: dict[str, ScoreCheck] = {}  # by checkpoint
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Outcome(_Record):
    """outcome.json: the official result."""

    points_red: int
    points_blue: int
    autonomous_winner: Literal["red", "blue", "tie", "none"]
    # Where each part came from, for example {"final": "events_api", "autonomous": "stream_overlay"}
    sources: dict[str, str] = {}


class StrategyRecord(BaseModel):
    """strategy.json: a strategy code per alliance, and who coded it.

    On disk the codes sit at the top level next to "coder" and "code_version", for example
    {"red": {...}, "blue": {...}, "coder": "pipeline", "code_version": "0.1"}.
    """

    codes: dict[str, dict[str, Any]]  # by alliance
    coder: str = Field(pattern=r"^(pipeline|human:.+)$")
    code_version: str

    def to_json_dict(self) -> dict[str, Any]:
        return {**self.codes, "coder": self.coder, "code_version": self.code_version}

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> Self:
        codes = dict(data)
        coder = codes.pop("coder", None)
        code_version = codes.pop("code_version", None)
        return cls(codes=codes, coder=coder, code_version=code_version)
