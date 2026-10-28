"""spar-schema: shared contracts: scene loading, table schemas, episode reader and writer (spec
section 5).

Public API only.
"""

from spar_schema.episode import (
    ACTIONS_SCHEMA,
    EVENTS_SCHEMA,
    OBJECTS_SCHEMA,
    STATE_SCHEMA,
    TABLES,
    TRACKS_SCHEMA,
    EpisodeKind,
    EpisodeMeta,
    EventSource,
    ObjectStatus,
    Outcome,
    PipelineInfo,
    Provenance,
    RosterEntry,
    ScoreCheck,
    StrategyRecord,
    TableSpec,
    VideoRef,
)
from spar_schema.io import EpisodeReader, EpisodeWriter, SchemaError, read_table, validate_table
from spar_schema.scene import (
    Scene,
    SceneError,
    SceneIncompleteError,
    SceneSpec,
    SceneVersionError,
    load_scene,
)
from spar_schema.start_state import (
    START_STATES_SCHEMA,
    RobotState,
    StartState,
    read_start_states,
    write_start_states,
)
from spar_schema.task_state import StateFieldSpec, TaskState, TaskStateSpec
from spar_schema.version import SCHEMA_VERSION, SchemaVersionError, check_compatible

__all__ = [
    "ACTIONS_SCHEMA",
    "EVENTS_SCHEMA",
    "OBJECTS_SCHEMA",
    "SCHEMA_VERSION",
    "START_STATES_SCHEMA",
    "STATE_SCHEMA",
    "TABLES",
    "TRACKS_SCHEMA",
    "EpisodeKind",
    "EpisodeMeta",
    "EpisodeReader",
    "EpisodeWriter",
    "EventSource",
    "ObjectStatus",
    "Outcome",
    "PipelineInfo",
    "Provenance",
    "RobotState",
    "RosterEntry",
    "Scene",
    "SceneError",
    "SceneIncompleteError",
    "SceneSpec",
    "SceneVersionError",
    "SchemaError",
    "SchemaVersionError",
    "ScoreCheck",
    "StartState",
    "StateFieldSpec",
    "StrategyRecord",
    "TableSpec",
    "TaskState",
    "TaskStateSpec",
    "VideoRef",
    "check_compatible",
    "load_scene",
    "read_start_states",
    "read_table",
    "validate_table",
    "write_start_states",
]
