"""EpisodeWriter, EpisodeReader and validation."""

import json
from functools import cached_property
from pathlib import Path
from types import TracebackType
from typing import Self

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from spar_schema.episode import TABLES, EpisodeMeta, Outcome, StrategyRecord
from spar_schema.scene import Scene
from spar_schema.version import META_SCHEMA_VERSION, SCHEMA_VERSION, check_compatible, file_metadata

META_FILE = "episode.json"
OUTCOME_FILE = "outcome.json"
STRATEGY_FILE = "strategy.json"


class SchemaError(ValueError):
    """A table or record does not match the episode format."""


def validate_table(
    table: pa.Table,
    schema: pa.Schema,
    *,
    name: str,
    allowed: dict[str, set[str]] | None = None,
) -> pa.Table:
    """Check a table against a schema and return it in the schema's column order and types.

    The table must have exactly the schema's columns, no nulls in required columns, and values
    that cast safely. `allowed` maps columns to the values they may hold; nulls are always allowed
    in nullable columns.
    """
    missing = [column for column in schema.names if column not in table.column_names]
    unexpected = sorted(set(table.column_names) - set(schema.names))
    if missing or unexpected:
        raise SchemaError(f"{name}: missing columns {missing}, unexpected columns {unexpected}")
    table = table.select(schema.names)
    for field in schema:
        if not field.nullable and table.column(field.name).null_count:
            raise SchemaError(f"{name}.{field.name}: nulls in a required column")
    try:
        table = table.cast(schema)
    except pa.ArrowException as exc:
        raise SchemaError(f"{name}: {exc}") from exc
    for column, values in (allowed or {}).items():
        found = set(pc.unique(table.column(column)).to_pylist()) - {None}
        if unknown := sorted(found - values):
            raise SchemaError(
                f"{name}.{column}: unknown values {unknown}, expected one of {sorted(values)}"
            )
    return table


def read_table(path: str | Path, schema: pa.Schema) -> pa.Table:
    """Read a Parquet file SPAR wrote, refusing files from an unknown schema major version.

    Columns added by a newer minor version are kept, after the known ones.
    """
    table = pq.read_table(path)
    metadata = table.schema.metadata or {}
    version = metadata.get(META_SCHEMA_VERSION.encode())
    if version is None:
        raise SchemaError(f"{path}: no {META_SCHEMA_VERSION} metadata, so not written by SPAR")
    check_compatible(version.decode())
    missing = [column for column in schema.names if column not in table.column_names]
    if missing:
        raise SchemaError(f"{path}: missing columns {missing}")
    try:
        known = table.select(schema.names).cast(schema)
    except pa.ArrowException as exc:
        raise SchemaError(f"{path}: {exc}") from exc
    for column in table.column_names:
        if column not in schema.names:
            known = known.append_column(table.schema.field(column), table.column(column))
    return known.replace_schema_metadata(metadata)


class EpisodeWriter:
    """Writes one episode folder, checking every file against the episode format and the scene.

    episode.json is written last, when the writer closes without an error, so a folder without it
    is an unfinished episode, and EpisodeReader refuses it.

        with EpisodeWriter(out_dir, scene, meta) as w:
            w.write_tracks(tracks_table)
            w.write_outcome(outcome)

    exist_ok=True adds to or replaces files in an existing episode, for stages that extend it.
    allow_incomplete_scene=True permits a scene with TODO values, for development only.
    """

    def __init__(
        self,
        out_dir: str | Path,
        scene: Scene,
        meta: EpisodeMeta,
        *,
        exist_ok: bool = False,
        allow_incomplete_scene: bool = False,
    ) -> None:
        if not allow_incomplete_scene:
            scene.require_complete()
        self.out_dir = Path(out_dir)
        if self.out_dir.exists() and any(self.out_dir.iterdir()) and not exist_ok:
            raise FileExistsError(f"{self.out_dir} is not empty; pass exist_ok=True to update it")
        self.scene = scene
        self.meta = self._check_meta(meta)
        self._agents = set(self.meta.roster)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exc_type is None:
            self.close()

    def close(self) -> None:
        """Write episode.json, which marks the episode as complete."""
        self._write_text(META_FILE, self.meta.model_dump_json(indent=2))

    def write_tracks(self, table: pa.Table) -> None:
        self._write_table("tracks", table, {"agent_id": self._agents})

    def write_objects(self, table: pa.Table) -> None:
        classes = set(self.scene.object_classes)
        self._write_table("objects", table, {"cls": classes, "holder": self._agents})

    def write_events(self, table: pa.Table) -> None:
        self._write_table("events", table, {"agent_id": self._agents})

    def write_state(self, table: pa.Table) -> None:
        self._write_table("state", table, {"checkpoint": set(self.scene.checkpoints)})

    def write_actions(self, table: pa.Table) -> None:
        self._write_table("actions", table, {"agent_id": self._agents})

    def write_outcome(self, outcome: Outcome) -> None:
        self._write_text(OUTCOME_FILE, outcome.model_dump_json(indent=2))

    def write_strategy(self, record: StrategyRecord) -> None:
        if unknown := sorted(set(record.codes) - set(self.scene.alliances)):
            raise SchemaError(f"strategy: unknown alliances {unknown}")
        self._write_text(STRATEGY_FILE, json.dumps(record.to_json_dict(), indent=2))

    def _check_meta(self, meta: EpisodeMeta) -> EpisodeMeta:
        scene = self.scene
        if (meta.scene, meta.rules_version) != (scene.name, scene.rules_version):
            raise SchemaError(
                f"episode is for scene {meta.scene!r} under {meta.rules_version!r}, but the "
                f"writer has {scene.name!r} under {scene.rules_version!r}"
            )
        if meta.program not in scene.programs:
            raise SchemaError(
                f"unknown program {meta.program!r}, expected one of {list(scene.programs)}"
            )
        if unknown := sorted({e.alliance for e in meta.roster.values()} - set(scene.alliances)):
            raise SchemaError(f"roster: unknown alliances {unknown}")
        if unknown := sorted(set(meta.score_check) - set(scene.checkpoints)):
            raise SchemaError(f"score_check: unknown checkpoints {unknown}")
        return meta.model_copy(update={"schema_version": SCHEMA_VERSION})

    def _write_table(self, name: str, table: pa.Table, allowed: dict[str, set[str]]) -> None:
        spec = TABLES[name]
        enums = {column: {member.value for member in enum} for column, enum in spec.enums.items()}
        table = validate_table(table, spec.schema, name=name, allowed=enums | allowed)
        metadata = file_metadata(
            self.scene.name, self.scene.rules_version, self.meta.pipeline.git_commit
        )
        pq.write_table(table.replace_schema_metadata(metadata), self.out_dir / spec.filename)

    def _write_text(self, filename: str, text: str) -> None:
        (self.out_dir / filename).write_text(text + "\n", encoding="utf-8")


class EpisodeReader:
    """Reads one episode folder.

        ep = EpisodeReader(path)
        ep.meta.roster, ep.tracks, ep.outcome

    Each file loads on first access, and is None when the episode does not have it.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        meta_path = self.path / META_FILE
        if not meta_path.is_file():
            raise FileNotFoundError(f"{meta_path} not found: not an episode, or an unfinished one")
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
        check_compatible(str(raw.get("schema_version", "")))
        self.meta = EpisodeMeta.model_validate(raw)

    @cached_property
    def tracks(self) -> pa.Table | None:
        return self._read_table("tracks")

    @cached_property
    def objects(self) -> pa.Table | None:
        return self._read_table("objects")

    @cached_property
    def events(self) -> pa.Table | None:
        return self._read_table("events")

    @cached_property
    def state(self) -> pa.Table | None:
        return self._read_table("state")

    @cached_property
    def actions(self) -> pa.Table | None:
        return self._read_table("actions")

    @cached_property
    def outcome(self) -> Outcome | None:
        path = self.path / OUTCOME_FILE
        return (
            Outcome.model_validate_json(path.read_text(encoding="utf-8"))
            if path.is_file()
            else None
        )

    @cached_property
    def strategy(self) -> StrategyRecord | None:
        path = self.path / STRATEGY_FILE
        if not path.is_file():
            return None
        return StrategyRecord.from_json_dict(json.loads(path.read_text(encoding="utf-8")))

    def _read_table(self, name: str) -> pa.Table | None:
        spec = TABLES[name]
        path = self.path / spec.filename
        return read_table(path, spec.schema) if path.is_file() else None
