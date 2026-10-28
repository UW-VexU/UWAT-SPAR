"""EpisodeWriter and EpisodeReader: round trip, validation and version checks."""

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from spar_schema import (
    TABLES,
    EpisodeReader,
    EpisodeWriter,
    Outcome,
    RosterEntry,
    SceneIncompleteError,
    SchemaError,
    SchemaVersionError,
    StrategyRecord,
    load_scene,
)

OUTCOME = Outcome(
    points_red=1,
    points_blue=0,
    autonomous_winner="red",
    sources={"final": "events_api", "autonomous": "stream_overlay"},
)
STRATEGY = StrategyRecord(
    codes={"red": {"first_target": ["G1"]}, "blue": {"first_target": []}},
    coder="pipeline",
    code_version="0.1",
)


def write_episode(path, scene, meta, tables):
    with EpisodeWriter(path, scene, meta) as w:
        for name, table in tables.items():
            getattr(w, f"write_{name}")(table)
        w.write_outcome(OUTCOME)
        w.write_strategy(STRATEGY)


def replace(table, column, values):
    return table.set_column(table.schema.get_field_index(column), column, pa.array(values))


def test_round_trip(tmp_path, scene, meta, tables):
    write_episode(tmp_path / "ep", scene, meta, tables)
    ep = EpisodeReader(tmp_path / "ep")
    assert ep.meta == meta
    for name, table in tables.items():
        assert getattr(ep, name).equals(table.cast(TABLES[name].schema)), name
    assert ep.outcome == OUTCOME
    assert ep.strategy == STRATEGY
    assert json.loads((tmp_path / "ep" / "strategy.json").read_text())["coder"] == "pipeline"


def test_absent_files_read_as_none(tmp_path, scene, meta, tables):
    with EpisodeWriter(tmp_path / "ep", scene, meta) as w:
        w.write_tracks(tables["tracks"])
    ep = EpisodeReader(tmp_path / "ep")
    assert ep.tracks.num_rows == 3
    assert ep.events is None
    assert ep.outcome is None
    assert ep.strategy is None


def test_unfinished_episode_is_refused(tmp_path, scene, meta, tables):
    with pytest.raises(RuntimeError), EpisodeWriter(tmp_path / "ep", scene, meta) as w:
        w.write_tracks(tables["tracks"])
        raise RuntimeError("stage failed")
    assert (tmp_path / "ep" / "tracks.parquet").exists()
    with pytest.raises(FileNotFoundError):
        EpisodeReader(tmp_path / "ep")


def test_existing_episode_needs_exist_ok(tmp_path, scene, meta, tables):
    write_episode(tmp_path / "ep", scene, meta, tables)
    with pytest.raises(FileExistsError):
        EpisodeWriter(tmp_path / "ep", scene, meta)
    with EpisodeWriter(tmp_path / "ep", scene, meta, exist_ok=True) as w:
        w.write_events(tables["events"])
    assert EpisodeReader(tmp_path / "ep").tracks is not None


BAD_TABLES = {
    "missing column": ("tracks", lambda t: t.drop_columns(["conf"]), "missing columns"),
    "unexpected column": (
        "tracks",
        lambda t: t.append_column("speed", pa.array([0.0, 0.0, 0.0])),
        "unexpected columns",
    ),
    "null in required column": ("tracks", lambda t: replace(t, "x", [0.0, None, 0.0]), "nulls"),
    "uncastable value": ("tracks", lambda t: replace(t, "x", ["a", "b", "c"]), "tracks"),
    "unknown provenance": (
        "tracks",
        lambda t: replace(t, "provenance", ["detected", "guessed", "detected"]),
        "guessed",
    ),
    "unknown agent": (
        "tracks",
        lambda t: replace(t, "agent_id", ["red_1", "red_9", "red_1"]),
        "red_9",
    ),
    "unknown object class": ("objects", lambda t: replace(t, "cls", ["ball", "cup"]), "cup"),
    "unknown holder": ("objects", lambda t: replace(t, "holder", [None, "green_1"]), "green_1"),
    "unknown event source": ("events", lambda t: replace(t, "source", ["guess"]), "guess"),
    "unknown checkpoint": (
        "state",
        lambda t: replace(t, "checkpoint", [None, "halftime"]),
        "halftime",
    ),
}


@pytest.mark.parametrize("name, mutate, message", BAD_TABLES.values(), ids=list(BAD_TABLES))
def test_writer_rejects_bad_tables(tmp_path, scene, meta, tables, name, mutate, message):
    writer = EpisodeWriter(tmp_path / "ep", scene, meta)
    with pytest.raises(SchemaError, match=message):
        getattr(writer, f"write_{name}")(mutate(tables[name]))


def test_meta_must_match_scene(tmp_path, scene, meta):
    for update in [
        {"scene": "other"},
        {"rules_version": "v2"},
        {"program": "duo"},
        {"roster": {"green_1": RosterEntry(alliance="green")}},
    ]:
        with pytest.raises(SchemaError):
            EpisodeWriter(tmp_path / "ep", scene, meta.model_copy(update=update))


def test_strategy_alliances_must_match_scene(tmp_path, scene, meta):
    record = StrategyRecord(codes={"green": {}}, coder="human:lx", code_version="0.1")
    with pytest.raises(SchemaError, match="green"):
        EpisodeWriter(tmp_path / "ep", scene, meta).write_strategy(record)


def test_incomplete_scene_is_refused(tmp_path, toy_scene_dir, meta, edit):
    edit(toy_scene_dir / "scene.yaml", "driver_s: 50", "driver_s: TODO")
    draft = load_scene(toy_scene_dir)
    with pytest.raises(SceneIncompleteError):
        EpisodeWriter(tmp_path / "ep", draft, meta)
    EpisodeWriter(tmp_path / "ep", draft, meta, allow_incomplete_scene=True).close()


def test_unknown_major_in_meta_is_refused(tmp_path, scene, meta, tables, edit):
    write_episode(tmp_path / "ep", scene, meta, tables)
    edit(tmp_path / "ep" / "episode.json", '"schema_version": "0.1.0"', '"schema_version": "1.0.0"')
    with pytest.raises(SchemaVersionError):
        EpisodeReader(tmp_path / "ep")


def rewrite_tracks(path, version, add_column=False):
    """Rewrite tracks.parquet as if a different schema version had written it."""
    table = pq.read_table(path)
    if add_column:
        table = table.append_column("speed", pa.array([1.0] * table.num_rows))
    metadata = {**table.schema.metadata, b"spar.schema_version": version.encode()}
    pq.write_table(table.replace_schema_metadata(metadata), path)


def test_unknown_major_in_table_is_refused(tmp_path, scene, meta, tables):
    write_episode(tmp_path / "ep", scene, meta, tables)
    rewrite_tracks(tmp_path / "ep" / "tracks.parquet", "1.0.0")
    with pytest.raises(SchemaVersionError):
        _ = EpisodeReader(tmp_path / "ep").tracks


def test_newer_minor_columns_are_kept(tmp_path, scene, meta, tables):
    write_episode(tmp_path / "ep", scene, meta, tables)
    rewrite_tracks(tmp_path / "ep" / "tracks.parquet", "0.2.0", add_column=True)
    tracks = EpisodeReader(tmp_path / "ep").tracks
    assert tracks.column_names == [*TABLES["tracks"].schema.names, "speed"]
