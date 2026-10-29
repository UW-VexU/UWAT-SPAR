"""Index of all episodes with filters (scene, rules version, program, score check result, date).

scan() walks a folder for episodes (normally $SPAR_DATA/episodes) and reads each one's metadata.
Catalog.filter() narrows the list; Catalog.table() gives one row per episode, for a quick look or
to save as a Parquet file. Each episode's score check sums up as:
    passed     at least one checkpoint passed and none failed
    failed     at least one checkpoint failed
    unchecked  spar state run has not checked it
"""

import os
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pyarrow as pa

from spar_schema import EpisodeMeta, EpisodeReader, SchemaVersionError


@dataclass(frozen=True)
class Entry:
    path: Path
    meta: EpisodeMeta

    @property
    def score_check(self) -> str:
        results = [check.passed for check in self.meta.score_check.values()]
        if any(result is False for result in results):
            return "failed"
        if any(result is True for result in results):
            return "passed"
        return "unchecked"


class Catalog:
    def __init__(self, entries: list[Entry], skipped: list[tuple[Path, str]] | None = None):
        self.entries = sorted(entries, key=lambda entry: entry.meta.episode_id)
        self.skipped = skipped or []  # (folder, reason) for episodes that could not be read

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[Entry]:
        return iter(self.entries)

    def filter(
        self,
        *,
        scene: str | None = None,
        rules_version: str | None = None,
        program: str | None = None,
        kind: str | None = None,
        event_sku: str | None = None,
        score_check: str | None = None,
        played_from: date | None = None,
        played_to: date | None = None,
    ) -> "Catalog":
        """The episodes that match every filter given. Date filters include both ends and drop
        episodes without a match_date."""
        keep = []
        for entry in self.entries:
            meta = entry.meta
            if scene is not None and meta.scene != scene:
                continue
            if rules_version is not None and meta.rules_version != rules_version:
                continue
            if program is not None and meta.program != program:
                continue
            if kind is not None and meta.kind != kind:
                continue
            if event_sku is not None and meta.event_sku != event_sku:
                continue
            if score_check is not None and entry.score_check != score_check:
                continue
            if (played_from or played_to) and meta.match_date is None:
                continue
            if played_from is not None and meta.match_date < played_from:
                continue
            if played_to is not None and meta.match_date > played_to:
                continue
            keep.append(entry)
        return Catalog(keep, self.skipped)

    def counts(self, *fields: str) -> Counter:
        """How many episodes share each combination of the given fields (or "score_check")."""
        return Counter(
            tuple(
                entry.score_check if f == "score_check" else getattr(entry.meta, f) for f in fields
            )
            for entry in self.entries
        )

    def table(self) -> pa.Table:
        rows = [
            {
                "episode_id": e.meta.episode_id,
                "kind": str(e.meta.kind),
                "scene": e.meta.scene,
                "rules_version": e.meta.rules_version,
                "program": e.meta.program,
                "event_sku": e.meta.event_sku,
                "match_name": e.meta.match_name,
                "match_date": e.meta.match_date,
                "score_check": e.score_check,
                "path": str(e.path),
            }
            for e in self.entries
        ]
        schema = pa.schema(
            [
                ("episode_id", pa.string()),
                ("kind", pa.string()),
                ("scene", pa.string()),
                ("rules_version", pa.string()),
                ("program", pa.string()),
                ("event_sku", pa.string()),
                ("match_name", pa.string()),
                ("match_date", pa.date32()),
                ("score_check", pa.string()),
                ("path", pa.string()),
            ]
        )
        return pa.Table.from_pylist(rows, schema=schema)


def scan(root: str | Path) -> Catalog:
    """Every episode under root, by its episode.json. Unreadable episodes are listed as skipped."""
    entries, skipped = [], []
    for meta_file in sorted(Path(root).rglob("episode.json")):
        folder = meta_file.parent
        try:
            entries.append(Entry(path=folder, meta=EpisodeReader(folder).meta))
        except (SchemaVersionError, ValueError) as exc:
            skipped.append((folder, str(exc)))
    return Catalog(entries, skipped)


def default_episodes_dir() -> Path:
    """$SPAR_DATA/episodes."""
    root = os.environ.get("SPAR_DATA")
    if not root:
        raise RuntimeError("SPAR_DATA is not set; point it at the data root (spec section 5.6)")
    return Path(root) / "episodes"
