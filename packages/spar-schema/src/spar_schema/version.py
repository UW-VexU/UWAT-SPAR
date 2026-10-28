"""Schema version and compatibility checks (spec section 5.7).

spar-schema uses semantic versions. Adding a column is a minor bump; renaming or removing one is a
major bump. Readers accept data with the same major version as this code and refuse the rest.
"""

SCHEMA_VERSION = "0.1.0"

# Keys in the metadata of every Parquet file SPAR writes.
META_SCHEMA_VERSION = "spar.schema_version"
META_SCENE = "spar.scene"
META_RULES_VERSION = "spar.rules_version"
META_GIT_COMMIT = "spar.git_commit"


class SchemaVersionError(ValueError):
    """Data was written under a schema major version this code does not know."""


def parse_version(version: str) -> tuple[int, int, int]:
    """Split "MAJOR.MINOR.PATCH" into three integers."""
    parts = version.split(".")
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        raise ValueError(f"not a MAJOR.MINOR.PATCH version: {version!r}")
    major, minor, patch = (int(part) for part in parts)
    return major, minor, patch


def check_compatible(version: str) -> None:
    """Raise SchemaVersionError unless data written under `version` can be read by this code."""
    ours = parse_version(SCHEMA_VERSION)[0]
    if parse_version(version)[0] != ours:
        raise SchemaVersionError(
            f"data uses schema version {version}, but this spar-schema ({SCHEMA_VERSION}) only "
            f"reads major version {ours}"
        )


def file_metadata(scene: str, rules_version: str, git_commit: str | None = None) -> dict[str, str]:
    """The metadata recorded in every Parquet file SPAR writes."""
    metadata = {
        META_SCHEMA_VERSION: SCHEMA_VERSION,
        META_SCENE: scene,
        META_RULES_VERSION: rules_version,
    }
    if git_commit:
        metadata[META_GIT_COMMIT] = git_commit
    return metadata
