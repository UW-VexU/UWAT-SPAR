"""Pre-registration manifests and the training data cutoff guard (spec section 6.9).

Before any footage recorded after the cutoff is processed, the strategy code definition, its
distance and threshold, and the ranked candidate list are frozen into a manifest in
$SPAR_DATA/prereg/ by `spar eval seal`. This module holds the file format and the check every
stage runs before it processes footage:

    policy = CutoffPolicy.load("configs/eval/prereg.yaml")
    require_processable(event_date, policy)  # CutoffError after the cutoff, until sealed

A manifest is <name>.manifest.json next to <name>.manifest.json.sha256, which holds its SHA-256.
It counts as sealed only while the two still match.
"""

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Self

import yaml

MANIFEST_SUFFIX = ".manifest.json"
HASH_SUFFIX = ".sha256"


class ManifestError(ValueError):
    """A manifest is missing its hash file, or no longer matches it."""


class CutoffError(RuntimeError):
    """Footage was recorded after the cutoff, and the candidate list is not sealed yet."""


@dataclass(frozen=True)
class CutoffPolicy:
    """Which scene's footage the cutoff guards, and the last day usable before sealing."""

    scene: str
    cutoff: date

    @classmethod
    def load(cls, path: str | Path) -> Self:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(scene=str(data["scene"]), cutoff=as_date(data["cutoff"]))


def default_prereg_dir() -> Path:
    """$SPAR_DATA/prereg."""
    root = os.environ.get("SPAR_DATA")
    if not root:
        raise RuntimeError("SPAR_DATA is not set; point it at the data root (spec section 5.6)")
    return Path(root) / "prereg"


def manifest_bytes(manifest: dict[str, Any]) -> bytes:
    """The exact bytes written for a manifest: sorted keys, so the same content hashes the same."""
    return (json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_path(manifest_path: str | Path) -> Path:
    return Path(str(manifest_path) + HASH_SUFFIX)


def check_manifest(path: str | Path) -> str:
    """A manifest's SHA-256, if it still matches its .sha256 file. Raises ManifestError if not."""
    path = Path(path)
    if not path.is_file():
        raise ManifestError(f"{path} not found")
    recorded_path = hash_path(path)
    if not recorded_path.is_file():
        raise ManifestError(f"{recorded_path} not found, so {path.name} was never sealed")
    recorded = recorded_path.read_text(encoding="utf-8").split()[0]
    actual = sha256_hex(path.read_bytes())
    if actual != recorded:
        raise ManifestError(
            f"{path.name} changed after sealing: its sha256 is {actual}, sealed as {recorded}"
        )
    return actual


def read_manifest(path: str | Path) -> dict[str, Any]:
    check_manifest(path)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sealed_manifest(prereg_dir: str | Path, scene: str) -> Path | None:
    """The first manifest for `scene` in prereg_dir that still matches its hash, or None."""
    for path in sorted(Path(prereg_dir).glob(f"*{MANIFEST_SUFFIX}")):
        try:
            if read_manifest(path).get("scene") == scene:
                return path
        except (ManifestError, json.JSONDecodeError):
            continue
    return None


def require_processable(
    recorded_on: date | datetime | str,
    policy: CutoffPolicy,
    prereg_dir: str | Path | None = None,
) -> None:
    """Raise CutoffError unless footage recorded on `recorded_on` may be processed now.

    Footage up to and including the cutoff day is always allowed. Later footage is allowed once a
    sealed manifest for the policy's scene is in prereg_dir ($SPAR_DATA/prereg by default).
    Downloading is never blocked; call this before cutting, labeling or any other processing.
    """
    day = as_date(recorded_on)
    if day <= policy.cutoff:
        return
    prereg_dir = Path(prereg_dir) if prereg_dir is not None else default_prereg_dir()
    if sealed_manifest(prereg_dir, policy.scene) is None:
        raise CutoffError(
            f"footage from {day} is after the {policy.cutoff} cutoff, and {prereg_dir} has no "
            f"sealed {policy.scene!r} manifest. Run `spar eval seal` first."
        )


def as_date(value: date | datetime | str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))
